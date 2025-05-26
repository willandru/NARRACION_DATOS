import requests
from bs4 import BeautifulSoup
import csv
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# === CONFIG ===
API_URL = "https://en.wikipedia.org/w/api.php"
TITLE = "Theoretical physics"
TARGET_SECTIONS = ["Mainstream theories", "Proposed theories", "Fringe theories"]
EXCLUDED_SECTIONS = {"See also", "References", "Further reading", "External links", "Bibliography", "Notes"}

# === MODELS ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
ner_pipeline = pipeline("ner", model="allenai/scibert_scivocab_uncased", aggregation_strategy="simple")

# === UTILITIES ===

def preprocess_text(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\{\\displaystyle.*?\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\[\s*edit\s*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Main article|See also|Further reading):.*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    return re.sub(r'\s+', ' ', soup.get_text(" ", strip=True))

def get_sentiment_score(text):
    try:
        result = sentiment_pipeline(text[:512])[0]
        score = result["score"]
        return score if result["label"] == "POSITIVE" else -score
    except:
        return 0.0

def get_embedding(text):
    return embedder.encode(text)

def calculate_readability(text):
    sentence_count = text.count('.') or 1
    word_count = len(text.split())
    alpha_count = sum(map(str.isalpha, text))
    return float(206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (alpha_count / word_count))

def analyze_text(text):
    text = preprocess_text(text)
    polarity = get_sentiment_score(text)
    embedding = get_embedding(text)
    subjectivity = float(np.std(embedding))
    readability = calculate_readability(text)
    return polarity, subjectivity, readability

# === AUTHOR DETECTION ===

def clean_author_name(name):
    name = name.strip()
    if name.startswith("##") or len(name) < 3 or re.search(r'\d', name):
        return None
    if len(name.split()) > 5:
        return None
    return name.title()

def extract_people_ner(text):
    entities = ner_pipeline(text[:1000])
    return {clean_author_name(ent['word']) for ent in entities if ent['entity_group'] == "PER"}

def extract_people_regex(text):
    patterns = [
        r"proposed by ([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
        r"developed by ([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
        r"introduced by ([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
        r"formulated by ([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
        r"named after ([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
    ]
    people = set()
    for pat in patterns:
        matches = re.findall(pat, text)
        for m in matches:
            people.add(clean_author_name(m))
    return people

def get_authors(text):
    ner_people = extract_people_ner(text)
    regex_people = extract_people_regex(text)
    return {p for p in ner_people.union(regex_people) if p}

# === WIKIPEDIA FUNCTIONS ===

def get_section_index(title):
    params = {"action": "parse", "format": "json", "page": title, "prop": "sections"}
    response = requests.get(API_URL, params=params).json()
    return response["parse"]["sections"]

def get_section_text(title, index):
    params = {"action": "parse", "format": "json", "page": title, "prop": "text", "section": index}
    response = requests.get(API_URL, params=params).json()
    return html_to_text(response["parse"]["text"]["*"])

def extract_lead_section(title):
    params = {"action": "parse", "format": "json", "page": title, "prop": "text", "redirects": True}
    response = requests.get(API_URL, params=params).json()
    soup = BeautifulSoup(response["parse"]["text"]["*"], "html.parser")
    return " ".join(tag.get_text(" ", strip=True) for tag in soup.find_all(["p", "h2"]) if tag.name == "p")

def extract_all_sections_excluding(title):
    full_text = extract_lead_section(title) + " "
    try:
        for sec in get_section_index(title):
            if sec["line"].strip() in EXCLUDED_SECTIONS:
                continue
            full_text += get_section_text(title, sec["index"]) + " "
    except Exception as e:
        print(f"[Error extracting full article for {title}] {e}")
    return preprocess_text(full_text)

def get_section_html(title, index):
    params = {"action": "parse", "format": "json", "page": title, "prop": "text", "section": index}
    return requests.get(API_URL, params=params).json()["parse"]["text"]["*"]

def extract_links_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return sorted({a.get_text(strip=True) for a in soup.find_all("a", href=True) if a["href"].startswith("/wiki/") and not a["href"].startswith("/wiki/Special:")})

# === EXTRAER TEORÍAS DESDE "Theoretical physics" ===

section_indices = {s["line"].strip(): s["index"] for s in get_section_index(TITLE) if s["line"].strip() in TARGET_SECTIONS}
all_labels = set()

for sec_name, index in section_indices.items():
    html = get_section_html(TITLE, index)
    all_labels.update(extract_links_from_html(html))

# === PIPELINE PRINCIPAL ===

results = []
edges = []
seen = set()

for label in sorted(all_labels):
    if label in seen:
        continue
    seen.add(label)
    try:
        full_text = extract_all_sections_excluding(label)
        if not full_text.strip():
            raise Exception("Empty content")
        print(f"\n--- {label} ---")
        print(full_text[:700] + (" [...]" if len(full_text) > 700 else ""))

        polarity, subjectivity, readability = analyze_text(full_text)
        results.append({
            "Theory": label,
            "Polarity": polarity,
            "Subjectivity": subjectivity,
            "Readability": readability
        })

        authors = get_authors(full_text)
        for author in authors:
            edges.append((label, author))

    except Exception as e:
        print(f"[Error processing {label}] {e}")

# === GUARDAR CSVs ===

with open("theory8_sentiment_embeddings.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Theory", "Polarity", "Subjectivity", "Readability"])
    writer.writeheader()
    writer.writerows(results)

with open("theory8_author_bipartite.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Theory", "Author"])
    for edge in edges:
        writer.writerow(edge)
