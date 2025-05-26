import requests
from bs4 import BeautifulSoup
import csv
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

API_URL = "https://en.wikipedia.org/w/api.php"
TITLE = "Theoretical physics"
TARGET_SECTIONS = ["Mainstream theories", "Proposed theories", "Fringe theories"]
EXCLUDED_SECTIONS = {"See also", "References", "Further reading", "External links", "Bibliography", "Notes"}

# === MODELOS ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# === FUNCIONES UTILITARIAS ===

def preprocess_text(text):
    text = re.sub(r'\[\d+\]', '', text)           # remove [1], [2], etc.
    text = re.sub(r'\s+', ' ', text).strip()      # normalize whitespace
    return text

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

def extract_people_ner(text):
    entities = ner_pipeline(text[:1000])  # limit for performance
    return {ent['word'].strip() for ent in entities if ent['entity_group'] == "PER"}

# === FUNCIONES WIKIPEDIA ===

def get_section_index(title):
    params = {"action": "parse", "format": "json", "page": title, "prop": "sections"}
    response = requests.get(API_URL, params=params).json()
    section_map = {}
    for section in response["parse"]["sections"]:
        name = section["line"].strip()
        index = section["index"]
        if name in TARGET_SECTIONS:
            section_map[name] = index
    return section_map

def get_section_html(title, index):
    params = {"action": "parse", "format": "json", "page": title, "prop": "text", "section": index}
    response = requests.get(API_URL, params=params).json()
    return response["parse"]["text"]["*"]

def extract_links_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        label = a.get_text(strip=True)
        if href.startswith("/wiki/") and not href.startswith("/wiki/Special:"):
            links.add(label)
    return sorted(list(links))

def extract_lead_section(title):
    params = {"action": "parse", "format": "json", "page": title, "prop": "text", "redirects": True}
    response = requests.get(API_URL, params=params).json()
    if "error" in response:
        raise Exception(response["error"]["info"])
    html = response["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")
    lead_paragraphs = []
    for tag in soup.find_all(["p", "h2"]):
        if tag.name == "h2":
            break
        if tag.name == "p":
            lead_paragraphs.append(tag.get_text(strip=True))
    return " ".join(lead_paragraphs)

def extract_all_sections_excluding(title):
    full_text = extract_lead_section(title) + " "
    try:
        params = {"action": "parse", "format": "json", "page": title, "prop": "sections"}
        response = requests.get(API_URL, params=params).json()
        for sec in response["parse"]["sections"]:
            name = sec["line"].strip()
            if name in EXCLUDED_SECTIONS:
                continue
            index = sec["index"]
            params = {"action": "parse", "format": "json", "page": title, "prop": "text", "section": index}
            sec_html = requests.get(API_URL, params=params).json()
            html = sec_html["parse"]["text"]["*"]
            soup = BeautifulSoup(html, "html.parser")
            full_text += soup.get_text(separator=" ", strip=True) + " "
    except Exception as e:
        print(f"[Error extracting all sections for {title}] {e}")
    return preprocess_text(full_text)

# === PIPELINE PRINCIPAL ===

section_indices = get_section_index(TITLE)
all_labels = set()

for section_name in TARGET_SECTIONS:
    index = section_indices.get(section_name)
    if not index:
        continue
    html = get_section_html(TITLE, index)
    section_links = extract_links_from_html(html)
    all_labels.update(section_links)

all_results = []
bipartite_edges = []
seen_titles = set()

for label in sorted(all_labels):
    if label in seen_titles:
        continue
    seen_titles.add(label)
    try:
        full_text = extract_all_sections_excluding(label)
        if not full_text.strip():
            raise Exception("Empty or invalid content.")
        print(f"\n--- {label} ---")
        print(full_text[:700] + ("\n[...] (truncated)" if len(full_text) > 700 else ""))

        polarity, subjectivity, readability = analyze_text(full_text)
        all_results.append({
            "Theory": label,
            "Polarity": polarity,
            "Subjectivity": subjectivity,
            "Readability": readability
        })

        found_authors = extract_people_ner(full_text)
        for author in found_authors:
            bipartite_edges.append((label, author))

    except Exception as e:
        print(f"[Error processing {label}] {e}")
        continue

# === GUARDAR CSV ===

with open("theory4_sentiment_embeddings.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Theory", "Polarity", "Subjectivity", "Readability"])
    writer.writeheader()
    for row in all_results:
        writer.writerow(row)

with open("theory_author_bipartite.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Theory", "Author"])
    for theory, author in bipartite_edges:
        writer.writerow([theory, author])
