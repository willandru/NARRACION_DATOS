import requests
from bs4 import BeautifulSoup
import csv
import re
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# === CONFIGURACIÓN ===
API_URL = "https://en.wikipedia.org/w/api.php"
TITLE = "Theoretical physics"
TARGET_SECTIONS = ["Mainstream theories", "Proposed theories", "Fringe theories"]
EXCLUDED_SECTIONS = {"See also", "References", "Further reading", "External links", "Bibliography", "Notes"}

# === MODELOS ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

# === FUNCIONES DE TEXTO Y MÉTRICAS ===

def preprocess_text(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
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

# === NER: Extracción de personas (autores) ===

def extract_people_ner(text):
    entities = ner_pipeline(text[:1000])  # truncado por eficiencia
    return {ent['word'].strip() for ent in entities if ent['entity_group'] == "PER"}

# === FUNCIONES WIKIPEDIA ===

def get_section_index(title):
    params = {"action": "parse", "format": "json", "page": title, "prop": "sections"}
    response = requests.get(API_URL, params=params).json()
    return response["parse"]["sections"]

def get_section_text(title, index):
    params = {"action": "parse", "format": "json", "page": title, "prop": "text", "section": index}
    response = requests.get(API_URL, params=params).json()
    html = response["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def get_lead_paragraphs(title):
    params = {"action": "parse", "format": "json", "page": title, "prop": "text", "redirects": True}
    response = requests.get(API_URL, params=params).json()
    html = response["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")
    lead_paragraphs = []
    for tag in soup.find_all(["p", "h2"]):
        if tag.name == "h2":
            break
        if tag.name == "p":
            lead_paragraphs.append(tag.get_text(strip=True))
    return " ".join(lead_paragraphs)

def get_full_article_text_excluding(title):
    try:
        sections = get_section_index(title)
        lead = get_lead_paragraphs(title)
        full_text = lead + " "
        for sec in sections:
            name = sec["line"].strip()
            index = sec["index"]
            if name not in EXCLUDED_SECTIONS:
                section_text = get_section_text(title, index)
                full_text += " " + section_text
        return preprocess_text(full_text)
    except Exception as e:
        print(f"[Error extracting full article for {title}] {e}")
        return ""

def extract_links_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        label = a.get_text(strip=True)
        if href.startswith("/wiki/") and not href.startswith("/wiki/Special:"):
            links.add(label)
    return sorted(list(links))

def get_section_html(title, index):
    params = {"action": "parse", "format": "json", "page": title, "prop": "text", "section": index}
    response = requests.get(API_URL, params=params).json()
    return response["parse"]["text"]["*"]

# === EXTRAER TEORÍAS DESDE SECCIONES TARGET ===

params = {"action": "parse", "format": "json", "page": TITLE, "prop": "sections"}
response = requests.get(API_URL, params=params).json()
section_indices = {sec["line"].strip(): sec["index"] for sec in response["parse"]["sections"] if sec["line"].strip() in TARGET_SECTIONS}

all_labels = set()
for section_name in TARGET_SECTIONS:
    index = section_indices.get(section_name)
    if not index:
        continue
    html = get_section_html(TITLE, index)
    section_links = extract_links_from_html(html)
    all_labels.update(section_links)

# === PROCESAR CADA TEORÍA ===

all_results = []
bipartite_edges = []
seen_titles = set()

for label in sorted(all_labels):
    if label in seen_titles:
        continue
    seen_titles.add(label)
    try:
        full_text = get_full_article_text_excluding(label)
        if not full_text.strip():
            raise Exception("No usable text.")
        print(f"\n--- {label} ---")
        print(full_text[:600] + " [...]" if len(full_text) > 600 else full_text)

        polarity, subjectivity, readability = analyze_text(full_text)
        all_results.append({
            "Theory": label,
            "Polarity": polarity,
            "Subjectivity": subjectivity,
            "Readability": readability
        })

        found_people = extract_people_ner(full_text)
        for person in found_people:
            bipartite_edges.append((label, person))

    except Exception as e:
        print(f"[Error processing {label}] {e}")
        continue

# === GUARDAR CSV DE MÉTRICAS Y DE GRAFO BIPARTITO ===

with open("theory6_sentiment_embeddings.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Theory", "Polarity", "Subjectivity", "Readability"])
    writer.writeheader()
    for row in all_results:
        writer.writerow(row)

with open("theory6_author_bipartite.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Theory", "Author"])
    for theory, author in bipartite_edges:
        writer.writerow([theory, author])
