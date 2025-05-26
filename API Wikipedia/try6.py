# Este script mejora la detección de autores asociados a teorías científicas
# usando NER extendido y consultas avanzadas a Wikidata para propiedades como P50, P61, etc.

import requests
from bs4 import BeautifulSoup
import csv
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# === CONFIGURACION ===
API_URL = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
TITLE = "Theoretical physics"
TARGET_SECTIONS = ["Mainstream theories", "Proposed theories", "Fringe theories"]
EXCLUDED_SECTIONS = {"See also", "References", "Further reading", "External links", "Bibliography", "Notes"}

# === MODELOS ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
ner_pipeline = pipeline("ner", model="allenai/scibert_scivocab_uncased", aggregation_strategy="simple")

# === FUNCIONES DE LIMPIEZA ===
def preprocess_text(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\{\\displaystyle.*?\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\[\s*edit\s*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Main article|See also|Further reading):.*', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    return re.sub(r'\s+', ' ', soup.get_text(" ", strip=True))

# === METRICAS ===
def get_sentiment_score(text):
    result = sentiment_pipeline(text[:512])[0]
    return result['score'] if result['label'] == 'POSITIVE' else -result['score']

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

# === NER AVANZADO ===
def extract_people_ner(text):
    entities = []
    chunk_size = 800
    for i in range(0, len(text), chunk_size):
        entities += ner_pipeline(text[i:i+chunk_size])
    return {ent['word'].strip().title() for ent in entities if ent['entity_group'] == "PER" and len(ent['word']) > 2}

# === WIKIDATA ===
def get_wikidata_id(title):
    params = {"action": "query", "format": "json", "titles": title, "prop": "pageprops"}
    res = requests.get(API_URL, params=params).json()
    page = next(iter(res['query']['pages'].values()))
    return page['pageprops'].get('wikibase_item') if 'pageprops' in page else None

def get_authors_from_wikidata(wikidata_id):
    people = set()
    props = ['P50', 'P61', 'P737']
    for prop in props:
        params = {"action": "wbgetclaims", "format": "json", "entity": wikidata_id, "property": prop}
        res = requests.get(WIKIDATA_API, params=params).json()
        if prop in res.get("claims", {}):
            for claim in res["claims"][prop]:
                if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
                    qid = claim["mainsnak"]["datavalue"]["value"]["id"]
                    name = get_label_from_qid(qid)
                    if name:
                        people.add(name)
    return people

def get_label_from_qid(qid):
    params = {"action": "wbgetentities", "format": "json", "ids": qid, "props": "labels", "languages": "en"}
    res = requests.get(WIKIDATA_API, params=params).json()
    return res.get("entities", {}).get(qid, {}).get("labels", {}).get("en", {}).get("value")

# === WIKIPEDIA ===
def get_section_index(title):
    params = {"action": "parse", "format": "json", "page": title, "prop": "sections"}
    return requests.get(API_URL, params=params).json()["parse"]["sections"]

def get_section_text(title, index):
    params = {"action": "parse", "format": "json", "page": title, "prop": "text", "section": index}
    html = requests.get(API_URL, params=params).json()["parse"]["text"]["*"]
    return html_to_text(html)

def extract_lead_section(title):
    params = {"action": "parse", "format": "json", "page": title, "prop": "text", "redirects": True}
    html = requests.get(API_URL, params=params).json()["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")
    return " ".join(tag.get_text(" ", strip=True) for tag in soup.find_all("p"))

def extract_all_sections(title):
    text = extract_lead_section(title) + " "
    for sec in get_section_index(title):
        if sec['line'].strip() not in EXCLUDED_SECTIONS:
            text += get_section_text(title, sec['index']) + " "
    return preprocess_text(text)

def get_section_html(title, index):
    params = {"action": "parse", "format": "json", "page": title, "prop": "text", "section": index}
    return requests.get(API_URL, params=params).json()["parse"]["text"]["*"]

def extract_links_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return sorted({a.get_text(strip=True) for a in soup.find_all("a", href=True) if a['href'].startswith("/wiki/") and not a['href'].startswith("/wiki/Special:")})

# === PROCESAMIENTO PRINCIPAL ===
section_indices = {s['line'].strip(): s['index'] for s in get_section_index(TITLE) if s['line'].strip() in TARGET_SECTIONS}
all_labels = set()
for section_name, index in section_indices.items():
    html = get_section_html(TITLE, index)
    all_labels.update(extract_links_from_html(html))

results, edges, seen = [], [], set()

for label in sorted(all_labels):
    if label in seen: continue
    seen.add(label)
    try:
        text = extract_all_sections(label)
        print(f"\n--- {label} ---\n{text[:300]}...")
        pol, subj, read = analyze_text(text)
        results.append({"Theory": label, "Polarity": pol, "Subjectivity": subj, "Readability": read})

        authors = extract_people_ner(text)
        wikidata_id = get_wikidata_id(label)
        if wikidata_id:
            authors.update(get_authors_from_wikidata(wikidata_id))
        for author in authors:
            edges.append((label, author))

    except Exception as e:
        print(f"[Error {label}] {e}")

with open("theory_sentiment_embeddings.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Theory", "Polarity", "Subjectivity", "Readability"])
    writer.writeheader()
    writer.writerows(results)

with open("theory_author_bipartite.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Theory", "Author"])
    for row in edges:
        writer.writerow(row)
