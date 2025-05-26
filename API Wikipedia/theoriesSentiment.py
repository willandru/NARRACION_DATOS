import requests
from bs4 import BeautifulSoup
import csv
from textblob import TextBlob
import textstat

API_URL = "https://en.wikipedia.org/w/api.php"
TITLE = "Theoretical physics"
TARGET_SECTIONS = ["Mainstream theories", "Proposed theories", "Fringe theories"]

def get_section_index(title):
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "sections"
    }
    response = requests.get(API_URL, params=params).json()
    section_map = {}
    for section in response["parse"]["sections"]:
        name = section["line"].strip()
        index = section["index"]
        if name in TARGET_SECTIONS:
            section_map[name] = index
    return section_map

def get_section_html(title, index):
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "text",
        "section": index
    }
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
    return sorted(list(links))  # sorted for consistency

def extract_lead_section(title):
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "text",
        "redirects": True
    }
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

def analyze_text(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    readability = textstat.flesch_reading_ease(text)
    return polarity, subjectivity, readability

# === Main pipeline ===
section_indices = get_section_index(TITLE)
all_labels = set()

for section_name in TARGET_SECTIONS:
    index = section_indices.get(section_name)
    if not index:
        continue
    html = get_section_html(TITLE, index)
    section_links = extract_links_from_html(html)
    all_labels.update(section_links)  # avoid duplicates

all_results = []
seen_titles = set()

for label in sorted(all_labels):
    if label in seen_titles:
        continue
    seen_titles.add(label)
    try:
        text = extract_lead_section(label)
        if not text.strip():
            raise Exception("Empty lead section or redirect with no intro.")
        print(f"\n--- {label} ---")
        print(text[:700] + ("\n[...] (truncated)" if len(text) > 700 else ""))
        polarity, subjectivity, readability = analyze_text(text)
        all_results.append({
            "Theory": label,
            "Polarity": polarity,
            "Subjectivity": subjectivity,
            "Readability": readability
        })
    except Exception as e:
        print(f"[Error processing {label}] {e}")
        continue

# Save results
with open("theory3_sentiment_metrics.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Theory", "Polarity", "Subjectivity", "Readability"])
    writer.writeheader()
    for row in all_results:
        writer.writerow(row)
