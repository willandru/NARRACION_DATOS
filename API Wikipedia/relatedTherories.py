import requests
from bs4 import BeautifulSoup
import csv

API_URL = "https://en.wikipedia.org/w/api.php"
TITLE = "Theoretical physics"
TARGET_SECTIONS = ["Mainstream theories", "Proposed theories", "Fringe theories"]

# Paso 1: Obtener índices de secciones relevantes
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

# Paso 2: Obtener HTML de cada sección
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

# Paso 3: Extraer enlaces internos de cada sección
def extract_links_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        if href.startswith("/wiki/") and not href.startswith("/wiki/Special:"):
            full_url = f"https://en.wikipedia.org{href}"
            links.append((text, full_url))
    return links

# Paso 4: Extraer teorías desde secciones relevantes
section_indices = get_section_index(TITLE)
theory_links = []
for section_name in TARGET_SECTIONS:
    index = section_indices.get(section_name)
    if not index:
        continue
    html = get_section_html(TITLE, index)
    theory_links += extract_links_from_html(html)

# Paso 5: Construir aristas basadas en mención cruzada
edges = []
valid_theory_titles = set(label for label, _ in theory_links)

for label, url in theory_links:
    title = url.split("/wiki/")[1]
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "text"
    }
    try:
        response = requests.get(API_URL, params=params).json()
        html = response["parse"]["text"]["*"]
        soup = BeautifulSoup(html, "html.parser")

        # Eliminar contenido de secciones irrelevantes
        for h2 in soup.find_all("h2"):
            span = h2.find("span", class_="mw-headline")
            if span and span.text.strip() in ["See also", "References", "External links"]:
                for sibling in list(h2.find_next_siblings()):
                    sibling.decompose()

        for a in soup.find_all("a", href=True):
            linked_label = a.get_text(strip=True)
            if linked_label in valid_theory_titles:
                edges.append((label, linked_label))
    except Exception:
        continue

# Paso 6: Guardar CSV
with open("citation_edges.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["From", "To"])
    for source, target in edges:
        writer.writerow([source, target])
