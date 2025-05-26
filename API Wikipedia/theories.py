import requests
from bs4 import BeautifulSoup

API_URL = "https://en.wikipedia.org/w/api.php"
TITLE = "Theoretical physics"
TARGET_SECTIONS = ["Mainstream theories", "Proposed theories", "Fringe theories"]

# Step 1: Get list of all sections
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

# Step 2: Get full HTML of a section
def get_section_html(title, index):
    params = {
        "action": "parse",
        "format": "json",
        "page": title,
        "prop": "text",
        "section": index
    }
    response = requests.get(API_URL, params=params).json()
    html = response["parse"]["text"]["*"]
    return html

# Step 3: Extract links from HTML
def extract_links_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        if href.startswith("/wiki/") and not href.startswith("/wiki/Special:"):
            links.append((text, f"https://en.wikipedia.org{href}"))
    return links

# Run the extraction
section_indices = get_section_index(TITLE)
for section_name in TARGET_SECTIONS:
    print(f"\n=== {section_name} ===")
    index = section_indices.get(section_name)
    if not index:
        print("Section not found.")
        continue
    html = get_section_html(TITLE, index)
    links = extract_links_from_html(html)
    print(f"Found {len(links)} links.")
    for label, url in links:
        print(f" - {label}: {url}")
