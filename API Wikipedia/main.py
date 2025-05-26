
############################################################################################
# GET THE FIRST PARAGRAPH OF A SPECIFIC TOPIC
###########################################################################################

import wikipedia

wikipedia.set_lang("en")  # or 'es', 'fr', etc.
summary = wikipedia.summary("Theoretical physics", sentences=2)
print(summary)

page = wikipedia.page("Theoretical physics")
print("Title:", page.title)
print("URL:", page.url)
print("Content length:", len(page.content))

############################################################################################
# GET THE WORDS WITH HIPERLINKS
###########################################################################################
import wikipedia

wikipedia.set_lang("en")  # Set language, e.g. "en" for English

page = wikipedia.page("Theoretical physics")

# Get list of links in the page
links = page.links

print(f"Found {len(links)} links.")
for link in links[:10]:  # Show first 10 links
    print(link)


import requests

API_URL = "https://en.wikipedia.org/w/api.php"
TITLE = "Theoretical physics"

############################################################################################
# Get plain text
###########################################################################################
def get_intro_text(title):
    params = {
        "action": "query",
        "prop": "extracts",
        "format": "json",
        "titles": title,
        "explaintext": True,
        "exintro": True
    }
    r = requests.get(API_URL, params=params)
    page = next(iter(r.json()["query"]["pages"].values()))
    return page["extract"]

print('GET PLAIN TEXT: ',get_intro_text(TITLE))




############################################################################################
#Get all internal links (to other Wikipedia pages)
###########################################################################################

def get_internal_links(title):
    links = []
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "links",
        "pllimit": "max"  # max per request
    }

    while True:
        r = requests.get(API_URL, params=params).json()
        page = next(iter(r["query"]["pages"].values()))
        links.extend([link["title"] for link in page.get("links", [])])

        if "continue" in r:
            params.update(r["continue"])
        else:
            break

    return links

links = get_internal_links(TITLE)
print("Get all internal links (to other Wikipedia pages)")
print(f"Found {len(links)} internal links")
print(links[:10])  # sample



############################################################################################
#Get categories of the article
###########################################################################################

def get_categories(title):
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "categories",
        "cllimit": "max"
    }
    r = requests.get(API_URL, params=params)
    page = next(iter(r.json()["query"]["pages"].values()))
    return [cat["title"] for cat in page.get("categories", [])]

print('Get categories of the article:')
print(get_categories(TITLE))


############################################################################################
# Get external references (URLs)
###########################################################################################
def get_external_links(title):
    links = []
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extlinks",
        "ellimit": "max"
    }

    while True:
        r = requests.get(API_URL, params=params).json()
        page = next(iter(r["query"]["pages"].values()))
        links.extend([link["*"] for link in page.get("extlinks", [])])

        if "continue" in r:
            params.update(r["continue"])
        else:
            break

    return links
print('Get external references (URLs)')
print(get_external_links(TITLE))



############################################################################################
#
###########################################################################################


