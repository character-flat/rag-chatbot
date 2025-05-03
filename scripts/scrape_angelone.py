import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
import time
import re

BASE_URL = "https://www.angelone.in/support"
OUTPUT_DIR = "../data/angelone_webpages"
HEADERS = {"User-Agent": "Mozilla/5.0"}

visited_urls = set()
to_visit = set([BASE_URL])

def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.netloc == "www.angelone.in" and parsed.path.startswith("/support")

def get_page_content(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.text
    except requests.RequestException:
        pass
    return None

def extract_links(html, current_url):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag['href']
        full_url = urljoin(current_url, href)
        if is_valid_url(full_url) and full_url not in visited_urls:
            links.add(full_url)
    return links

def save_page_content(url, html):
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_")
    filename = f"{path if path else 'index'}.html"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    while to_visit:
        current_url = to_visit.pop()
        if current_url in visited_urls:
            continue
        html = get_page_content(current_url)
        if html:
            save_page_content(current_url, html)
            new_links = extract_links(html, current_url)
            to_visit.update(new_links)
        visited_urls.add(current_url)
        time.sleep(1)  # Be polite and avoid overwhelming the server

if __name__ == "__main__":
    main()
