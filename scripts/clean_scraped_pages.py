import os
import re
from bs4 import BeautifulSoup
from langdetect import detect
from tqdm import tqdm

SCRAPED_DIR = "/workspaces/rag-chatbot/data/angelone_webpages"
CLEANED_DIR = "/workspaces/rag-chatbot/data/cleaned_pages"
os.makedirs(CLEANED_DIR, exist_ok=True)

# Tags to keep
ALLOWED_TAGS = {"p", "li", "h1", "h2", "h3", "h4"}

# Junk phrases to remove
JUNK_PATTERNS = [
    r"(?i)please wait",
    r"(?i)popular stocks",
    r"(?i)join our .* customers",
    r"(?i)Angel One Super App",
    r"(?i)related search",
    r"(?i)download the app",
    r"(?i)minimal brokerage charges",
    r"(?i)ARQ prime",
    r"(?i)open free demat",
    r"(?i)privacy policy",
    r"(?i)disclaimer",
    r"(?i)copyright",
    # —— extra menu/nav junk —— 
    r"(?i)share price",
    r"(?i)calculate your sip return",
    r"(?i)mutual funds?",
    r"(?i)we are here to help",
    r"(?i)step by step guide",
    r"(?i)learn how to",
    r"(?i)know (all about|how to)",
    r"(?i)trusted by",
    r"(?i)start an sip",
    r"(?i)brokerage & other charges",
    r"(?i)angel one - community",
    r"(?i)benefits (of|of being)",
    r"(?i)dos and don'ts",
    r"(?i)by proceeding",
]

def is_mostly_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

def clean_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # remove script/style/noscript tags entirely
    for bad in soup(["script", "style", "noscript"]):
        bad.decompose()

    # collect text only from allowed tags
    raw_lines = []
    for tag in soup.find_all(ALLOWED_TAGS):
        text = tag.get_text(separator=" ").strip()
        if text:
            raw_lines.append(text)

    # compile junk patterns once
    junk_re = [re.compile(p) for p in JUNK_PATTERNS]

    filtered = []
    for line in raw_lines:
        if any(rx.search(line) for rx in junk_re):
            continue
        if len(line.split()) < 4:
            continue
        if line.isupper():
            continue
        if filtered and line == filtered[-1]:
            continue
        filtered.append(line)

    # only keep content between the two markers
    start_phrase = "Know about how refer and earn works on Angel One"
    end_phrase   = "Our experts will be happy to assist you"
    try:
        # find the first occurrence of start_phrase, then slice until end_phrase
        start_idx = next(i for i, txt in enumerate(filtered) if start_phrase in txt) + 1
        end_idx   = next(i for i, txt in enumerate(filtered[start_idx:]) if end_phrase in txt) + start_idx
        filtered  = filtered[start_idx:end_idx]
    except StopIteration:
        # if markers not found, leave filtered as is
        pass

    return "\n".join(filtered)

if __name__ == "__main__":
    files = os.listdir(SCRAPED_DIR)
    kept, skipped = 0, 0

    for filename in tqdm(files, desc="Cleaning pages"):
        if not filename.endswith(".html") or "hindi" in filename.lower():
            skipped += 1
            continue

        path = os.path.join(SCRAPED_DIR, filename)
        cleaned_text = clean_html(path)

        if not cleaned_text.strip() or not is_mostly_english(cleaned_text):
            skipped += 1
            continue

        output_path = os.path.join(CLEANED_DIR, filename.replace(".html", ".txt"))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        kept += 1

    print(f"\n✅ Cleaned {kept} pages | ❌ Skipped {skipped} pages")
