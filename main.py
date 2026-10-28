"""SIACC headless pipeline: Google News -> scrape (threaded) -> Draft/Post to SIACC.

Designed to run on GitHub Actions (stateless runner), configured via env vars.
"""
import csv
import json
import os
import random
import re
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import feedparser
import nltk
import requests
from newspaper import Article, Config as NewsConfig
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# ---------------------------------------------------------------- config
TZ = ZoneInfo("Asia/Makassar")  # WITA, runner is UTC
BASE_URL = "https://siaccinfo.id"
TOKEN = os.getenv("SIACC_TOKEN", "").strip()
MODE = os.getenv("SIACC_MODE", "Draft").strip().title()  # Draft | Post
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "30"))
WORKERS = int(os.getenv("WORKERS", "5"))
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() == "true"
IGNORED_DOMAINS = [
    d.strip().lower()
    for d in os.getenv("IGNORED_DOMAINS", "rumah123.com, lamudi.co.id, gentra.co.id").split(",")
    if d.strip()
]
DEFAULT_QUERIES = """Tabanan+when:1d
Penebel+when:1d
Pupuan+when:1d
Selemadeg+when:1d
Kerambitan+when:1d
Baturiti+when:1d
Kediri%20AND%20Tabanan+when:1d
Marga%20AND%20Tabanan+when:1d"""
QUERIES = [q.strip() for q in os.getenv("QUERIES", "").splitlines() if q.strip()] or DEFAULT_QUERIES.splitlines()

STATE_FILE = Path("state/posted_urls.json")
OUT_DIR = Path("output") / datetime.now(TZ).strftime("%Y-%m-%d")
IMG_DIR = OUT_DIR / "images"

FIXED_DEFAULTS = {
    "data_of_information": [2, 7],
    "latitude": -8.536609490935,
    "longitude": 115.13552944186335,
    "location_name": "Dajan Peken, 82114, Tabanan, Tabanan, Bali, Indonesia",
    "where": "Kabupaten Tabanan, Bali, Indonesia",
}
ID_DAYS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
ID_MONTHS = [
    "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json",
    "Referer": f"{BASE_URL}/beranda",
}

if not VERIFY_SSL:
    requests.packages.urllib3.disable_warnings()


def log(msg: str) -> None:
    print(f"[{datetime.now(TZ):%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------- helpers
def generate_hashtags() -> list:
    mandatory = ["kejaksaan", "kejatibali", "kejaritabanan"]
    pool = [
        "viral", "indonesia", "beritaterkini", "infobali", "baliterkini",
        "seputartabanan", "hukum", "kriminal", "kejaksaanri", "trending",
        "infotabanan", "tabananviral", "tabanannow", "tabananterkini", "fyp", "terdepan",
    ]
    return mandatory + random.sample(pool, k=random.randint(3, 5))


def time_sekarang() -> str:
    midnight = datetime.now(TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def extract_source_from_title(title: str) -> str:
    m = re.search(r"-\s*([^-]+)$", title)
    return m.group(1).strip() if m else ""


def split_sentences(text: str) -> list:
    if not text:
        return []
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]


def get_sentences(text: str, start: int, end: int) -> str:
    return " ".join(split_sentences(text)[start:end])


def format_how_prefix(pub_dt: datetime) -> str:
    return (
        f"Pada Hari {ID_DAYS[pub_dt.weekday()]} , {pub_dt.day} "
        f"{ID_MONTHS[pub_dt.month]} {pub_dt.year} {pub_dt:%H:%M}, Tabanan, Tabanan, Bali"
    )


def clean_string(text) -> str:
    if not text:
        return ""
    return "".join(c for c in str(text) if ord(c) >= 32).strip()


def clean_title_logic(title: str) -> str:
    if "-" in title:
        title = title.rsplit("-", 1)[0]
    title = re.sub(r"[^a-zA-Z0-9\s]", "", title)
    return " ".join(title.split())


def make_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    return webdriver.Chrome(options=opts)


def resolve_url(driver, google_url: str):
    try:
        driver.get(google_url)
        time.sleep(2)
        real = driver.current_url
        if real and "google.com" not in real:
            return clean_string(real)
    except Exception as e:
        log(f"resolve gagal: {e}")
    return None


def scrape_article(url: str):
    cfg = NewsConfig()
    cfg.fetch_images = True
    cfg.request_timeout = 20
    cfg.browser_user_agent = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    article = Article(url, language="id", config=cfg)
    article.download()
    article.parse()
    try:
        article.nlp()
        summary = article.summary
    except Exception:
        summary = ""
    return article.text, summary, article.top_image


def download_image(url: str, index: int) -> str:
    if not url or not url.startswith("http"):
        return ""
    try:
        r = requests.get(url, timeout=15, verify=VERIFY_SSL)
        if r.status_code == 200 and r.content:
            ctype = r.headers.get("Content-Type", "")
            ext = ".png" if "image/png" in ctype else ".webp" if "image/webp" in ctype else ".jpg"
            path = IMG_DIR / f"img_{index}_{int(time.time())}{ext}"
            path.write_bytes(r.content)
            return str(path)
    except Exception as e:
        log(f"gagal download gambar: {e}")
    return ""


# ---------------------------------------------------------------- state (dedupe across runs)
def load_state() -> list:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return []


def save_state(urls: list) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(urls[-3000:], indent=1))


# ---------------------------------------------------------------- collect
def process_candidate(idx: int, cand: dict):
    """Runs inside a worker thread: scrape article + download image."""
    try:
        full_text, summary, top_image = scrape_article(cand["url"])
    except Exception as e:
        log(f"scrape gagal ({cand['url'][:50]}): {e}")
        return None
    full_text = (full_text or "").replace("ADVERTISEMENT SCROLL TO RESUME CONTENT", "").strip()
    if len(full_text) <= 100:
        return None

    img_path = download_image(top_image, idx)
    if not img_path:
        log(f"skip (tanpa gambar): {cand['title'][:50]}")
        return None

    title, pub_dt = cand["title"], cand["pub_dt"]
    how = f"{format_how_prefix(pub_dt)}\n\n{full_text}"
    log(f"OK: {title[:60]}")
    return {
        "url": cand["url"],
        "image_path": img_path,
        "title": clean_title_logic(title),
        "when_": time_sekarang(),
        "what": get_sentences(full_text, 0, 2).strip() or summary.strip(),
        "why": title,
        "how": textwrap.shorten(how, width=5000, placeholder=""),
        "source_agent_report": extract_source_from_title(title) or "Media Online",
        "category": [5, int(random.choice([3, 7, 9]))],
        "hashtags": generate_hashtags(),
    }


def collect(seen_urls: set) -> list:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    candidates, seen_titles = [], set()
    limit = MAX_ARTICLES * 2  # buffer: some will fail (no image, blocked, too short)

    # Phase 1: resolve Google News links (sequential, Selenium is not thread-safe)
    driver = make_driver()
    try:
        for q in QUERIES:
            if len(candidates) >= limit:
                break
            log(f"Query: {q}")
            feed = feedparser.parse(
                f"https://news.google.com/rss/search?q={q}&hl=id&gl=ID&ceid=ID:id"
            )
            for entry in feed.entries:
                if len(candidates) >= limit:
                    break
                title, link = entry.get("title", ""), entry.get("link", "")
                key = clean_title_logic(title).lower()
                if not link or key in seen_titles:
                    continue
                seen_titles.add(key)

                url = resolve_url(driver, link)
                if not url or url in seen_urls:
                    continue
                if any(d in url.lower() for d in IGNORED_DOMAINS):
                    log(f"skip domain: {url[:60]}")
                    continue
                seen_urls.add(url)

                pub_dt = (
                    datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).astimezone(TZ)
                    if entry.get("published_parsed")
                    else datetime.now(TZ)
                )
                candidates.append({"url": url, "title": title, "pub_dt": pub_dt})
    finally:
        driver.quit()

    log(f"{len(candidates)} kandidat, scraping dengan {WORKERS} thread...")

    # Phase 2: scrape + image download in parallel
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(lambda a: process_candidate(*a), enumerate(candidates, 1)))

    return [r for r in results if r][:MAX_ARTICLES]


# ---------------------------------------------------------------- post (sequential on purpose)
def build_payload(item: dict) -> dict:
    return {
        "title": clean_string(item["title"]),
        "when_": item["when_"],
        "where": FIXED_DEFAULTS["where"],
        "who": clean_string(item["title"]),
        "what": clean_string(item["what"]),
        "why": clean_string(item["why"]),
        "how": clean_string(item["how"]),
        "source_agent_report": clean_string(item["source_agent_report"]),
        "location_name": FIXED_DEFAULTS["location_name"],
        "latitude": FIXED_DEFAULTS["latitude"],
        "longitude": FIXED_DEFAULTS["longitude"],
        "category": item["category"],
        "data_of_information": FIXED_DEFAULTS["data_of_information"],
        "hashtags": item["hashtags"],
    }


def upload_attachment(record_id, image_path: str):
    path = Path(image_path)
    ctype = "image/png" if path.suffix == ".png" else "image/webp" if path.suffix == ".webp" else "image/jpeg"
    field = "draft_id" if MODE == "Draft" else "agent_report_id"
    headers = {k: v for k, v in HEADERS.items() if k.lower() != "content-type"}
    with path.open("rb") as f:
        r = requests.post(
            f"{BASE_URL}/apis/feeds/agent-report/upload-attachment?api_version=v2",
            headers=headers,
            data={field: str(record_id)},
            files={"attachment[]": (path.name, f, ctype)},
            verify=VERIFY_SSL,
            timeout=60,
        )
    return r.status_code in (200, 201), f"HTTP {r.status_code}: {r.text[:100]}"


def post_items(items: list, posted: list) -> tuple:
    endpoint = "/apis/feeds/draft/create" if MODE == "Draft" else "/apis/feeds/agent-report/create"
    url = f"{BASE_URL}{endpoint}?api_version=v2"
    ok = fail = 0
    for i, item in enumerate(items, 1):
        tag = f"[{i}/{len(items)}] {item['title'][:50]}"
        try:
            r = requests.post(url, headers=HEADERS, json=build_payload(item), verify=VERIFY_SSL, timeout=60)
        except requests.RequestException as e:
            log(f"{tag} -> request error: {e}")
            fail += 1
            continue

        if r.status_code == 401:
            log("401 Unauthorized: SIACC_TOKEN kedaluwarsa/salah. Perbarui secret.")
            save_state(posted)
            sys.exit(2)
        if r.status_code not in (200, 201):
            log(f"{tag} -> gagal buat record: HTTP {r.status_code} {r.text[:120]}")
            fail += 1
            continue

        # record exists now -> mark as posted so a retry never duplicates it
        posted.append(item["url"])
        save_state(posted)
        ok += 1

        try:
            rid = r.json().get("data", {}).get("result", {}).get("id")
        except Exception:
            rid = None
        if not rid:
            log(f"{tag} -> record dibuat, tapi ID tidak ditemukan (lampiran dilewati)")
        else:
            up_ok, up_msg = upload_attachment(rid, item["image_path"])
            log(f"{tag} -> {MODE} ID {rid}, lampiran {'OK' if up_ok else 'GAGAL ' + up_msg}")
        time.sleep(3)
    return ok, fail


# ---------------------------------------------------------------- main
def main() -> int:
    if MODE not in ("Draft", "Post"):
        log(f"SIACC_MODE tidak valid: {MODE}")
        return 2
    if not TOKEN and not DRY_RUN:
        log("SIACC_TOKEN belum diset.")
        return 2

    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    posted = load_state()
    log(f"Mode={MODE} dry_run={DRY_RUN} max={MAX_ARTICLES} workers={WORKERS} state={len(posted)} URL")
    items = collect(set(posted))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if items:
        csv_path = OUT_DIR / f"{datetime.now(TZ):%H%M}-success.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(items[0].keys()))
            w.writeheader()
            for it in items:
                w.writerow({**it, "category": ", ".join(map(str, it["category"])), "hashtags": ", ".join(it["hashtags"])})
    log(f"{len(items)} berita siap")

    if DRY_RUN or not items:
        return 0

    ok, fail = post_items(items, posted)
    log(f"Selesai: {ok} berhasil, {fail} gagal")
    return 1 if ok == 0 and fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())