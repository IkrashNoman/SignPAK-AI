# scripts/01_download_dataset.py
import sys
from pathlib import Path

# Appends the root folder to system paths dynamically so Python can find config.py from inside /scripts
sys.path.append(str(Path(__file__).resolve().parents[1]))

import re
import json
import time
import logging
import shutil
import threading
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

import config

# ==========================================================
# SYSTEM SETUP
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# --- STRICT VOCABULARY AND CATEGORY MAPPING PROTOCOL ---
VOCABULARY_ROUTING = {
    # Greetings & Politeness
    "hello": ("Greetings & Politeness", "Hello"),
    "goodbye": ("Greetings & Politeness", "Goodbye"),
    "welcome": ("Greetings & Politeness", "Welcome"),
    "thank you": ("Greetings & Politeness", "Thank you"),
    "please": ("Greetings & Politeness", "Please"),
    "sorry": ("Greetings & Politeness", "Sorry"),
    "excuse me": ("Greetings & Politeness", "Excuse me"),
    "congratulations": ("Greetings & Politeness", "Congratulations"),
    "good morning": ("Greetings & Politeness", "Good morning"),
    "good night": ("Greetings & Politeness", "Good night"),
    # Yes/No & Common Expressions
    "yes": ("Yes_No & Common Expressions", "Yes"),
    "no": ("Yes_No & Common Expressions", "No"),
    "okay": ("Yes_No & Common Expressions", "Okay"),
    "ok": ("Yes_No & Common Expressions", "Okay"), # Handle website short abbreviation
    "maybe": ("Yes_No & Common Expressions", "Maybe"),
    "good": ("Yes_No & Common Expressions", "Good"),
    "bad": ("Yes_No & Common Expressions", "Bad"),
    "correct": ("Yes_No & Common Expressions", "Correct"),
    "wrong": ("Yes_No & Common Expressions", "Wrong"),
    "finished": ("Yes_No & Common Expressions", "Finished"),
    "wait": ("Yes_No & Common Expressions", "Wait")
}

def create_session() -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=config.MAX_RETRIES,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

session = create_session()
metadata_lock = threading.Lock()
driver_lock = threading.Lock()

# ==========================================================
# FILE UTILITIES
# ==========================================================
def load_json(path: Path, default=None):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return default
    return default

def save_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    shutil.move(tmp, path)

def log_failure(item: dict):
    with metadata_lock:
        failures = load_json(config.FAILED_FILE, [])
        failures.append(item)
        save_json(config.FAILED_FILE, failures)

def download_json(url: str, output_path: Path):
    if output_path.exists():
        return load_json(output_path)

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            logger.info(f"Downloading API data: {url}")
            resp = session.get(url, timeout=config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            save_json(output_path, data)
            return data
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{config.MAX_RETRIES} failed for {url}: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to fetch {url} after {config.MAX_RETRIES} attempts")

# ==========================================================
# AUTOMATION DRIVER EXTENSION
# ==========================================================
def create_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--log-level=3") 
    options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver = create_driver()

def reconnect_driver():
    global driver
    try: driver.quit()
    except: pass
    logger.warning("Reconnecting Selenium driver...")
    driver = create_driver()

CLEAN_MP4_REGEX = re.compile(r"https://d2a517kx38mlos\.cloudfront\.net/[^\s\"'<>]+\.mp4")

def extract_video_url(page_url: str) -> Optional[str]:
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            driver.get(page_url)
            time.sleep(config.SELENIUM_WAIT)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            html = driver.page_source
            match = CLEAN_MP4_REGEX.search(html)
            if match: return match.group(0)

            try:
                video_elem = driver.find_element(By.TAG_NAME, "video")
                src = video_elem.get_attribute("src")
                if src and src.strip(): return src.strip()
            except: pass

            logs = driver.get_log("performance")
            for entry in logs:
                match = CLEAN_MP4_REGEX.search(entry["message"])
                if match: return match.group(0)

            return None
        except WebDriverException as e:
            logger.error(f"WebDriver error on attempt {attempt+1}: {e}")
            if attempt < max_attempts - 1:
                reconnect_driver()
                time.sleep(2)
            else: return None
        except Exception as e:
            logger.error(f"Selenium extraction error: {e}")
            return None
    return None

def download_video(url: str, output_path: Path) -> bool:
    if output_path.exists():
        return True
    
    # Attempt 1: Fast direct download via Requests
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            with session.get(url, stream=True, timeout=120) as r:
                if r.status_code == 200:
                    with open(output_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    return True
        except Exception as e:
            logger.warning(f"Download attempt {attempt} via requests failed: {e}")
            time.sleep(1)
            
    # Attempt 2: Smart Fallback via browser context
    with driver_lock:
        logger.info(f"Requests blocked with 403. Attempting browser-based recovery download for: {output_path.name}")
        try:
            driver.get(url)
            time.sleep(2)
            bytes_script = "return fetch(arguments[0]).then(res => res.blob()).then(blob => new Promise((resolve) => { const reader = new FileReader(); reader.onloadend = () => resolve(reader.result); reader.readAsDataURL(blob); }));"
            base64_data = driver.execute_script(bytes_script, url)
            if base64_data and "," in base64_data:
                import base64
                media_bytes = base64.b64decode(base64_data.split(",")[1])
                with open(output_path, "wb") as f:
                    f.write(media_bytes)
                return True
        except Exception as e:
            logger.error(f"Browser recovery download failed: {e}")
        
    return False

# ==========================================================
# CONTEXT UPDATER
# ==========================================================
def update_master_dataset(custom_cat_name: str, concept_data: dict, final_path: Path, v_url: str):
    with metadata_lock:
        existing_cat = next((item for item in master_dataset if item["category"]["title"] == custom_cat_name), None)
        concept_entry = {
            "id": concept_data.get("id"),
            "title": concept_data.get("title"),
            "title_secondary": concept_data.get("title_secondary"),
            "video_path": str(final_path),
            "video_url": v_url
        }
        if existing_cat:
            if not any(c["id"] == concept_data["id"] for c in existing_cat["concepts"]):
                existing_cat["concepts"].append(concept_entry)
        else:
            master_dataset.append({
                "category": {
                    "id": custom_cat_name.lower().replace(" ", "_"),
                    "slug": custom_cat_name.lower().replace(" ", "_"),
                    "title": custom_cat_name,
                    "title_secondary": "",
                    "storage_slug": custom_cat_name.lower().replace(" ", "_")
                },
                "concepts": [concept_entry]
            })
        save_json(config.MASTER_DATASET_FILE, master_dataset)

def download_task(concept_id: str, video_url: str, video_path: Path, custom_cat_name: str, concept_info: dict, cat_slug: str):
    success = download_video(video_url, video_path)
    status = "downloaded" if success else "failed"
    
    video_metadata_entry = {
        "id": str(concept_id),
        "custom_category": custom_cat_name,
        "english_word": concept_info.get("title"),
        "urdu_word": concept_info.get("title_secondary", ""),
        "video_url": video_url,
        "video_path": str(video_path),
        "video_status": status,
        "source": "psl.org.pk"
    }

    if not success:
        log_failure({"category": custom_cat_name, "word": concept_info.get("title"), "video": video_url, "reason": "Download Error"})
        return False

    with metadata_lock:
        cat_metadata_file = config.CAT_METADATA_DIR / f"{custom_cat_name.lower().replace(' ', '_')}.json"
        category_metadata = load_json(cat_metadata_file, [])
        if not any(entry["id"] == str(concept_id) for entry in category_metadata):
            category_metadata.append(video_metadata_entry)
            save_json(cat_metadata_file, category_metadata)
    
    update_master_dataset(custom_cat_name, concept_info, video_path, video_url)
    return True

# ==========================================================
# EVALUATOR MATCHER
# ==========================================================
def evaluate_routing_protocol(eng_word_lower: str) -> Optional[Tuple[str, str, bool]]:
    """
    Returns: (custom_category, final_filename_base, is_extra_flag)
    """
    for target, (custom_cat, formal_name) in VOCABULARY_ROUTING.items():
        # Match Variant 1: Exact target match or clean structural division (e.g. "No / Not")
        if (eng_word_lower == target or 
            eng_word_lower.startswith(f"{target} /") or 
            f"/ {target}" in eng_word_lower):
            return custom_cat, formal_name, False
            
        # Match Variant 2: Substring phrases (e.g., "No Smoking", "Have a good day!") -> Route to Extra
        if f" {target} " in f" {eng_word_lower} ":
            clean_phrase_name = eng_word_lower.title().replace(" ", "_").replace("/", "_")
            return custom_cat, clean_phrase_name, True
            
    return None

# ==========================================================
# PROCESSING PIPELINE
# ==========================================================
# Initialize or clean base folders cleanly
config.CAT_METADATA_DIR.mkdir(parents=True, exist_ok=True)
config.RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

categories_raw = download_json(config.CATEGORIES_URL, config.CATEGORIES_FILE)
categories_list = categories_raw.get("data", [])

master_dataset = load_json(config.MASTER_DATASET_FILE, [])

downloaded_count = 0
skipped_count = 0
failed_count = 0
download_futures = []

executor = ThreadPoolExecutor(max_workers=config.MAX_DOWNLOAD_THREADS)

try:
    for category in categories_list:
        cat_id = category["id"]
        cat_slug = category["slug"]
        
        detail_path = config.CAT_METADATA_DIR / f"category_{cat_slug}.json"
        try:
            detail_data = download_json(config.CATEGORY_DETAIL_URL.format(cat_id), detail_path)
        except Exception as e:
            logger.error(f"Could not download details for category {cat_slug}: {e}")
            continue

        concepts = detail_data.get("data", {}).get("concepts", [])
        if not concepts:
            continue

        for concept in concepts:
            eng_word = (concept.get("title") or "").strip()
            eng_word_lower = eng_word.lower()
            
            # Evaluate against your exact 20 custom constraints
            route_info = evaluate_routing_protocol(eng_word_lower)
            if not route_info:
                continue
                
            custom_category, clean_filename, is_extra = route_info
            
            # Dynamically resolve file structure hierarchy paths based on your categories
            if is_extra:
                target_folder = config.RAW_VIDEOS_DIR / custom_category / "Extra"
            else:
                target_folder = config.RAW_VIDEOS_DIR / custom_category
                
            target_folder.mkdir(parents=True, exist_ok=True)
            
            safe_english = re.sub(r'[\\/:*?"<>|]', "_", clean_filename)
            video_path = target_folder / f"{safe_english}.mp4"

            concept_id = concept.get("id")
            concept_slug = concept.get("slug") or eng_word_lower.replace(" ", "-")

            # Duplicate Check
            if any(str(c.get("id")) == str(concept_id) for item in master_dataset for c in item.get("concepts", [])) or video_path.exists():
                skipped_count += 1
                if video_path.exists() and not any(str(c.get("id")) == str(concept_id) for item in master_dataset for c in item.get("concepts", [])):
                     update_master_dataset(custom_category, concept, video_path, "")
                continue

            page_url = f"https://psl.org.pk/dictionary/{cat_id}-{cat_slug}/{concept_id}-{concept_slug}"
            logger.info(f"Target Hit 🎯 Routing '{eng_word}' into Category: [{custom_category}] {'(Extra)' if is_extra else ''}...")
            
            video_url = extract_video_url(page_url)
            if not video_url:
                failed_count += 1
                log_failure({"category": custom_category, "word": eng_word, "page": page_url, "reason": "No valid video URL detected"})
                continue

            future = executor.submit(
                download_task,
                concept_id=concept_id,
                video_url=video_url,
                video_path=video_path,
                custom_cat_name=custom_category,
                concept_info=concept,
                cat_slug=cat_slug
            )
            download_futures.append(future)

        for future in download_futures:
            if future.result(): downloaded_count += 1
            else: failed_count += 1
        download_futures.clear()

finally:
    executor.shutdown(wait=True)
    try: driver.quit()
    except: pass

print("\n" + "="*60)
print("FINAL PIPELINE STATISTICS")
print("="*60)
print(f"Downloaded Successfully : {downloaded_count}")
print(f"Skipped (Processed)    : {skipped_count}")
print(f"Failed Tasks           : {failed_count}")
print(f"Total Filtered Actions : {downloaded_count + skipped_count}")
print("="*60)