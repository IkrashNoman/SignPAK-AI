# import os
# import re
# import json
# import time
# import logging
# import shutil
# import threading
# from pathlib import Path
# from typing import Optional
# from concurrent.futures import ThreadPoolExecutor

# import requests
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry

# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# from selenium.common.exceptions import WebDriverException
# from webdriver_manager.chrome import ChromeDriverManager

# # ==========================================================
# # CONFIG & TARGET WORDS
# # ==========================================================
# ROOT_DIR = Path(r"D:\\FYP- Final Year Project\\SignPAK-AI")

# # Exact folder structure matching your requested deliverables
# RAW_VIDEOS_DIR = ROOT_DIR / "data" / "raw"
# METADATA_DIR = ROOT_DIR / "data" / "metadata"
# LOGS_DIR = ROOT_DIR / "data" / "logs"

# CATEGORIES_URL = "https://admin.psl.org.pk/api/category"
# CATEGORY_DETAIL_URL = "https://admin.psl.org.pk/api/category/{}"

# MAX_RETRIES = 5
# REQUEST_TIMEOUT = 60
# SELENIUM_WAIT = 5
# MAX_DOWNLOAD_THREADS = 8

# # Target list for the first 20 words. Expand this list later to 1,000 words.
# TARGET_WORDS = [
#     "Hello", "Goodbye", "Welcome", "Thank you", "Please", "Sorry", "Excuse me", "Congratulations", "Good morning", "Good night",
#     "Yes", "No", "Okay", "Maybe", "Good", "Bad", "Correct", "Wrong", "Finished", "Wait"
# ]
# # Convert target words to lowercase set for fast and flexible lookups
# TARGET_WORDS_CLEAN = {word.lower().strip() for word in TARGET_WORDS}

# # Metadata File Definitions matching your specific naming schemas
# CATEGORIES_FILE = METADATA_DIR / "categories.json"
# MASTER_DATASET_FILE = METADATA_DIR / "master_dataset.json"
# FAILED_FILE = LOGS_DIR / "failed.json"

# # ==========================================================
# # CREATE FOLDERS
# # ==========================================================
# for folder in [RAW_VIDEOS_DIR, METADATA_DIR, LOGS_DIR]:
#     folder.mkdir(parents=True, exist_ok=True)

# # ==========================================================
# # LOGGING
# # ==========================================================
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s"
# )
# logger = logging.getLogger(__name__)

# # ==========================================================
# # SESSION WITH RETRIES
# # ==========================================================
# def create_session() -> requests.Session:
#     session = requests.Session()
#     retry_strategy = Retry(
#         total=MAX_RETRIES,
#         backoff_factor=2,
#         status_forcelist=[429, 500, 502, 503, 504],
#         allowed_methods=["GET"]
#     )
#     adapter = HTTPAdapter(max_retries=retry_strategy)
#     session.mount("http://", adapter)
#     session.mount("https://", adapter)
#     return session

# session = create_session()
# metadata_lock = threading.Lock()

# # ==========================================================
# # HELPER JSON UTILITIES
# # ==========================================================
# def load_json(path: Path, default=None):
#     if path.exists():
#         try:
#             with open(path, "r", encoding="utf-8") as f:
#                 return json.load(f)
#         except:
#             return default
#     return default

# def save_json(path: Path, data):
#     tmp = path.with_suffix(".tmp")
#     with open(tmp, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
#     shutil.move(tmp, path)

# def log_failure(item: dict):
#     with metadata_lock:
#         failures = load_json(FAILED_FILE, [])
#         failures.append(item)
#         save_json(FAILED_FILE, failures)

# # ==========================================================
# # DOWNLOAD JSON WITH CACHING
# # ==========================================================
# def download_json(url: str, output_path: Path):
#     if output_path.exists():
#         logger.info(f"Using cached API response: {output_path.name}")
#         return load_json(output_path)

#     for attempt in range(1, MAX_RETRIES + 1):
#         try:
#             logger.info(f"Downloading API data: {url}")
#             resp = session.get(url, timeout=REQUEST_TIMEOUT)
#             resp.raise_for_status()
#             data = resp.json()
#             save_json(output_path, data)
#             return data
#         except Exception as e:
#             logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {url}: {e}")
#             time.sleep(2)
#     raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} attempts")

# # ==========================================================
# # SELENIUM DRIVER UTILITIES
# # ==========================================================
# def create_driver() -> webdriver.Chrome:
#     options = Options()
#     options.add_argument("--start-maximized")
#     options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
#     return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# driver = create_driver()

# def reconnect_driver():
#     global driver
#     try: driver.quit()
#     except: pass
#     logger.warning("Reconnecting Selenium driver...")
#     driver = create_driver()

# CLEAN_MP4_REGEX = re.compile(r"https://d2a517kx38mlos\.cloudfront\.net/[^\s\"'<>]+\.mp4")

# def extract_video_url(page_url: str) -> Optional[str]:
#     max_attempts = 3
#     for attempt in range(max_attempts):
#         try:
#             driver.get(page_url)
#             time.sleep(SELENIUM_WAIT)
#             driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#             time.sleep(3)

#             html = driver.page_source
#             match = CLEAN_MP4_REGEX.search(html)
#             if match: return match.group(0)

#             try:
#                 video_elem = driver.find_element(By.TAG_NAME, "video")
#                 src = video_elem.get_attribute("src")
#                 if src and src.strip(): return src.strip()
#             except: pass

#             logs = driver.get_log("performance")
#             for entry in logs:
#                 match = CLEAN_MP4_REGEX.search(entry["message"])
#                 if match: return match.group(0)

#             return None
#         except WebDriverException as e:
#             logger.error(f"WebDriver error on attempt {attempt+1}: {e}")
#             if attempt < max_attempts - 1:
#                 reconnect_driver()
#                 time.sleep(2)
#             else: return None
#         except Exception as e:
#             logger.error(f"Selenium extraction error: {e}")
#             return None
#     return None

# def download_video(url: str, output_path: Path) -> bool:
#     if output_path.exists():
#         return True
#     for attempt in range(1, MAX_RETRIES + 1):
#         try:
#             with session.get(url, stream=True, timeout=120) as r:
#                 r.raise_for_status()
#                 with open(output_path, "wb") as f:
#                     for chunk in r.iter_content(chunk_size=8192):
#                         f.write(chunk)
#             return True
#         except Exception as e:
#             logger.warning(f"Download attempt {attempt}/{MAX_RETRIES} failed: {e}")
#             time.sleep(3)
#     return False

# # ==========================================================
# # MAIN EXECUTION PIPELINE
# # ==========================================================
# # Step 1: Download Categories Master List
# categories_raw = download_json(CATEGORIES_URL, CATEGORIES_FILE)
# categories_list = categories_raw.get("data", [])

# master_dataset = load_json(MASTER_DATASET_FILE, [])

# downloaded_count = 0
# skipped_count = 0
# failed_count = 0
# download_futures = []

# executor = ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_THREADS)

# # Helper function to append to master_dataset following your Schema #4
# def update_master_dataset(cat_data, concept_data, final_path, v_url):
#     with metadata_lock:
#         existing_cat = next((item for item in master_dataset if item["category"]["id"] == cat_data["id"]), None)
#         concept_entry = {
#             "id": concept_data.get("id"),
#             "title": concept_data.get("title"),
#             "title_secondary": concept_data.get("title_secondary"),
#             "video_path": str(final_path),
#             "video_url": v_url
#         }
#         if existing_cat:
#             # Avoid duplicate concept insertion
#             if not any(c["id"] == concept_data["id"] for c in existing_cat["concepts"]):
#                 existing_cat["concepts"].append(concept_entry)
#         else:
#             master_dataset.append({
#                 "category": {
#                     "id": cat_data.get("id"),
#                     "slug": cat_data.get("slug"),
#                     "title": cat_data.get("title"),
#                     "title_secondary": cat_data.get("title_secondary"),
#                     "storage_slug": cat_data.get("storage_slug")
#                 },
#                 "concepts": [concept_entry]
#             })
#         save_json(MASTER_DATASET_FILE, master_dataset)

# def download_task(concept_id: str, video_url: str, video_path: Path, cat_info: dict, concept_info: dict, category_metadata: list):
#     success = download_video(video_url, video_path)
#     status = "downloaded" if success else "failed"
    
#     video_metadata_entry = {
#         "id": str(concept_id),
#         "category": cat_info.get("title"),
#         "english_word": concept_info.get("title"),
#         "urdu_word": concept_info.get("title_secondary", ""),
#         "video_url": video_url,
#         "video_path": str(video_path),
#         "video_status": status,
#         "source": "psl.org.pk"
#     }

#     if not success:
#         log_failure({"category": cat_info.get("title"), "word": concept_info.get("title"), "video": video_url, "reason": "Download Error"})
#         return False

#     with metadata_lock:
#         category_metadata.append(video_metadata_entry)
#         save_json(METADATA_DIR / f"video_metadata_{cat_info.get('slug')}.json", category_metadata)
    
#     update_master_dataset(cat_info, concept_info, video_path, video_url)
#     return True

# try:
#     for category in categories_list:
#         cat_id = category["id"]
#         cat_slug = category["slug"]
        
#         # Load Category Details (Schema #2)
#         detail_path = METADATA_DIR / f"category_{cat_slug}.json"
#         try:
#             detail_data = download_json(CATEGORY_DETAIL_URL.format(cat_id), detail_path)
#         except Exception as e:
#             logger.error(f"Could not download details for category {cat_slug}: {e}")
#             continue

#         concepts = detail_data.get("data", {}).get("concepts", [])
#         if not concepts:
#             continue

#         # Look for local category video metadata file (Schema #3)
#         cat_metadata_file = METADATA_DIR / f"video_metadata_{cat_slug}.json"
#         cat_metadata = load_json(cat_metadata_file, [])

#         cat_folder = RAW_VIDEOS_DIR / cat_slug
#         cat_folder.mkdir(parents=True, exist_ok=True)

#         for concept in concepts:
#             eng_word = (concept.get("title") or "").strip()
            
#             # --- CRITICAL FILTER STEP ---
#             # Checks if the word is in your target list (Case-Insensitive)
#             if eng_word.lower() not in TARGET_WORDS_CLEAN:
#                 continue

#             concept_id = concept.get("id")
#             concept_slug = concept.get("slug") or eng_word.lower().replace(" ", "-")
#             safe_english = re.sub(r'[\\/:*?"<>|]', "_", eng_word)
#             video_path = cat_folder / f"{safe_english}.mp4"

#             # Check if this precise concept has been handled already
#             if any(str(c.get("id")) == str(concept_id) for item in master_dataset for c in item.get("concepts", [])) or video_path.exists():
#                 logger.info(f" ⏭  [{category.get('title')}] {eng_word} already completely tracked or downloaded.")
#                 skipped_count += 1
#                 if video_path.exists() and not any(str(c.get("id")) == str(concept_id) for item in master_dataset for c in item.get("concepts", [])):
#                      update_master_dataset(category, concept, video_path, "")
#                 continue

#             # Scrape page link layout
#             page_url = f"https://psl.org.pk/dictionary/{cat_id}-{cat_slug}/{concept_id}-{concept_slug}"
#             logger.info(f"Target Hit 🎯 Extracting: {eng_word}")
            
#             video_url = extract_video_url(page_url)
#             if not video_url:
#                 failed_count += 1
#                 log_failure({"category": category.get("title"), "word": eng_word, "page": page_url, "reason": "No valid video URL detected"})
#                 continue

#             future = executor.submit(
#                 download_task,
#                 concept_id=concept_id,
#                 video_url=video_url,
#                 video_path=video_path,
#                 cat_info=category,
#                 concept_info=concept,
#                 category_metadata=cat_metadata
#             )
#             download_futures.append(future)

#         # Await completion per category unit to correctly populate distinct files
#         for future in download_futures:
#             if future.result(): downloaded_count += 1
#             else: failed_count += 1
#         download_futures.clear()

# finally:
#     executor.shutdown(wait=True)
#     try: driver.quit()
#     except: pass

# # ==========================================================
# # FINAL REPORTING
# # ==========================================================
# print("\n" + "="*60)
# print("FINAL EXECUTION STATISTICS")
# print("="*60)
# print(f"Downloaded Successfully : {downloaded_count}")
# print(f"Skipped (Processed)    : {skipped_count}")
# print(f"Failed Tasks           : {failed_count}")
# print(f"Total Target Progress  : {downloaded_count + skipped_count} / {len(TARGET_WORDS)}")
# print("="*60)