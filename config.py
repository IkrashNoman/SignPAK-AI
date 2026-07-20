# config.py
import re
from pathlib import Path

# Automatically hooks into the root directory of SignPAK-AI wherever it runs
BASE_DIR = Path(__file__).resolve().parent

# ==========================================================
# CENTRALIZED SYSTEM DIRECTORIES
# ==========================================================
DATA_DIR = BASE_DIR / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw"
REPETITIONS_DIR = DATA_DIR / "repetitions"
CROPPED_DIR = DATA_DIR / "cropped"
PROCESSED_DIR = DATA_DIR / "processed"
LANDMARKS_DIR = DATA_DIR / "landmarks"
AUGMENTED_DIR = DATA_DIR / "augmented"
LOGS_DIR = DATA_DIR / "logs"

# Metadata Substructure Tracks
METADATA_DIR = DATA_DIR / "metadata"
CAT_METADATA_DIR = METADATA_DIR / "categories"
MASTER_METADATA_DIR = METADATA_DIR / "master"
REPORTS_DIR = METADATA_DIR / "reports"

CSV_DIR = DATA_DIR / "csv"
MODELS_DIR = BASE_DIR / "models"

# Ensure all structural base paths exist right away
for folder in [RAW_VIDEOS_DIR, REPETITIONS_DIR, CROPPED_DIR, PROCESSED_DIR, 
               LANDMARKS_DIR, AUGMENTED_DIR, LOGS_DIR, CAT_METADATA_DIR, 
               MASTER_METADATA_DIR, REPORTS_DIR, CSV_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ==========================================================
# MASTER METADATA SCHEMAS
# ==========================================================
CATEGORIES_FILE = MASTER_METADATA_DIR / "categories.json"
MASTER_DATASET_FILE = MASTER_METADATA_DIR / "master_dataset.json"
FAILED_FILE = LOGS_DIR / "failed.json"

# ==========================================================
# SCRAPER CONNECTIONS & POLICIES
# ==========================================================
CATEGORIES_URL = "https://admin.psl.org.pk/api/category"
CATEGORY_DETAIL_URL = "https://admin.psl.org.pk/api/category/{}"

MAX_RETRIES = 5
REQUEST_TIMEOUT = 60
SELENIUM_WAIT = 5
MAX_DOWNLOAD_THREADS = 8

# Target list for the first 20 words. (Can scale to 1,000 smoothly later)
TARGET_WORDS = [
    "Hello", "Goodbye", "Welcome", "Thank you", "Please", "Sorry", "Excuse me", "Congratulations", "Good morning", "Good night",
    "Yes", "No", "Okay", "Maybe", "Good", "Bad", "Correct", "Wrong", "Finished", "Wait"
]
TARGET_WORDS_CLEAN = {word.lower().strip() for word in TARGET_WORDS}