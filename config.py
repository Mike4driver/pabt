import logging
from pathlib import Path
import os
import re
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from database import get_setting, update_setting # Assuming database.py is in the same root

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Base Directory ---
BASE_DIR = Path(__file__).resolve().parent

# --- Media Directory Configuration ---
DEFAULT_MEDIA_SUBDIR = "media"
MEDIA_DIR_NAME = get_setting("media_directory_name")
if not MEDIA_DIR_NAME:
    logger.warning(f"Media directory name not found in database settings. Falling back to '{DEFAULT_MEDIA_SUBDIR}'.")
    MEDIA_DIR_NAME = DEFAULT_MEDIA_SUBDIR
    update_setting("media_directory_name", MEDIA_DIR_NAME)
MEDIA_DIR = BASE_DIR / MEDIA_DIR_NAME
logger.info(f"Media directory set to: {MEDIA_DIR}")

# --- Generated Content Directories ---
TRANSCODED_DIR = BASE_DIR / "media_transcoded"
PREVIEWS_DIR = BASE_DIR / "static" / "previews"
THUMBNAILS_DIR = BASE_DIR / "static" / "thumbnails"
STATIC_ICONS_DIR = BASE_DIR / "static" / "icons"

# Ensure base directories for generated content exist
TRANSCODED_DIR.mkdir(exist_ok=True)
PREVIEWS_DIR.mkdir(exist_ok=True)
THUMBNAILS_DIR.mkdir(exist_ok=True)
STATIC_ICONS_DIR.mkdir(exist_ok=True)
# MEDIA_DIR itself should be managed by the user or checked for existence at app startup.

# --- Supported File Extensions ---
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
SUPPORTED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg'}
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif'}

# --- Jinja2 Templates ---
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# --- Static Files Mounting (to be done on the app instance in main.py) ---
# This function can be called from main.py to mount static files
def mount_static_files(app):
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# --- Utility Functions ---
def slugify_for_id(value: str) -> str:
    # Remove characters that are not alphanumeric, underscores, or hyphens
    value = re.sub(r'[^\\w\\s-]', '', value).strip().lower()
    # Replace whitespace and sequences of hyphens with a single hyphen
    value = re.sub(r'[-\\s]+', '-', value)
    # Ensure it starts with a letter (important for CSS IDs)
    if not value or not value[0].isalpha():
        value = "id-" + value
    return value

templates.env.filters['slugify_for_id'] = slugify_for_id # Add to Jinja environment

def safe_path_join(base_path: Path, filename: str) -> Path:
    """Safely join a base path with a filename, preventing directory traversal."""
    safe_filename = os.path.basename(filename)
    return base_path / safe_filename

def get_media_type_from_extension(file_path: Path) -> str:
    """Determine media type from file extension."""
    suffix = file_path.suffix.lower()
    if suffix in SUPPORTED_VIDEO_EXTENSIONS:
        return "video"
    elif suffix in SUPPORTED_AUDIO_EXTENSIONS:
        return "audio"
    elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    return "unknown"

logger.info("Config module loaded.") 