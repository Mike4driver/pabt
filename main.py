from fastapi import FastAPI, Request, Query, HTTPException, Response, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import os
from pathlib import Path
import logging
import subprocess # For FFmpeg
from PIL import Image
from moviepy.editor import VideoFileClip
from mutagen import File
import shutil
from datetime import timedelta
import json # For config file
import sqlite3 # For type hinting if needed later
from database import get_db_connection, create_tables, get_setting, update_setting
from typing import Optional, List
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Database Initialization ---
# This will create tables if they don't exist and initialize settings
# if the DB is new, potentially reading from config.json for the first run.
create_tables()


# --- Directory Setup & Configuration --- 
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MEDIA_SUBDIR = "media" # Define this for settings_page JS fallback
# CONFIG_FILE = BASE_DIR / "config.json" # Keep for initial DB population, but DB is source of truth now

# DEFAULT_MEDIA_SUBDIR = "media" # Default if not in config or DB

# def load_config() -> dict: # Old function, will be replaced by DB access
#     if CONFIG_FILE.exists():
#         try:
#             with open(CONFIG_FILE, 'r') as f:
#                 config = json.load(f)
#                 logger.info(f"Loaded configuration from config.json: {config}")
#                 return config
#         except json.JSONDecodeError:
#             logger.error(f"Error decoding {CONFIG_FILE}. Using default configuration.")
#         except Exception as e:
#             logger.error(f"Error loading {CONFIG_FILE}: {e}. Using default configuration.")
#     return {"media_directory_name": get_setting("media_directory_name") or DEFAULT_MEDIA_SUBDIR}

# def save_config(config: dict): # Old function, will be replaced by DB access
#     try:
#         # Also update the database
#         if "media_directory_name" in config:
#             update_setting("media_directory_name", config["media_directory_name"])
#         # Optionally, still write to config.json for backup or external reference,
#         # but the DB is the primary source of truth.
#         with open(CONFIG_FILE, 'w') as f:
#             json.dump(config, f, indent=4)
#         logger.info(f"Saved configuration to config.json and DB: {config}")
#     except Exception as e:
#         logger.error(f"Error saving configuration: {e}")


# Load configuration and set MEDIA_DIR from Database
MEDIA_DIR_NAME = get_setting("media_directory_name")
if not MEDIA_DIR_NAME:
    logger.warning("Media directory name not found in database settings. Falling back to 'media'.")
    MEDIA_DIR_NAME = "media" # Fallback, should have been set by create_tables
    update_setting("media_directory_name", MEDIA_DIR_NAME) # Ensure it's in DB for next time

MEDIA_DIR = BASE_DIR / MEDIA_DIR_NAME
logger.info(f"Media directory set to: {MEDIA_DIR}")


# --- Media Scanning and Database Sync ---
def scan_media_directory_and_update_db():
    """
    Scans the MEDIA_DIR for supported files, extracts metadata, 
    and updates the media_files table in the database.
    Also removes entries for files that no longer exist.
    """
    logger.info(f"Starting media scan in directory: {MEDIA_DIR}")
    if not MEDIA_DIR.exists() or not MEDIA_DIR.is_dir():
        logger.error(f"Media directory {MEDIA_DIR} does not exist or is not a directory. Skipping scan.")
        return

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get all filenames currently in the database to find orphans later
        cursor.execute("SELECT id, filename, original_path FROM media_files")
        db_files_tuples = cursor.fetchall()
        db_files_map = {Path(row['original_path']).name: row['id'] for row in db_files_tuples}
        
        found_files_in_scan = set()

        for file_path_obj in MEDIA_DIR.iterdir():
            if file_path_obj.is_file():
                filename = file_path_obj.name
                original_path_str = str(file_path_obj.resolve())
                found_files_in_scan.add(filename)

                media_type = None
                suffix = file_path_obj.suffix.lower()
                if suffix in ['.mp4', '.avi', '.mov', '.mkv']: media_type = 'video'
                elif suffix in ['.mp3', '.wav', '.ogg']: media_type = 'audio'
                elif suffix in ['.jpg', '.jpeg', '.png', '.gif']: media_type = 'image'
                else:
                    # logger.debug(f"Skipping unsupported file type: {filename}")
                    continue

                logger.info(f"Processing file: {filename} (Type: {media_type})")

                # Check if file already exists in DB by original_path
                cursor.execute("SELECT id, last_scanned FROM media_files WHERE original_path = ?", (original_path_str,))
                existing_file = cursor.fetchone()

                # For now, we re-scan metadata each time.
                # Later, we can add a check for file modification time if needed for performance.
                # if existing_file and (datetime.now() - datetime.fromisoformat(existing_file['last_scanned'])).total_seconds() < 3600: # e.g. re-scan if older than 1 hour
                #     logger.info(f"File {filename} recently scanned. Skipping full metadata refresh.")
                #     continue

                duration = get_media_duration(file_path_obj) # Returns string like HH:MM:SS or None
                duration_seconds = None
                if duration:
                    parts = list(map(int, duration.split(':')))
                    duration_seconds = timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2]).total_seconds()

                width, height, fps = None, None, None
                size_bytes = file_path_obj.stat().st_size
                
                # More detailed metadata, especially for video
                # For videos, try to get resolution and FPS (example, needs refinement)
                if media_type == 'video':
                    try:
                        clip = VideoFileClip(str(file_path_obj.resolve()))
                        width, height = clip.size
                        fps = clip.fps
                        clip.close()
                    except Exception as e:
                        logger.warning(f"Could not get video metadata for {filename}: {e}")
                elif media_type == 'image':
                    try:
                        with Image.open(file_path_obj.resolve()) as img:
                            width, height = img.size
                    except Exception as e:
                        logger.warning(f"Could not get image metadata for {filename}: {e}")
                
                # Thumbnail, transcode, preview paths (these are potential paths, existence checked separately)
                # These functions might need adjustment to not assume MEDIA_DIR directly
                # but work with the provided file_path_obj
                display_thumb_url = get_display_thumbnail_url(file_path_obj, media_type) # this returns a URL
                # We need to store the *relative static path* or an indicator if it's generic
                actual_thumbnail_p = get_thumbnail_path(file_path_obj)
                has_specific_thumbnail = actual_thumbnail_p.exists()
                db_thumbnail_path = str(actual_thumbnail_p.relative_to(BASE_DIR)) if has_specific_thumbnail else None
                
                transcoded_p = TRANSCODED_DIR / f"{slugify_for_id(file_path_obj.stem)}.mp4" # Example, align with transcode_video output
                has_transcoded_version = transcoded_p.exists()
                db_transcoded_path = str(transcoded_p.relative_to(BASE_DIR)) if has_transcoded_version else None

                preview_p = get_preview_path(file_path_obj)
                has_preview = preview_p.exists()
                db_preview_path = str(preview_p.relative_to(BASE_DIR)) if has_preview else None

                metadata_dict = {"source": "filesystem_scan"} # Basic metadata
                if width and height: metadata_dict['resolution'] = f"{width}x{height}"
                if fps: metadata_dict['fps'] = round(fps, 2)
                
                metadata_json_str = json.dumps(metadata_dict)

                if existing_file:
                    # Update existing record
                    # Only update scan-related fields, preserve user_title and tags
                    logger.debug(f"Updating existing DB entry for: {filename} (scan data only)")
                    cursor.execute("""
                        UPDATE media_files 
                        SET media_type=?, duration=?, width=?, height=?, fps=?, size_bytes=?, 
                            last_scanned=CURRENT_TIMESTAMP, thumbnail_path=?, has_specific_thumbnail=?, 
                            transcoded_path=?, has_transcoded_version=?, preview_path=?, has_preview=?,
                            metadata_json=?
                        WHERE id=?
                    """, (media_type, duration_seconds, width, height, fps, size_bytes, 
                          db_thumbnail_path, has_specific_thumbnail, 
                          db_transcoded_path, has_transcoded_version, db_preview_path, has_preview,
                          metadata_json_str, existing_file['id']))
                else:
                    # Insert new record
                    logger.debug(f"Adding new DB entry for: {filename}")
                    cursor.execute("""
                        INSERT INTO media_files 
                            (filename, original_path, media_type, user_title, duration, width, height, fps, size_bytes, 
                             last_scanned, thumbnail_path, has_specific_thumbnail, 
                             transcoded_path, has_transcoded_version, preview_path, has_preview, 
                             tags, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (filename, original_path_str, media_type, None, # user_title defaults to None (NULL)
                          duration_seconds, width, height, fps, size_bytes,
                          db_thumbnail_path, has_specific_thumbnail, 
                          db_transcoded_path, has_transcoded_version, db_preview_path, has_preview, 
                          '[]', metadata_json_str)) # tags default to empty JSON list
                
                conn.commit()

        # Remove orphaned files from DB (files in DB but not found in current scan)
        db_filenames_set = set(db_files_map.keys())
        orphaned_filenames = db_filenames_set - found_files_in_scan
        if orphaned_filenames:
            logger.info(f"Found orphaned files in DB (will be removed): {orphaned_filenames}")
            for orphaned_file_name in orphaned_filenames:
                file_id_to_delete = db_files_map[orphaned_file_name]
                # Optionally, before deleting, consider if we should remove associated generated files
                # (thumbnails, transcodes, previews) from disk. For now, just DB record.
                cursor.execute("DELETE FROM media_files WHERE id = ?", (file_id_to_delete,))
                logger.info(f"Removed orphaned DB entry for: {orphaned_file_name} (ID: {file_id_to_delete})")
            conn.commit()

        logger.info("Media scan and database update completed.")

    except sqlite3.Error as e:
        logger.error(f"SQLite error during media scan: {e}")
        if conn:
            conn.rollback() # Rollback any partial changes if an error occurs
    except Exception as e:
        logger.error(f"Unexpected error during media scan: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# Call the scan function on startup
# This ensures the DB is populated/updated when the app starts.
# scan_media_directory_and_update_db() # Will be called after MEDIA_DIR is fully confirmed.


TRANSCODED_DIR = BASE_DIR / "media_transcoded"
PREVIEWS_DIR = BASE_DIR / "static" / "previews"   # For hover previews
THUMBNAILS_DIR = BASE_DIR / "static" / "thumbnails"
STATIC_ICONS_DIR = BASE_DIR / "static" / "icons"

# Ensure base directories for generated content exist (MEDIA_DIR should exist or be creatable by user)
TRANSCODED_DIR.mkdir(exist_ok=True)
PREVIEWS_DIR.mkdir(exist_ok=True)
THUMBNAILS_DIR.mkdir(exist_ok=True)
STATIC_ICONS_DIR.mkdir(exist_ok=True)
# We won't create MEDIA_DIR here; it should be a directory the user points to.
# A check could be added later if it does not exist after user configuration.

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# --- Jinja2 Filter for creating safe IDs ---
def slugify_for_id(value: str) -> str:
    import re
    # Remove characters that are not alphanumeric, underscores, or hyphens
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    # Replace whitespace and sequences of hyphens with a single hyphen
    value = re.sub(r'[-\s]+', '-', value)
    # Ensure it starts with a letter (important for CSS IDs)
    if not value or not value[0].isalpha():
        value = "id-" + value
    return value

templates.env.filters['slugify_for_id'] = slugify_for_id

# --- FFmpeg Helper Functions (Basic Implementation) ---
def run_ffmpeg_command(command_list):
    try:
        logger.info(f"Running FFmpeg command: {' '.join(command_list)}")
        process = subprocess.run(command_list, capture_output=True, text=True, check=True)
        logger.info(f"FFmpeg STDOUT: {process.stdout}")
        logger.info(f"FFmpeg STDERR: {process.stderr}") # FFmpeg often outputs info to stderr
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error for command {' '.join(command_list)}:")
        logger.error(f"FFmpeg STDOUT: {e.stdout}")
        logger.error(f"FFmpeg STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg command not found. Please ensure FFmpeg is installed and in your system PATH.")
        return False

def transcode_video(input_path: Path, output_path: Path, options: dict = None):
    if output_path.exists():
        logger.info(f"Transcoded file already exists, skipping: {output_path}")
        return True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Default options
    default_options = {
        'resolution': 'original',  # 'original', '1080p', '720p', '480p'
        'quality_mode': 'crf',     # 'crf' or 'bitrate'
        'crf': 23,                 # Constant Rate Factor (18=high, 23=medium, 28=low)
        'video_bitrate': '2M',     # For bitrate mode
        'audio_bitrate': '128k',   # Audio bitrate
        'preset': 'medium',        # ultrafast, fast, medium, slow
        'profile': 'high'          # baseline, main, high
    }
    
    # Merge provided options with defaults
    if options:
        default_options.update(options)
    opts = default_options
    
    # Build FFmpeg command
    command = ["ffmpeg", "-i", str(input_path.resolve())]
    
    # Video codec and settings
    command.extend(["-c:v", "libx264", "-preset", opts['preset'], "-profile:v", opts['profile']])
    
    # Quality settings
    if opts['quality_mode'] == 'crf':
        command.extend(["-crf", str(opts['crf'])])
    else:
        command.extend(["-b:v", opts['video_bitrate']])
    
    # Resolution settings
    if opts['resolution'] != 'original':
        if opts['resolution'] == '1080p':
            command.extend(["-vf", "scale=-2:1080"])
        elif opts['resolution'] == '720p':
            command.extend(["-vf", "scale=-2:720"])
        elif opts['resolution'] == '480p':
            command.extend(["-vf", "scale=-2:480"])
    
    # Audio settings
    command.extend(["-c:a", "aac", "-b:a", opts['audio_bitrate']])
    
    # Web optimization
    command.extend(["-movflags", "+faststart"])
    
    # Output file
    command.append(str(output_path.resolve()))
    
    return run_ffmpeg_command(command)

def create_hover_preview(input_path: Path, output_path: Path):
    if output_path.exists():
        logger.info(f"Hover preview already exists, skipping: {output_path}")
        return True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Short, low-res, muted, looping preview (e.g., 5-10 seconds)
    # -ss 5 (start at 5s), -t 5 (duration 5s)
    # -vf scale=320:-1 (scale width to 320, maintain aspect ratio)
    # -an (no audio)
    # -loop 0 (for MP4, this doesn't make it loop in player; player attribute needed)
    # We will rely on HTML <video loop> attribute for looping.
    command = [
        "ffmpeg", "-i", str(input_path.resolve()),
        "-ss", "2",  # Start 2 seconds in (avoid black frames/titles)
        "-t", "5",   # 5 seconds long preview
        "-vf", "scale=320:-2,crop=iw:min(ih\\,ih/9*16)", # Scale to 320px width, then crop to 16:9
        "-c:v", "libx264", "-preset", "ultrafast", # Fast encoding for previews
        "-crf", "28", # Higher CRF for smaller size
        "-an", # No audio
        "-movflags", "+faststart",
        str(output_path.resolve())
    ]
    return run_ffmpeg_command(command)

# --- Thumbnail and Duration Logic (largely unchanged, but paths updated) ---
def get_thumbnail_path(file_path: Path):
    slugified_stem = slugify_for_id(file_path.stem)
    return THUMBNAILS_DIR / f"{slugified_stem}.jpg"

def get_media_duration(file_path):
    try:
        if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            video = VideoFileClip(str(file_path.resolve()))
            duration = video.duration
            video.close()
            return str(timedelta(seconds=int(duration)))
        elif file_path.suffix.lower() in ['.mp3', '.wav', '.ogg']:
            audio = File(str(file_path.resolve()))
            if hasattr(audio.info, 'length'):
                return str(timedelta(seconds=int(audio.info.length)))
    except Exception as e:
        logger.error(f"Error getting duration for {file_path.resolve()}: {e}")
    return None

def _actually_create_thumbnail(file_path: Path, force_creation: bool = False):
    thumbnail_p = get_thumbnail_path(file_path)
    if not force_creation and thumbnail_p.exists():
        return f"/static/thumbnails/{thumbnail_p.name}"
    try:
        # Ensure the target directory exists
        thumbnail_p.parent.mkdir(parents=True, exist_ok=True)

        if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            video = VideoFileClip(str(file_path.resolve()))
            frame_array = video.get_frame(1) 
            video.close()
            img_frame = Image.fromarray(frame_array)
            img_frame.thumbnail((320, 180), Image.Resampling.LANCZOS)
            img_frame.save(thumbnail_p)
            logger.info(f"Generated thumbnail for video: {file_path.name}")
        elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif']:
            if not thumbnail_p.exists(): 
                img = Image.open(file_path.resolve())
                img.thumbnail((320, 180), Image.Resampling.LANCZOS)
                img.save(thumbnail_p)
                logger.info(f"Generated thumbnail for image: {file_path.name}")
        return f"/static/thumbnails/{thumbnail_p.name}" if thumbnail_p.exists() else None
    except Exception as e:
        logger.error(f"Error creating thumbnail for {file_path.resolve()}: {e}")
    return None

def get_display_thumbnail_url(file_path: Path, media_type: str):
    thumbnail_p = get_thumbnail_path(file_path)
    if thumbnail_p.exists():
        return f"/static/thumbnails/{thumbnail_p.name}"
    if media_type == 'video': return "/static/icons/generic-video-icon.svg"
    if media_type == 'image': 
        created_path = _actually_create_thumbnail(file_path)
        return created_path if created_path else "/static/icons/generic-image-icon.svg"
    if media_type == 'audio': return "/static/icons/generic-audio-icon.svg"
    return None

# --- Data Fetching Logic (Updated for Transcodes and Previews) ---
def get_preview_path(file_path: Path):
    slugified_stem = slugify_for_id(file_path.stem)
    return PREVIEWS_DIR / f"{slugified_stem}_preview.mp4"

def format_media_duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    return str(timedelta(seconds=int(seconds)))

def get_media_files_from_db(search_query: str = None):
    """Fetches media files from the database, optionally filtering by search_query."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM media_files"
    params = []
    if search_query:
        query += " WHERE filename LIKE ?"
        params.append(f"%{search_query}%")
    query += " ORDER BY date_added DESC"

    cursor.execute(query, params)
    db_rows = cursor.fetchall()
    conn.close()

    media_list = []
    for row in db_rows:
        # Convert DB row to the dictionary structure expected by templates
        # Ensure paths are converted to URLs for the frontend
        thumbnail_url = None
        if row['has_specific_thumbnail'] and row['thumbnail_path']:
            # Assuming thumbnail_path is stored relative to BASE_DIR e.g. "static/thumbnails/file.jpg"
            thumbnail_url = f"/{row['thumbnail_path'].replace('\\', '/')}" 
        else:
            # Fallback to generic icons based on media_type
            if row['media_type'] == 'video': thumbnail_url = "/static/icons/generic-video-icon.svg"
            elif row['media_type'] == 'image': thumbnail_url = "/static/icons/generic-image-icon.svg"
            elif row['media_type'] == 'audio': thumbnail_url = "/static/icons/generic-audio-icon.svg"
        
        playable_path_url = None
        original_path_for_download_url = f"/media_content/{row['filename']}" # Default to original
        
        if row['has_transcoded_version'] and row['transcoded_path']:
            playable_path_url = f"/media_content_transcoded/{Path(row['transcoded_path']).name}"
        else:
            playable_path_url = original_path_for_download_url

        preview_url = None
        if row['has_preview'] and row['preview_path']:
            preview_url = f"/{row['preview_path'].replace('\\', '/')}"

        display_name = row['user_title'] if row['user_title'] else row['filename']
        tags_list = json.loads(row['tags']) if row['tags'] else []

        media_item = {
            "id_db": row['id'], # Actual database ID
            "name": row['filename'], # Original filename, used for fetching/linking media content
            "display_title": display_name, # User title or filename for display
            "user_title": row['user_title'], # Actual user_title from DB
            "path": playable_path_url, # Path to play (transcoded if available, else original)
            "original_path_for_download": original_path_for_download_url,
            "type": row['media_type'],
            "thumbnail": thumbnail_url, # This should now be the correct URL or generic icon path
            "duration": format_media_duration(row['duration']), # Format seconds to HH:MM:SS
            "id": slugify_for_id(row['filename']), # slugify filename for html id
            "has_specific_thumbnail": bool(row['has_specific_thumbnail']),
            "has_transcoded_version": bool(row['has_transcoded_version']),
            "has_preview": bool(row['has_preview']),
            "preview_url": preview_url,
            "tags": tags_list,
            "size_bytes": row['size_bytes'],
            "resolution": json.loads(row['metadata_json']).get('resolution') if row['metadata_json'] else None,
            "original_full_path": row['original_path'] # Keep for backend operations if needed
        }
        media_list.append(media_item)
    
    return media_list

# Remove or comment out the old get_media_files function
# def get_media_files(search_query: str = None):
#     media_files = []
#     for file_path_obj in MEDIA_DIR.iterdir():
# ... (old implementation) ...
#     return media_files

def get_single_video_details_from_db(video_filename: str):
    """Fetches detailed information for a single video from the database by its filename."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # Assuming filename is unique in the media_files table for videos
    cursor.execute("SELECT * FROM media_files WHERE filename = ? AND media_type = 'video'", (video_filename,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    thumbnail_url = None
    if row['has_specific_thumbnail'] and row['thumbnail_path']:
        thumbnail_url = f"/{row['thumbnail_path'].replace('\\', '/')}"
    else:
        thumbnail_url = "/static/icons/generic-video-icon.svg" # Video specific

    playable_path_url = None
    original_path_for_download_url = f"/media_content/{row['filename']}"

    if row['has_transcoded_version'] and row['transcoded_path']:
        playable_path_url = f"/media_content_transcoded/{Path(row['transcoded_path']).name}"
    else:
        playable_path_url = original_path_for_download_url
    
    preview_url = None
    if row['has_preview'] and row['preview_path']:
        preview_url = f"/{row['preview_path'].replace('\\', '/')}"

    display_name = row['user_title'] if row['user_title'] else row['filename']
    tags_list = json.loads(row['tags']) if row['tags'] else []

    # Extract structured metadata from JSON if available
    metadata = json.loads(row['metadata_json']) if row['metadata_json'] else {}
    resolution = metadata.get('resolution', f"{row['width']}x{row['height']}" if row['width'] and row['height'] else 'N/A')
    fps = metadata.get('fps', row['fps'] if row['fps'] else 'N/A')

    video_details = {
        "id_db": row['id'],
        "name": row['filename'],
        "display_title": display_name,
        "user_title": row['user_title'],
        "path": playable_path_url,
        "original_path_for_download": original_path_for_download_url,
        "type": row['media_type'], # Should be 'video'
        "thumbnail": thumbnail_url,
        "duration": format_media_duration(row['duration']), # Assumes duration is in seconds
        "id": slugify_for_id(row['filename']),
        "has_specific_thumbnail": bool(row['has_specific_thumbnail']),
        "has_transcoded_version": bool(row['has_transcoded_version']),
        "has_preview": bool(row['has_preview']),
        "preview_url": preview_url,
        "tags": tags_list,
        "size": row['size_bytes'], # Raw bytes
        "resolution": resolution,
        "fps": fps,
        "original_full_path": row['original_path'] # For backend operations
        # Add any other specific fields needed for the video player page
    }
    return video_details

# Comment out or remove old get_single_video_details
# def get_single_video_details(video_name: str):
# ... (old implementation) ...
# return None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "title": "Media Browser"})

@app.get("/files", response_class=HTMLResponse)
async def list_files(request: Request, search: str = Query(None)):
    # Use the new database-driven function
    media_items = get_media_files_from_db(search_query=search)
    # Pass MEDIA_DIR_NAME and MEDIA_DIR to the template for the info message
    return templates.TemplateResponse(
        "file_list.html", 
        {
            "request": request, 
            "media_files": media_items, 
            "search_query": search or "",
            "MEDIA_DIR_NAME_FOR_TEMPLATE": MEDIA_DIR_NAME, # Pass the actual name used
            "MEDIA_DIR_PATH_FOR_TEMPLATE": str(MEDIA_DIR.resolve()) # Pass the resolved full path
        }
    )

@app.get("/media_content/{file_name:path}")
async def serve_media_file(file_name: str, request: Request):
    file_path = MEDIA_DIR / file_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Original file not found: {file_path}")
    return FileResponse(file_path, media_type=f"video/{file_path.suffix.lstrip('.').lower()}", filename=file_path.name)

@app.get("/media_content_transcoded/{file_name:path}")
async def serve_transcoded_media_file(file_name: str, request: Request):
    file_path = TRANSCODED_DIR / file_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Transcoded file not found: {file_path}")
    return FileResponse(file_path, media_type="video/mp4", filename=file_path.name) # Assuming MP4

@app.get("/video/{video_name:path}", response_class=HTMLResponse)
async def video_player_page(request: Request, video_name: str):
    video_details = get_single_video_details_from_db(video_name)
    if not video_details:
        raise HTTPException(status_code=404, detail="Video not found in database")
    
    # The following paths are used by buttons on the video player page
    # They need to be derived correctly based on the video_details from DB
    # (or the functions they call need to be updated to use DB info)
    
    # For thumbnail generation, ensure it uses original_full_path if needed
    # For transcoding, ensure it uses original_full_path
    
    return templates.TemplateResponse("video_player.html", {"request": request, "video": video_details})

@app.get("/tools/media-processing", response_class=HTMLResponse)
async def thumbnail_tools_page(request: Request): # Name matches url_for in index.html
    return templates.TemplateResponse("tools_page.html", {"request": request})

@app.post("/generate-thumbnail/{video_name:path}")
async def generate_specific_thumbnail_endpoint(video_name: str, response: Response):
    # Find the video in the database to get its original_path
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT original_path FROM media_files WHERE filename = ? AND media_type = 'video'", (video_name,))
    db_row = cursor.fetchone()
    # conn.close() # Close after all DB ops for this request

    if not db_row:
        # conn.close()
        raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found in database.")
    
    original_file_path = Path(db_row['original_path'])
    if not original_file_path.exists():
        conn.close()
        raise HTTPException(status_code=404, detail=f"Original media file for '{video_name}' not found at {original_file_path}.")

    thumbnail_url = _actually_create_thumbnail(original_file_path, force_creation=True)

    if thumbnail_url:
        # Update database
        actual_thumbnail_p = get_thumbnail_path(original_file_path)
        db_thumbnail_path_str = str(actual_thumbnail_p.relative_to(BASE_DIR)) if actual_thumbnail_p.exists() else None
        
        if db_thumbnail_path_str:
            try:
                # conn = get_db_connection() # Re-open if closed, or ensure it's open
                # cursor = conn.cursor()
                cursor.execute("UPDATE media_files SET thumbnail_path = ?, has_specific_thumbnail = TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename = ?", 
                               (db_thumbnail_path_str, video_name))
                conn.commit()
                logger.info(f"Database updated for new thumbnail of {video_name}.")
            except sqlite3.Error as e:
                logger.error(f"Failed to update database for thumbnail {video_name}: {e}")
                # Decide if we should raise an HTTP error or just log
        else:
            logger.warning(f"Thumbnail was reportedly created for {video_name}, but path was not resolvable for DB update.")

        response.headers["X-Thumbnail-Url"] = thumbnail_url
        return {"message": f"Thumbnail generated for {video_name}", "thumbnail_url": thumbnail_url}
    else:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to generate thumbnail for {video_name}")
    conn.close() # Ensure connection is closed

@app.post("/generate-all-video-thumbnails")
async def generate_all_thumbnails_endpoint():
    # Get all video files from DB that don't have a specific thumbnail
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename, original_path FROM media_files WHERE media_type = 'video' AND (has_specific_thumbnail = FALSE OR thumbnail_path IS NULL)")
    videos_to_process = cursor.fetchall()
    # conn.close() # Keep open for updates within the loop

    generated_count = 0
    failed_count = 0

    if not videos_to_process:
        conn.close()
        return {"message": "No video thumbnails to generate. All videos either have thumbnails or are not registered.", "generated": 0, "failed": 0}

    logger.info(f"Starting bulk thumbnail generation for {len(videos_to_process)} videos.")

    for video_row in videos_to_process:
        video_filename = video_row['filename']
        original_file_path = Path(video_row['original_path'])
        
        logger.info(f"Generating thumbnail for: {video_filename}")
        if not original_file_path.exists():
            logger.warning(f"Original file for {video_filename} not found at {original_file_path}, skipping thumbnail.")
            failed_count += 1
            continue

        thumbnail_url = _actually_create_thumbnail(original_file_path, force_creation=True)
        if thumbnail_url:
            actual_thumbnail_p = get_thumbnail_path(original_file_path)
            db_thumbnail_path_str = str(actual_thumbnail_p.relative_to(BASE_DIR)) if actual_thumbnail_p.exists() else None
            if db_thumbnail_path_str:
                try:
                    # cursor = conn.cursor() # Cursor should still be valid
                    cursor.execute("UPDATE media_files SET thumbnail_path = ?, has_specific_thumbnail = TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename = ?", 
                                   (db_thumbnail_path_str, video_filename))
                    conn.commit() # Commit after each successful generation and DB update
                    logger.info(f"DB updated for new thumbnail of {video_filename}.")
                    generated_count += 1
                except sqlite3.Error as e:
                    logger.error(f"DB update failed for thumbnail {video_filename}: {e}")
                    conn.rollback() # Rollback failed update for this file
                    failed_count += 1
            else:
                 logger.warning(f"Thumbnail created for {video_filename}, but path not resolvable for DB. Not counted as failure but needs review.")
                 failed_count +=1 # Or a new category like 'generated_no_db_update'
        else:
            logger.error(f"Failed to generate thumbnail for {video_filename}.")
            failed_count += 1
            
    conn.close()
    return {"message": f"Bulk thumbnail generation complete. Generated: {generated_count}, Failed: {failed_count}", "generated": generated_count, "failed": failed_count}

# --- NEW TRANSCODING ENDPOINTS ---
@app.post("/transcode-video/{video_name:path}")
async def transcode_specific_video_endpoint(video_name: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT original_path FROM media_files WHERE filename = ? AND media_type = 'video'", (video_name,))
    db_row = cursor.fetchone()

    if not db_row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found in database.")

    original_file_path = Path(db_row['original_path'])
    if not original_file_path.exists():
        conn.close()
        raise HTTPException(status_code=404, detail=f"Original media file for '{video_name}' not found at {original_file_path}.")

    # Default transcoding options for this simple endpoint
    # Output path will be based on slugified stem, consistent with get_media_files_from_db logic
    slugified_stem = slugify_for_id(original_file_path.stem)
    output_filename = f"{slugified_stem}.mp4" # Standardized name
    output_path = TRANSCODED_DIR / output_filename
    
    # Use default options for simple transcoding endpoint
    success = transcode_video(original_file_path, output_path) 

    if success:
        db_transcoded_path_str = str(output_path.relative_to(BASE_DIR)) if output_path.exists() else None
        if db_transcoded_path_str:
            try:
                cursor.execute("UPDATE media_files SET transcoded_path = ?, has_transcoded_version = TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename = ?", 
                               (db_transcoded_path_str, video_name))
                conn.commit()
                logger.info(f"Database updated for transcoded video {video_name}.")
                return {"message": f"Video {video_name} is being transcoded (or already exists). Refresh to see changes.", "output_path": db_transcoded_path_str}
            except sqlite3.Error as e:
                logger.error(f"DB update failed for transcoded video {video_name}: {e}")
                # If DB update fails, the file is transcoded but not reflected. Decide on error handling.
                conn.rollback()
                raise HTTPException(status_code=500, detail="Transcoding initiated, but database update failed.")
        else:
            # This case should ideally not happen if transcode_video reported success and file exists
            logger.error(f"Transcode reported success for {video_name} but output path {output_path} not found or not relative.")
            raise HTTPException(status_code=500, detail="Transcoding finished but output file issue occurred.")
    else:
        # transcode_video logs its own errors (ffmpeg not found, ffmpeg error)
        raise HTTPException(status_code=500, detail=f"Failed to start transcoding for {video_name}. Check server logs for FFmpeg errors.")
    conn.close()

@app.post("/transcode-video-advanced/{video_name:path}")
async def transcode_specific_video_advanced_endpoint(
    video_name: str,
    resolution: str = Form("720p"),
    quality_mode: str = Form("crf"),
    crf: str = Form("23"),
    video_bitrate: str = Form("2M"),
    audio_bitrate: str = Form("128k"),
    preset: str = Form("medium"),
    profile: str = Form("high")
):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT original_path FROM media_files WHERE filename = ? AND media_type = 'video'", (video_name,))
    db_row = cursor.fetchone()

    if not db_row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found in database.")

    original_file_path = Path(db_row['original_path'])
    if not original_file_path.exists():
        conn.close()
        raise HTTPException(status_code=404, detail=f"Original media file for '{video_name}' not found at {original_file_path}.")

    # Output path: Use a consistent naming convention, perhaps including options if they vary significantly
    # For simplicity, using a similar slugified approach as the basic transcode for now.
    # Advanced transcodes might warrant more descriptive names or a separate directory structure if multiple versions are kept.
    slugified_stem = slugify_for_id(original_file_path.stem)
    # Consider adding options to filename if multiple advanced transcodes are stored, e.g.: {slugified_stem}_{resolution}_{crf}.mp4
    # For now, assume one primary transcoded version per original.
    output_filename = f"{slugified_stem}.mp4" 
    output_path = TRANSCODED_DIR / output_filename

    options = {
        "resolution": resolution,
        "quality_mode": quality_mode,
        "crf": int(crf) if quality_mode == 'crf' else None,
        "video_bitrate": video_bitrate if quality_mode == 'bitrate' else None,
        "audio_bitrate": audio_bitrate,
        "preset": preset,
        "profile": profile
    }
    # Filter out None values for options not relevant to the chosen quality_mode
    active_options = {k: v for k, v in options.items() if v is not None}

    success = transcode_video(original_file_path, output_path, options=active_options)

    if success:
        db_transcoded_path_str = str(output_path.relative_to(BASE_DIR)) if output_path.exists() else None
        if db_transcoded_path_str:
            try:
                cursor.execute("UPDATE media_files SET transcoded_path = ?, has_transcoded_version = TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename = ?", 
                               (db_transcoded_path_str, video_name))
                conn.commit()
                logger.info(f"Database updated for advanced transcoded video {video_name}.")
                return {"message": f"Advanced transcoding for {video_name} initiated (or file already existed).", "output_path": db_transcoded_path_str, "options_used": active_options}
            except sqlite3.Error as e:
                logger.error(f"DB update failed for advanced transcoded video {video_name}: {e}")
                conn.rollback()
                raise HTTPException(status_code=500, detail="Advanced transcoding initiated, but database update failed.")
        else:
            logger.error(f"Advanced transcode for {video_name} reported success but output path {output_path} issue.")
            raise HTTPException(status_code=500, detail="Advanced transcoding finished but output file issue occurred.")
    else:
        raise HTTPException(status_code=500, detail=f"Failed to start advanced transcoding for {video_name}. Check logs.")
    conn.close()

@app.post("/transcode-all-videos")
async def transcode_all_videos_endpoint():
    # This endpoint will use default transcoding options.
    # The transcode_all_videos_with_options will allow specifying them.
    # For simplicity, let's make this a wrapper or share logic.
    # Calling the more general function with default options here.
    return await transcode_all_videos_with_options_logic(options=None) # Pass None to use defaults in transcode_video

@app.post("/transcode-all-videos-with-options")
async def transcode_all_videos_with_options_endpoint(
    resolution: str = Form("720p"),
    quality_mode: str = Form("crf"),
    crf: str = Form("23"),
    video_bitrate: str = Form("2M"),
    audio_bitrate: str = Form("128k"),
    preset: str = Form("medium")
    # profile is not in the form, will use default in transcode_video if not specified.
):
    options = {
        "resolution": resolution,
        "quality_mode": quality_mode,
        "crf": int(crf) if quality_mode == 'crf' else None,
        "video_bitrate": video_bitrate if quality_mode == 'bitrate' else None,
        "audio_bitrate": audio_bitrate,
        "preset": preset,
        # "profile": "high" # Default in transcode_video
    }
    active_options = {k: v for k, v in options.items() if v is not None}
    return await transcode_all_videos_with_options_logic(options=active_options)

async def transcode_all_videos_with_options_logic(options: dict = None):
    """Shared logic for bulk transcoding videos with specified or default options."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # Get videos that do not have a transcoded version or where the path is null
    cursor.execute("SELECT filename, original_path FROM media_files WHERE media_type = 'video' AND (has_transcoded_version = FALSE OR transcoded_path IS NULL)")
    videos_to_process = cursor.fetchall()
    # conn.close() # Keep open for updates in loop

    transcoded_count = 0
    failed_count = 0
    skipped_count = 0 # For files already existing that match criteria (handled by transcode_video)

    if not videos_to_process:
        conn.close()
        return {"message": "No videos to transcode or all compatible videos are already transcoded.", "processed": 0, "failed": 0, "skipped":0}

    logger.info(f"Starting bulk transcoding for {len(videos_to_process)} videos with options: {options or 'default'}")

    for video_row in videos_to_process:
        video_filename = video_row['filename']
        original_file_path = Path(video_row['original_path'])

        if not original_file_path.exists():
            logger.warning(f"Original file for {video_filename} not found at {original_file_path}, skipping transcoding.")
            failed_count += 1
            continue

        slugified_stem = slugify_for_id(original_file_path.stem)
        output_filename = f"{slugified_stem}.mp4"
        output_path = TRANSCODED_DIR / output_filename

        # transcode_video itself will skip if output_path exists. 
        # We rely on this, but if it skips, we should count it as skipped, not failed.
        # The transcode_video returns True if skipped or successfully transcoded.
        
        # Check if already exists before calling, to correctly update skipped_count
        # This is slightly redundant with transcode_video internal check but helps classify.
        if output_path.exists():
            logger.info(f"Transcoded file {output_path} already exists, ensuring DB is up to date.")
            # Ensure DB reflects this, might have been missed if server restarted during a previous bulk op
            db_transcoded_path_str = str(output_path.relative_to(BASE_DIR))
            try:
                cursor.execute("UPDATE media_files SET transcoded_path = ?, has_transcoded_version = TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename = ? AND (has_transcoded_version = FALSE OR transcoded_path IS NULL)", 
                               (db_transcoded_path_str, video_filename))
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(f"DB updated for pre-existing transcoded file: {video_filename}")
                else:
                    logger.info(f"DB already up-to-date for pre-existing transcoded file: {video_filename}")
                skipped_count += 1
                continue # Move to next file
            except sqlite3.Error as e:
                logger.error(f"DB update failed for pre-existing transcoded video {video_filename}: {e}")
                conn.rollback()
                failed_count +=1 # Count as fail if DB can't be updated
                continue

        logger.info(f"Starting transcode for: {video_filename} with options {options or 'default'}")
        success = transcode_video(original_file_path, output_path, options=options) 

        if success:
            # File was either newly transcoded or already existed and was skipped by transcode_video
            # The previous block handles explicit skips for already existing files for counting.
            # If success is true here, it means a new transcode occurred.
            db_transcoded_path_str = str(output_path.relative_to(BASE_DIR)) if output_path.exists() else None
            if db_transcoded_path_str:
                try:
                    cursor.execute("UPDATE media_files SET transcoded_path = ?, has_transcoded_version = TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename = ?", 
                                   (db_transcoded_path_str, video_filename))
                    conn.commit()
                    logger.info(f"DB updated for newly transcoded video {video_filename}.")
                    transcoded_count += 1
                except sqlite3.Error as e:
                    logger.error(f"DB update failed for newly transcoded video {video_filename}: {e}")
                    conn.rollback()
                    failed_count += 1 # Transcoded, but DB failed
            else:
                logger.error(f"Transcode of {video_filename} reported success, but output path {output_path} is problematic.")
                failed_count += 1
        else:
            logger.error(f"Failed to transcode {video_filename}.")
            failed_count += 1
            
    conn.close()
    return {"message": f"Bulk video transcoding complete. Newly Transcoded: {transcoded_count}, Failed: {failed_count}, Skipped (already existed): {skipped_count}", "transcoded": transcoded_count, "failed": failed_count, "skipped": skipped_count}

# Remove old async def transcode_all_videos(options=None): ...

@app.post("/generate-preview/{video_name:path}")
async def generate_specific_preview_endpoint(video_name: str, response: Response):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT original_path FROM media_files WHERE filename = ? AND media_type = 'video'", (video_name,))
    db_row = cursor.fetchone()

    if not db_row:
        conn.close()
        raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found in database.")

    original_file_path = Path(db_row['original_path'])
    if not original_file_path.exists():
        conn.close()
        raise HTTPException(status_code=404, detail=f"Original media file for '{video_name}' not found at {original_file_path}.")

    preview_p = get_preview_path(original_file_path) # This defines the standard preview path
    success = create_hover_preview(original_file_path, preview_p)

    if success:
        db_preview_path_str = str(preview_p.relative_to(BASE_DIR)) if preview_p.exists() else None
        preview_url = f"/{db_preview_path_str.replace('\\', '/')}" if db_preview_path_str else None
        if db_preview_path_str:
            try:
                cursor.execute("UPDATE media_files SET preview_path = ?, has_preview = TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename = ?", 
                               (db_preview_path_str, video_name))
                conn.commit()
                logger.info(f"Database updated for new preview of {video_name}.")
                response.headers["X-Preview-Url"] = preview_url
                return {"message": f"Hover preview generated for {video_name}", "preview_url": preview_url}
            except sqlite3.Error as e:
                logger.error(f"DB update failed for preview {video_name}: {e}")
                conn.rollback()
                raise HTTPException(status_code=500, detail="Preview generation completed, but database update failed.")
        else:
            logger.error(f"Preview generation for {video_name} reported success, but output path issue.")
            raise HTTPException(status_code=500, detail="Preview generation finished but output file issue occurred.")
    else:
        raise HTTPException(status_code=500, detail=f"Failed to generate hover preview for {video_name}. Check server logs.")
    conn.close()

@app.post("/generate-all-previews")
async def generate_all_previews_endpoint():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename, original_path FROM media_files WHERE media_type = 'video' AND (has_preview = FALSE OR preview_path IS NULL)")
    videos_to_process = cursor.fetchall()
    # conn.close() # Keep open for updates

    generated_count = 0
    failed_count = 0
    skipped_count = 0

    if not videos_to_process:
        conn.close()
        return {"message": "No video previews to generate. All videos either have previews or are not registered.", "generated": 0, "failed": 0, "skipped": 0}

    logger.info(f"Starting bulk preview generation for {len(videos_to_process)} videos.")

    for video_row in videos_to_process:
        video_filename = video_row['filename']
        original_file_path = Path(video_row['original_path'])
        preview_output_path = get_preview_path(original_file_path)

        if not original_file_path.exists():
            logger.warning(f"Original file for {video_filename} not found at {original_file_path}, skipping preview generation.")
            failed_count += 1
            continue
        
        # Check if preview already exists to manage skipped_count
        if preview_output_path.exists():
            logger.info(f"Preview file {preview_output_path} already exists, ensuring DB is up to date.")
            db_preview_path_str = str(preview_output_path.relative_to(BASE_DIR))
            try:
                cursor.execute("UPDATE media_files SET preview_path = ?, has_preview = TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename = ? AND (has_preview = FALSE OR preview_path IS NULL)", 
                               (db_preview_path_str, video_filename))
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(f"DB updated for pre-existing preview file: {video_filename}")
                else:
                     logger.info(f"DB already up-to-date for pre-existing preview file: {video_filename}")
                skipped_count += 1
                continue
            except sqlite3.Error as e:
                logger.error(f"DB update failed for pre-existing preview video {video_filename}: {e}")
                conn.rollback()
                failed_count +=1
                continue

        logger.info(f"Generating preview for: {video_filename}")
        success = create_hover_preview(original_file_path, preview_output_path)
        
        if success:
            db_preview_path_str = str(preview_output_path.relative_to(BASE_DIR)) if preview_output_path.exists() else None
            if db_preview_path_str:
                try:
                    cursor.execute("UPDATE media_files SET preview_path = ?, has_preview = TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename = ?", 
                                   (db_preview_path_str, video_filename))
                    conn.commit()
                    logger.info(f"DB updated for new preview of {video_filename}.")
                    generated_count += 1
                except sqlite3.Error as e:
                    logger.error(f"DB update failed for new preview {video_filename}: {e}")
                    conn.rollback()
                    failed_count += 1
            else:
                logger.error(f"Preview generation for {video_filename} reported success, but output path issue.")
                failed_count += 1
        else:
            logger.error(f"Failed to generate preview for {video_filename}.")
            failed_count += 1
            
    conn.close()
    return {"message": f"Bulk preview generation complete. Generated: {generated_count}, Failed: {failed_count}, Skipped: {skipped_count}", "generated": generated_count, "failed": failed_count, "skipped": skipped_count}

# Remove old async def generate_all_previews():
# ... (old implementation) ...

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    # Pass CWD for display purposes, and the default media subdir for JS fallback
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "CWD_DISPLAY_PATH": str(BASE_DIR).replace("\\", "/"), # For display in template help text
        "DEFAULT_MEDIA_SUBDIR_JS": DEFAULT_MEDIA_SUBDIR # For JS fallback
    })

@app.get("/settings/config", response_class=JSONResponse)
async def get_current_config():
    media_dir_name = get_setting("media_directory_name")
    return {"media_directory_name": media_dir_name or "media"}

@app.post("/settings/config")
async def update_app_config(request: Request):
    try:
        data = await request.json()
        new_media_dir_name = data.get("media_directory_name")
        
        if not new_media_dir_name or not isinstance(new_media_dir_name, str) or '/' in new_media_dir_name or '\\' in new_media_dir_name:
            raise HTTPException(status_code=400, detail="Invalid media directory name. Cannot be empty or contain path separators.")

        current_media_dir = get_setting("media_directory_name")
        
        if current_media_dir != new_media_dir_name:
            update_setting("media_directory_name", new_media_dir_name)
            logger.info(f"Media directory name updated in database to: {new_media_dir_name}. Application restart is required for changes to take full effect for MEDIA_DIR.")
            # Global MEDIA_DIR will only update on app restart.
            # Consider if we want to update it live or force a restart message.
            return {"message": f"Settings updated. Media directory changed to '{new_media_dir_name}'. Please restart the application for these changes to fully apply.", "requires_restart": True}
        
        return {"message": "Settings updated. No change in media directory name."}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Ensure the media directory (from DB or default) exists or can be created by user.
# This check is now more relevant after MEDIA_DIR is determined.
if not MEDIA_DIR.exists():
    logger.warning(f"The configured media directory {MEDIA_DIR} does not exist. Please create it or configure the correct path in Settings.")
    # Optionally, create it if it's the default and doesn't exist:
    # if MEDIA_DIR_NAME == "media": # Or your default name
    #     try:
    #         MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    #         logger.info(f"Default media directory {MEDIA_DIR} created.")
    #     except Exception as e:
    #         logger.error(f"Could not create default media directory {MEDIA_DIR}: {e}")
else:
    # If MEDIA_DIR exists, run the scan
    scan_media_directory_and_update_db()


class VideoMetadataUpdate(BaseModel):
    user_title: Optional[str] = None
    tags: Optional[List[str]] = None

@app.post("/video/{video_id_db}/metadata")
async def update_video_metadata(video_id_db: int, 
                                request: Request, 
                                user_title: Optional[str] = Form(None),
                                tags_str: Optional[str] = Form(None)): # Renamed from 'tags' to 'tags_str' to indicate it's a string
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT filename FROM media_files WHERE id = ? AND media_type = 'video'", (video_id_db,))
    video_exists = cursor.fetchone()
    if not video_exists:
        conn.close()
        raise HTTPException(status_code=404, detail="Video not found in database.")

    fields_to_update = {}
    if user_title is not None:
        fields_to_update["user_title"] = user_title.strip() if user_title.strip() else None
    
    parsed_tags_list = []
    if tags_str is not None: # Check if tags_str was provided in the form
        # Parse the comma-separated string from Choices.js/form input
        parsed_tags_list = sorted(list(set(tag.strip() for tag in tags_str.split(',') if tag.strip())))
        fields_to_update["tags"] = json.dumps(parsed_tags_list)

    if not fields_to_update:
        conn.close()
        # Fetch current details to re-render the sidebar even if no effective change, 
        # as HTMX expects a response to swap.
        # Alternatively, could return a 204 No Content and handle on client, but swapping is simpler.
        current_video_details = get_single_video_details_from_db(video_exists['filename'])
        if not current_video_details:
             # This case should ideally not happen if video_exists was found
             raise HTTPException(status_code=404, detail="Video details not found after attempted update.")
        return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": current_video_details})

    set_clause = ", ".join([f"{key} = ?" for key in fields_to_update.keys()])
    params = list(fields_to_update.values())
    params.append(video_id_db)

    try:
        cursor.execute(f"UPDATE media_files SET {set_clause}, last_scanned=CURRENT_TIMESTAMP WHERE id = ?", params)
        conn.commit()
        logger.info(f"Updated metadata for video ID {video_id_db}: {fields_to_update}")
    except sqlite3.Error as e:
        conn.rollback()
        logger.error(f"Database error updating metadata for video ID {video_id_db}: {e}")
        # Still try to return the sidebar with old data and an error message if possible
        # For simplicity, raising HTTP error for now.
        raise HTTPException(status_code=500, detail="Database error updating metadata.")
    finally:
        conn.close()
    
    updated_video_details = get_single_video_details_from_db(video_exists['filename']) # Fetch by original filename
    if not updated_video_details:
        # This case would be unusual if the update succeeded and video_exists was valid
        raise HTTPException(status_code=404, detail="Video details not found after successful update.")

    return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": updated_video_details})


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server. Media directory: {MEDIA_DIR}")
    # Ensure create_tables() is called before uvicorn.run if not at top level
    # create_tables() # Already called at module level
    uvicorn.run(app, host="0.0.0.0", port=8000) 