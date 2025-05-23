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
from datetime import timedelta
import json
import sqlite3
from database import get_db_connection, create_tables, get_setting, update_setting, db_connection
from typing import Optional, List
from config import (
    logger, templates, mount_static_files,
    BASE_DIR, MEDIA_DIR, TRANSCODED_DIR, PREVIEWS_DIR, THUMBNAILS_DIR,
    SUPPORTED_VIDEO_EXTENSIONS, SUPPORTED_AUDIO_EXTENSIONS, SUPPORTED_IMAGE_EXTENSIONS,
    slugify_for_id, safe_path_join, get_media_type_from_extension
)
from media_processing import (
    run_ffmpeg_command, transcode_video, create_hover_preview, 
    get_thumbnail_path, get_media_duration, _actually_create_thumbnail,
    get_display_thumbnail_url, get_preview_path
)
from data_access import (
    scan_media_directory_and_update_db,
    get_media_files_from_db, get_single_video_details_from_db
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for supported file types
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
SUPPORTED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.ogg'}
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif'}

app = FastAPI()

# --- Database Initialization ---
# This will create tables if they don't exist and initialize settings
# if the DB is new, potentially reading from config.json for the first run.
create_tables()


# --- Directory Setup & Configuration --- 
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MEDIA_SUBDIR = "media" # Define this for settings_page JS fallback

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
                if suffix in SUPPORTED_VIDEO_EXTENSIONS: 
                    media_type = 'video'
                elif suffix in SUPPORTED_AUDIO_EXTENSIONS: 
                    media_type = 'audio'
                elif suffix in SUPPORTED_IMAGE_EXTENSIONS: 
                    media_type = 'image'
                else:
                    continue  # Skip unsupported file types

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
                if media_type == 'video':
                    try:
                        with VideoFileClip(str(file_path_obj.resolve())) as clip:
                            width, height = clip.size
                            fps = clip.fps
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
            conn.rollback()
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

def safe_path_join(base_path: Path, filename: str) -> Path:
    """Safely join a base path with a filename, preventing directory traversal."""
    # Normalize the filename to prevent directory traversal
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
    """Get duration of media file in HH:MM:SS format."""
    try:
        suffix = file_path.suffix.lower()
        if suffix in SUPPORTED_VIDEO_EXTENSIONS:
            with VideoFileClip(str(file_path.resolve())) as video:
                duration = video.duration
                return str(timedelta(seconds=int(duration)))
        elif suffix in SUPPORTED_AUDIO_EXTENSIONS:
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

        suffix = file_path.suffix.lower()
        if suffix in SUPPORTED_VIDEO_EXTENSIONS:
            with VideoFileClip(str(file_path.resolve())) as video:
                frame_array = video.get_frame(1)
                img_frame = Image.fromarray(frame_array)
                img_frame.thumbnail((320, 180), Image.Resampling.LANCZOS)
                img_frame.save(thumbnail_p)
                logger.info(f"Generated thumbnail for video: {file_path.name}")
        elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    grid_size = get_setting("media_grid_size") or "medium"
    return templates.TemplateResponse("index.html", {"request": request, "title": "Media Browser", "grid_size": grid_size})

@app.get("/files", response_class=HTMLResponse)
async def list_files(request: Request, search: str = Query(None)):
    media_items = get_media_files_from_db(search_query=search)
    return templates.TemplateResponse(
        "file_list.html", 
        {
            "request": request, 
            "media_files": media_items, 
            "search_query": search or "",
            "MEDIA_DIR_NAME_FOR_TEMPLATE": MEDIA_DIR.name, 
            "MEDIA_DIR_PATH_FOR_TEMPLATE": str(MEDIA_DIR.resolve())
        }
    )

@app.get("/media_content/{file_name:path}")
async def serve_media_file(file_name: str, request: Request):
    file_path = safe_path_join(MEDIA_DIR, file_name)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Original file not found: {file_name}")
    
    media_type_str = get_media_type_from_extension(file_path)
    content_type = f"{media_type_str}/{file_path.suffix.lstrip('.').lower()}" if media_type_str != "unknown" else "application/octet-stream"
    
    return FileResponse(file_path, media_type=content_type, filename=file_path.name)

@app.get("/media_content_transcoded/{file_name:path}")
async def serve_transcoded_media_file(file_name: str, request: Request):
    file_path = safe_path_join(TRANSCODED_DIR, file_name)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Transcoded file not found: {file_name}")
    return FileResponse(file_path, media_type="video/mp4", filename=file_path.name)

@app.get("/video/{video_name:path}", response_class=HTMLResponse)
async def video_player_page(request: Request, video_name: str):
    video_details, video_queue, next_video = get_single_video_details_from_db(video_name)
    if not video_details:
        raise HTTPException(status_code=404, detail="Video not found in database")
    return templates.TemplateResponse("video_player.html", {"request": request, "video": video_details, "video_queue": video_queue, "next_video": next_video})

@app.get("/tools/media-processing", response_class=HTMLResponse)
async def thumbnail_tools_page(request: Request):
    return templates.TemplateResponse("tools_page.html", {"request": request})

@app.post("/generate-thumbnail/{video_name:path}")
async def generate_specific_thumbnail_endpoint(video_name: str, response: Response):
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT original_path FROM media_files WHERE filename = ? AND media_type = 'video'", (video_name,))
        db_row = cursor.fetchone()
        if not db_row:
            raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found in database.")

        original_file_path = Path(db_row['original_path'])
        if not original_file_path.exists():
            raise HTTPException(status_code=404, detail=f"Original media file for '{video_name}' not found at {original_file_path}.")

        thumbnail_url_rel = _actually_create_thumbnail(original_file_path, force_creation=True)

        if thumbnail_url_rel:
            actual_thumbnail_p = get_thumbnail_path(original_file_path)
            db_thumbnail_path_str = str(actual_thumbnail_p.relative_to(BASE_DIR)) if actual_thumbnail_p.exists() else None
            if db_thumbnail_path_str:
                try:
                    cursor.execute("UPDATE media_files SET thumbnail_path = ?, has_specific_thumbnail = TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename = ?", 
                                   (db_thumbnail_path_str, video_name))
                    logger.info(f"Database updated for new thumbnail of {video_name}.")
                except sqlite3.Error as e:
                    logger.error(f"Failed to update database for thumbnail {video_name}: {e}")
                    raise HTTPException(status_code=500, detail="Thumbnail generated but database update failed.")
            response.headers["X-Thumbnail-Url"] = thumbnail_url_rel
            return {"message": f"Thumbnail generated for {video_name}", "thumbnail_url": thumbnail_url_rel}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to generate thumbnail for {video_name}")

@app.post("/generate-all-video-thumbnails")
async def generate_all_thumbnails_endpoint():
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename, original_path FROM media_files WHERE media_type = 'video' AND (has_specific_thumbnail = FALSE OR thumbnail_path IS NULL)")
        videos_to_process = cursor.fetchall()
        generated_count = 0
        failed_count = 0
        if not videos_to_process:
            return {"message": "No video thumbnails to generate.", "generated": 0, "failed": 0}
        logger.info(f"Starting bulk thumbnail generation for {len(videos_to_process)} videos.")
        for video_row in videos_to_process:
            video_filename = video_row['filename']
            original_file_path = Path(video_row['original_path'])
            if not original_file_path.exists():
                logger.warning(f"Original for {video_filename} not found, skipping."); failed_count += 1; continue
            if _actually_create_thumbnail(original_file_path, force_creation=True):
                actual_thumbnail_p = get_thumbnail_path(original_file_path)
                db_thumb_path = str(actual_thumbnail_p.relative_to(BASE_DIR)) if actual_thumbnail_p.exists() else None
                if db_thumb_path:
                    try:
                        cursor.execute("UPDATE media_files SET thumbnail_path=?, has_specific_thumbnail=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_thumb_path, video_filename))
                        generated_count += 1
                    except sqlite3.Error as e: logger.error(f"DB update failed for thumb {video_filename}: {e}"); failed_count += 1
                else: logger.warning(f"Thumb created for {video_filename} but path issue."); failed_count += 1
            else: logger.error(f"Failed to gen thumb for {video_filename}."); failed_count += 1
    return {"message": f"Generated: {generated_count}, Failed: {failed_count}", "generated": generated_count, "failed": failed_count}

@app.post("/transcode-video/{video_name:path}")
async def transcode_specific_video_endpoint(video_name: str):
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT original_path FROM media_files WHERE filename = ? AND media_type = 'video'", (video_name,))
        db_row = cursor.fetchone()
        if not db_row: raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found.")
        original_file_path = Path(db_row['original_path'])
        if not original_file_path.exists(): raise HTTPException(status_code=404, detail=f"Original file for '{video_name}' not found.")
        output_path = TRANSCODED_DIR / f"{slugify_for_id(original_file_path.stem)}.mp4"
        if transcode_video(original_file_path, output_path):
            db_path = str(output_path.relative_to(BASE_DIR)) if output_path.exists() else None
            if db_path:
                try:
                    cursor.execute("UPDATE media_files SET transcoded_path=?, has_transcoded_version=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_path, video_name))
                    return {"message": f"Transcoding {video_name} started/exists.", "output_path": db_path}
                except sqlite3.Error as e: logger.error(f"DB update failed for transcode {video_name}: {e}"); raise HTTPException(status_code=500, detail="DB update failed post-transcode.")
            else: raise HTTPException(status_code=500, detail="Transcode success, but output file issue.")
        else: raise HTTPException(status_code=500, detail=f"Failed to transcode {video_name}. Check logs.")

@app.post("/transcode-video-advanced/{video_name:path}")
async def transcode_specific_video_advanced_endpoint(video_name: str, resolution: str=Form("720p"), quality_mode: str=Form("crf"), crf: str=Form("23"), video_bitrate: str=Form("2M"), audio_bitrate: str=Form("128k"), preset: str=Form("medium"), profile: str=Form("high")):
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT original_path FROM media_files WHERE filename = ? AND media_type = 'video'", (video_name,))
        db_row = cursor.fetchone()
        if not db_row: raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found.")
        original_file_path = Path(db_row['original_path'])
        if not original_file_path.exists(): raise HTTPException(status_code=404, detail=f"Original file for '{video_name}' not found.")
        
        output_path = TRANSCODED_DIR / f"{slugify_for_id(original_file_path.stem)}.mp4"
        options = {"resolution": resolution, "quality_mode": quality_mode, "crf": int(crf), "video_bitrate": video_bitrate, "audio_bitrate": audio_bitrate, "preset": preset, "profile": profile}
        active_options = {k: v for k, v in options.items() if v is not None and (k!='crf' or quality_mode=='crf') and (k!='video_bitrate' or quality_mode=='bitrate')}

        if transcode_video(original_file_path, output_path, options=active_options):
            db_path = str(output_path.relative_to(BASE_DIR)) if output_path.exists() else None
            if db_path:
                try:
                    cursor.execute("UPDATE media_files SET transcoded_path=?, has_transcoded_version=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_path, video_name))
                    return {"message": f"Advanced transcoding for {video_name} initiated.", "output_path": db_path, "options_used": active_options}
                except sqlite3.Error as e: logger.error(f"DB update failed for adv transcode {video_name}: {e}"); raise HTTPException(status_code=500, detail="DB update failed post-adv-transcode.")
            else: raise HTTPException(status_code=500, detail="Adv transcode success, but output file issue.")
        else: raise HTTPException(status_code=500, detail=f"Failed to start adv transcode for {video_name}. Check logs.")

async def _bulk_transcode_logic(options: Optional[dict] = None):
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename, original_path FROM media_files WHERE media_type='video' AND (has_transcoded_version=FALSE OR transcoded_path IS NULL)")
        videos = cursor.fetchall(); count=0; failed=0; skipped=0
        if not videos: return {"message": "No videos need transcoding.", "transcoded":0, "failed":0, "skipped":0}
        logger.info(f"Bulk transcode for {len(videos)} videos, options: {options or 'default'}")
        for video_row in videos:
            orig_path = Path(video_row['original_path']); filename = video_row['filename']
            if not orig_path.exists(): logger.warning(f"Original for {filename} not found."); failed+=1; continue
            out_path = TRANSCODED_DIR / f"{slugify_for_id(orig_path.stem)}.mp4"
            if out_path.exists(): # Already exists, ensure DB is up to date
                db_path_str = str(out_path.relative_to(BASE_DIR))
                try:
                    cursor.execute("UPDATE media_files SET transcoded_path=?, has_transcoded_version=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=? AND (has_transcoded_version=FALSE OR transcoded_path IS NULL)", (db_path_str, filename))
                    if cursor.rowcount > 0: logger.info(f"DB updated for existing transcode: {filename}")
                    skipped+=1; continue
                except sqlite3.Error as e: logger.error(f"DB update for existing transcode {filename} failed: {e}"); failed+=1; continue
            
            if transcode_video(orig_path, out_path, options=options):
                db_path = str(out_path.relative_to(BASE_DIR)) if out_path.exists() else None
                if db_path:
                    try: cursor.execute("UPDATE media_files SET transcoded_path=?, has_transcoded_version=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_path, filename)); count+=1
                    except sqlite3.Error as e: logger.error(f"DB update for {filename} failed: {e}"); failed+=1
                else: logger.error(f"Transcode {filename} success, but output path issue."); failed+=1
            else: logger.error(f"Failed to transcode {filename}."); failed+=1
    return {"message": f"Transcoded: {count}, Failed: {failed}, Skipped: {skipped}", "transcoded":count, "failed":failed, "skipped":skipped}

@app.post("/transcode-all-videos")
async def transcode_all_videos_endpoint(): return await _bulk_transcode_logic(options=None)

@app.post("/transcode-all-videos-with-options")
async def transcode_all_videos_with_options_endpoint(resolution:str=Form("720p"), quality_mode:str=Form("crf"), crf:str=Form("23"), video_bitrate:str=Form("2M"), audio_bitrate:str=Form("128k"), preset:str=Form("medium")):
    options = {"resolution":resolution, "quality_mode":quality_mode, "crf":int(crf) if quality_mode=='crf' else None, "video_bitrate":video_bitrate if quality_mode=='bitrate' else None, "audio_bitrate":audio_bitrate, "preset":preset}
    return await _bulk_transcode_logic(options={k:v for k,v in options.items() if v is not None})

@app.post("/generate-preview/{video_name:path}")
async def generate_specific_preview_endpoint(video_name: str, response: Response):
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT original_path FROM media_files WHERE filename=? AND media_type='video'", (video_name,))
        db_row = cursor.fetchone()
        if not db_row: raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found.")
        orig_path = Path(db_row['original_path'])
        if not orig_path.exists(): raise HTTPException(status_code=404, detail=f"Original for '{video_name}' not found.")
        preview_p = get_preview_path(orig_path)
        if create_hover_preview(orig_path, preview_p):
            db_path = str(preview_p.relative_to(BASE_DIR)) if preview_p.exists() else None
            url = f"/{db_path.replace('\\ ', '/')}" if db_path else None
            if db_path:
                try:
                    cursor.execute("UPDATE media_files SET preview_path=?, has_preview=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_path, video_name))
                    response.headers["X-Preview-Url"] = url
                    return {"message": f"Preview for {video_name} generated.", "preview_url": url}
                except sqlite3.Error as e: logger.error(f"DB update for preview {video_name} failed: {e}"); raise HTTPException(status_code=500, detail="DB update failed post-preview.")
            else: raise HTTPException(status_code=500, detail="Preview success, output file issue.")
        else: raise HTTPException(status_code=500, detail=f"Failed to gen preview for {video_name}. Check logs.")

@app.post("/generate-all-previews")
async def generate_all_previews_endpoint():
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename, original_path FROM media_files WHERE media_type='video' AND (has_preview=FALSE OR preview_path IS NULL)")
        videos = cursor.fetchall(); gen=0; failed=0; skipped=0
        if not videos: return {"message": "No previews to generate.", "generated":0, "failed":0, "skipped":0}
        logger.info(f"Bulk preview gen for {len(videos)} videos.")
        for video_row in videos:
            filename = video_row['filename']; orig_path = Path(video_row['original_path'])
            if not orig_path.exists(): logger.warning(f"Original for {filename} not found."); failed+=1; continue
            preview_p = get_preview_path(orig_path)
            if preview_p.exists(): # Already exists, ensure DB up to date
                db_path = str(preview_p.relative_to(BASE_DIR))
                try:
                    cursor.execute("UPDATE media_files SET preview_path=?, has_preview=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=? AND (has_preview=FALSE OR preview_path IS NULL)", (db_path, filename))
                    if cursor.rowcount > 0: logger.info(f"DB updated for existing preview: {filename}")
                    skipped+=1; continue
                except sqlite3.Error as e: logger.error(f"DB update for existing preview {filename} failed: {e}"); failed+=1; continue
            
            if create_hover_preview(orig_path, preview_p):
                db_path_str = str(preview_p.relative_to(BASE_DIR)) if preview_p.exists() else None
                if db_path_str:
                    try: cursor.execute("UPDATE media_files SET preview_path=?, has_preview=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_path_str, filename)); gen+=1
                    except sqlite3.Error as e: logger.error(f"DB update for {filename} preview failed: {e}"); failed+=1
                else: logger.error(f"Preview gen {filename} success, output path issue."); failed+=1
            else: logger.error(f"Failed to gen preview for {filename}."); failed+=1
    return {"message": f"Generated: {gen}, Failed: {failed}, Skipped: {skipped}", "generated":gen, "failed":failed, "skipped":skipped}

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    from config import DEFAULT_MEDIA_SUBDIR # Local import to avoid clutter at top for one-off use
    return templates.TemplateResponse("settings.html", {"request": request, "CWD_DISPLAY_PATH": str(BASE_DIR).replace("\\", "/"), "DEFAULT_MEDIA_SUBDIR_JS": DEFAULT_MEDIA_SUBDIR})

@app.get("/settings/config", response_class=JSONResponse)
async def get_current_config():
    from database import get_setting # Local import
    media_dir_name = get_setting("media_directory_name")
    grid_size = get_setting("media_grid_size")
    return {
        "media_directory_name": media_dir_name or "media",
        "media_grid_size": grid_size or "medium"
    }

@app.post("/settings/config")
async def update_app_config(request: Request):
    from database import update_setting, get_setting # Local import
    try:
        data = await request.json()
        new_media_dir_name = data.get("media_directory_name")
        new_grid_size = data.get("media_grid_size")
        
        requires_restart = False
        messages = []
        
        # Handle media directory name change
        if new_media_dir_name:
            if not isinstance(new_media_dir_name, str) or '/' in new_media_dir_name or '\\' in new_media_dir_name:
                raise HTTPException(status_code=400, detail="Invalid media dir name.")
            if get_setting("media_directory_name") != new_media_dir_name:
                update_setting("media_directory_name", new_media_dir_name)
                messages.append(f"Media dir changed to '{new_media_dir_name}'.")
                requires_restart = True
        
        # Handle grid size change
        if new_grid_size:
            if new_grid_size not in ["small", "medium", "large"]:
                raise HTTPException(status_code=400, detail="Invalid grid size.")
            if get_setting("media_grid_size") != new_grid_size:
                update_setting("media_grid_size", new_grid_size)
                messages.append(f"Grid size changed to '{new_grid_size}'.")
        
        if not messages:
            return {"message": "Settings updated. No changes made."}
        
        message = " ".join(messages)
        if requires_restart:
            message += " Restart required for media directory change."
            
        return {"message": message, "requires_restart": requires_restart}
    except json.JSONDecodeError: raise HTTPException(status_code=400, detail="Invalid JSON.")
    except Exception as e: logger.error(f"Error updating config: {e}"); raise HTTPException(status_code=500, detail=str(e))

@app.post("/settings/grid-size")
async def update_grid_size(grid_size: str = Form(...)):
    from database import update_setting, get_setting # Local import
    if grid_size not in ["small", "medium", "large"]:
        raise HTTPException(status_code=400, detail="Invalid grid size.")
    
    current_size = get_setting("media_grid_size")
    if current_size != grid_size:
        update_setting("media_grid_size", grid_size)
        return {"message": f"Grid size updated to {grid_size}", "grid_size": grid_size}
    
    return {"message": "Grid size unchanged", "grid_size": grid_size}

@app.post("/video/{video_id_db}/metadata")
async def update_video_metadata(video_id_db: int, request: Request, user_title: Optional[str] = Form(None), tags_str: Optional[str] = Form(None)):
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename FROM media_files WHERE id = ? AND media_type = 'video'", (video_id_db,))
        video_exists = cursor.fetchone()
        if not video_exists: raise HTTPException(status_code=404, detail="Video not found.")

        fields_to_update = {}
        if user_title is not None: fields_to_update["user_title"] = user_title.strip() if user_title.strip() else None
        if tags_str is not None:
            parsed_tags = sorted(list(set(tag.strip() for tag in tags_str.split(',') if tag.strip())))
            fields_to_update["tags"] = json.dumps(parsed_tags)

        if not fields_to_update:
            current_video_details = get_single_video_details_from_db(video_exists['filename'])
            if not current_video_details: raise HTTPException(status_code=404, detail="Video details not found.")
            return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": current_video_details})

        set_clause = ", ".join([f"{key} = ?" for key in fields_to_update.keys()])
        params = list(fields_to_update.values()) + [video_id_db]
        try:
            cursor.execute(f"UPDATE media_files SET {set_clause}, last_scanned=CURRENT_TIMESTAMP WHERE id = ?", params)
            logger.info(f"Updated metadata for video ID {video_id_db}: {fields_to_update}")
        except sqlite3.Error as e: logger.error(f"DB error for video ID {video_id_db}: {e}"); raise HTTPException(status_code=500, detail="DB error.")
    
    updated_video_details = get_single_video_details_from_db(video_exists['filename'])
    if not updated_video_details: raise HTTPException(status_code=404, detail="Video details not found post-update.")
    return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": updated_video_details})

@app.get("/video/{current_video_id_db}/next")
async def get_next_video(current_video_id_db: int, response: Response):
    """Finds the next video in the queue and redirects to its player page."""
    logger.info(f"Received request for next video after ID: {current_video_id_db}")
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Fetch all video files, ordered by filename (the queue)
        cursor.execute("SELECT id, filename FROM media_files WHERE media_type = 'video' ORDER BY filename ASC")
        all_videos_rows = cursor.fetchall()

    video_filenames_ordered = [row['filename'] for row in all_videos_rows]
    current_video_index = -1

    # Find the index of the current video by its ID
    for i, row in enumerate(all_videos_rows):
        if row['id'] == current_video_id_db:
            current_video_index = i
            break

    if current_video_index != -1 and current_video_index + 1 < len(video_filenames_ordered):
        next_video_filename = video_filenames_ordered[current_video_index + 1]
        # Redirect to the next video's player page using HX-Redirect header
        logger.info(f"Found next video: {next_video_filename}. Redirecting.")
        response.headers["HX-Redirect"] = f"/video/{next_video_filename}"
        return {"message": f"Redirecting to next video: {next_video_filename}"}
    else:
        # No next video found (either current video not in list or it's the last one)
        logger.warning(f"No next video found after ID {current_video_id_db} or current video not in queue.")
        raise HTTPException(status_code=404, detail="No next video found in the queue.")

# Startup scan
if MEDIA_DIR.exists() and MEDIA_DIR.is_dir():
    scan_media_directory_and_update_db()
elif not MEDIA_DIR.exists():
    logger.warning(f"Configured media directory {MEDIA_DIR} does not exist. Create or configure in Settings.")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting PABT server. Media directory: {MEDIA_DIR}. Access at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 