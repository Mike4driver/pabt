from fastapi import FastAPI, Request, Query, HTTPException, Response, Form, BackgroundTasks
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
from datetime import timedelta, datetime
import json
import sqlite3
import threading
import uuid
from typing import Optional, List

# --- Early Database Initialization ---
# Must happen before importing 'config' or other modules that read settings at import time.
import database
database.create_tables()

# --- Subsequent Imports (including those that might use the database via config) ---
from database import get_db_connection, get_setting, update_setting, db_connection

from config import (
    logger, templates, mount_static_files, # templates is also initialized in config now
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
    get_media_files_from_db, get_single_video_details_from_db, get_all_tags_from_db
)

# Set up logging - logger is now imported from config
# logging.basicConfig(level=logging.INFO) # This might be redundant if config.py handles it
# logger = logging.getLogger(__name__) # logger is now imported from config

app = FastAPI() # Initialize FastAPI app instance here

# --- Background Job Tracking System ---
background_jobs = {}
job_lock = threading.Lock()

class BackgroundJob:
    def __init__(self, job_id: str, job_type: str, description: str):
        self.job_id = job_id
        self.job_type = job_type
        self.description = description
        self.status = "running"
        self.progress = 0
        self.total = 0
        self.current_item = ""
        self.start_time = datetime.now()
        self.end_time = None
        self.result = {}
        self.error = None

    def update_progress(self, current: int, total: int, current_item: str = ""):
        self.progress = current
        self.total = total
        self.current_item = current_item

    def complete(self, result: dict):
        self.status = "completed"
        self.end_time = datetime.now()
        self.result = result
        self.progress = self.total

    def fail(self, error: str):
        self.status = "failed"
        self.end_time = datetime.now()
        self.error = error

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "description": self.description,
            "status": self.status,
            "progress": self.progress,
            "total": self.total,
            "current_item": self.current_item,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "result": self.result,
            "error": self.error
        }

def create_background_job(job_type: str, description: str) -> BackgroundJob:
    job_id = str(uuid.uuid4())
    job = BackgroundJob(job_id, job_type, description)
    with job_lock:
        background_jobs[job_id] = job
    return job

def get_background_job(job_id: str) -> Optional[BackgroundJob]:
    with job_lock:
        return background_jobs.get(job_id)

def cleanup_old_jobs():
    """Remove jobs older than 1 hour"""
    cutoff = datetime.now() - timedelta(hours=1)
    with job_lock:
        to_remove = []
        for job_id, job in background_jobs.items():
            if job.end_time and job.end_time < cutoff:
                to_remove.append(job_id)
        for job_id in to_remove:
            del background_jobs[job_id]

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
    per_page = get_setting("per_page") or "20"
    return templates.TemplateResponse("index.html", {"request": request, "title": "Media Browser", "grid_size": grid_size, "per_page": per_page})

@app.get("/files", response_class=HTMLResponse)
async def list_files(
    request: Request, 
    search: str = Query(None), 
    page: int = Query(1, ge=1), 
    per_page: int = Query(None, ge=1, le=100),
    media_type: str = Query(None),
    tags: str = Query(None),
    sort_by: str = Query("date_added"),
    sort_order: str = Query("desc")
):
    # Use saved per_page setting as default if not provided
    if per_page is None:
        per_page = int(get_setting("per_page") or "20")
    
    # Parse tags parameter (comma-separated string to list)
    tags_filter = None
    if tags:
        tags_filter = [tag.strip() for tag in tags.split(',') if tag.strip()]
    
    result = get_media_files_from_db(
        search_query=search, 
        page=page, 
        per_page=per_page,
        media_type_filter=media_type,
        tags_filter=tags_filter,
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    return templates.TemplateResponse(
        "file_list.html", 
        {
            "request": request, 
            "media_files": result['media_files'], 
            "pagination": result['pagination'],
            "search_query": search or "",
            "current_media_type": media_type or "",
            "current_tags": tags or "",
            "current_sort_by": sort_by,
            "current_sort_order": sort_order,
            "MEDIA_DIR_NAME_FOR_TEMPLATE": MEDIA_DIR.name, 
            "MEDIA_DIR_PATH_FOR_TEMPLATE": str(MEDIA_DIR.resolve())
        }
    )

@app.get("/api/tags", response_class=JSONResponse)
async def get_all_tags():
    """API endpoint to get all available tags for autocomplete/suggestions"""
    tags = get_all_tags_from_db()
    return {"tags": tags}

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
    
    # Get video player settings
    autoplay_enabled = get_setting("autoplay_enabled") or "true"
    default_muted = get_setting("default_muted") or "true"
    autoplay_next = get_setting("autoplay_next") or "false"
    auto_replay = get_setting("auto_replay") or "false"
    
    return templates.TemplateResponse("video_player.html", {
        "request": request, 
        "video": video_details, 
        "video_queue": video_queue, 
        "next_video": next_video,
        "autoplay_enabled": autoplay_enabled == "true",
        "default_muted": default_muted == "true",
        "autoplay_next": autoplay_next == "true",
        "auto_replay": auto_replay == "true"
    })

@app.get("/tools/media-processing", response_class=HTMLResponse)
async def thumbnail_tools_page(request: Request):
    return templates.TemplateResponse("tools_page.html", {"request": request})

# --- Background Job Status Endpoints ---

@app.get("/jobs", response_class=JSONResponse)
async def get_all_jobs():
    """Get status of all background jobs"""
    cleanup_old_jobs()
    with job_lock:
        jobs = [job.to_dict() for job in background_jobs.values()]
    return {"jobs": jobs}

@app.get("/jobs/monitor", response_class=HTMLResponse)
async def background_jobs_page(request: Request):
    """Background jobs monitoring page"""
    return templates.TemplateResponse("background_jobs.html", {"request": request})

@app.get("/jobs/{job_id}", response_class=JSONResponse)
async def get_job_status(job_id: str):
    """Get status of a specific background job"""
    job = get_background_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel/remove a background job"""
    with job_lock:
        if job_id in background_jobs:
            job = background_jobs[job_id]
            if job.status == "running":
                job.fail("Cancelled by user")
            del background_jobs[job_id]
            return {"message": "Job cancelled"}
    raise HTTPException(status_code=404, detail="Job not found")

@app.post("/generate-thumbnail/{video_name:path}")
async def generate_specific_thumbnail_endpoint(video_name: str, response: Response, request: Request):
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
                    # For consistency, even if DB update fails, try to re-render sidebar to show current state or error.
                    # However, this might show an inconsistent state if the DB update failed.
                    # A more robust solution would be to handle this error state explicitly in the template.
                    updated_video_details, _, _ = get_single_video_details_from_db(video_name)
                    if not updated_video_details: 
                        raise HTTPException(status_code=500, detail="Thumbnail generated, DB update failed, and failed to reload video details.")
                    return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": updated_video_details, "error_message": "Thumbnail generated but DB update failed."})
            
            response.headers["X-Thumbnail-Url"] = thumbnail_url_rel # This is still useful for JS if needed
            # Re-fetch details and re-render the sidebar
            updated_video_details, _, _ = get_single_video_details_from_db(video_name)
            if not updated_video_details: 
                 raise HTTPException(status_code=404, detail="Video details not found after thumbnail generation.")
            return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": updated_video_details})
        else:
            # If thumbnail generation itself failed, re-render with an error message
            current_video_details, _, _ = get_single_video_details_from_db(video_name)
            if not current_video_details: 
                 raise HTTPException(status_code=500, detail="Failed to generate thumbnail and also failed to reload video details.")
            return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": current_video_details, "error_message": "Failed to generate thumbnail. Check logs."})

@app.post("/generate-all-video-thumbnails")
async def generate_all_thumbnails_endpoint(background_tasks: BackgroundTasks):
    """Start thumbnail generation in background"""
    job = create_background_job("thumbnails", "Generating video thumbnails")
    background_tasks.add_task(generate_all_thumbnails_background, job.job_id)
    return {"message": "Thumbnail generation started in background", "job_id": job.job_id}

def generate_all_thumbnails_background(job_id: str):
    """Background task for generating all thumbnails"""
    job = get_background_job(job_id)
    if not job:
        return
    
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename, original_path FROM media_files WHERE media_type = 'video' AND (has_specific_thumbnail = FALSE OR thumbnail_path IS NULL)")
            videos_to_process = cursor.fetchall()
            
            if not videos_to_process:
                job.complete({"message": "No video thumbnails to generate.", "generated": 0, "failed": 0})
                return
            
            job.update_progress(0, len(videos_to_process), "Starting...")
            logger.info(f"Starting background thumbnail generation for {len(videos_to_process)} videos.")
            
            generated_count = 0
            failed_count = 0
            
            for i, video_row in enumerate(videos_to_process):
                video_filename = video_row['filename']
                original_file_path = Path(video_row['original_path'])
                
                job.update_progress(i, len(videos_to_process), f"Processing {video_filename}")
                
                if not original_file_path.exists():
                    logger.warning(f"Original for {video_filename} not found, skipping.")
                    failed_count += 1
                    continue
                
                if _actually_create_thumbnail(original_file_path, force_creation=True):
                    actual_thumbnail_p = get_thumbnail_path(original_file_path)
                    db_thumb_path = str(actual_thumbnail_p.relative_to(BASE_DIR)) if actual_thumbnail_p.exists() else None
                    if db_thumb_path:
                        try:
                            cursor.execute("UPDATE media_files SET thumbnail_path=?, has_specific_thumbnail=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_thumb_path, video_filename))
                            # conn.commit() # Commit after each thumbnail update - REMOVED
                            generated_count += 1
                        except sqlite3.Error as e:
                            logger.error(f"DB update failed for thumb {video_filename}: {e}")
                            failed_count += 1
                    else:
                        logger.warning(f"Thumb created for {video_filename} but path issue or does not exist after creation.")
                        failed_count += 1
                else:
                    logger.error(f"Failed to gen thumb for {video_filename}.")
                    failed_count += 1
            
            conn.commit() # Commit all updates after loop
            job.complete({"message": f"Generated: {generated_count}, Failed: {failed_count}", "generated": generated_count, "failed": failed_count})
            
    except Exception as e:
        logger.error(f"Thumbnail generation background task failed: {e}")
        job.fail(str(e))

@app.post("/transcode-video/{video_name:path}")
async def transcode_specific_video_endpoint(video_name: str, background_tasks: BackgroundTasks):
    """Start individual video transcoding in background"""
    job = create_background_job("transcode_single", f"Transcoding {video_name}")
    background_tasks.add_task(transcode_single_video_background, job.job_id, video_name, None)
    return {"message": f"Transcoding {video_name} started in background", "job_id": job.job_id}

@app.post("/transcode-video-advanced/{video_name:path}")
async def transcode_specific_video_advanced_endpoint(
    video_name: str, 
    background_tasks: BackgroundTasks,
    resolution: str = Form("720p"), 
    quality_mode: str = Form("crf"), 
    crf: str = Form("23"), 
    video_bitrate: str = Form("2M"), 
    audio_bitrate: str = Form("128k"), 
    preset: str = Form("medium"), 
    profile: str = Form("high")
):
    """Start individual video advanced transcoding in background"""
    options = {
        "resolution": resolution, 
        "quality_mode": quality_mode, 
        "crf": int(crf), 
        "video_bitrate": video_bitrate, 
        "audio_bitrate": audio_bitrate, 
        "preset": preset, 
        "profile": profile
    }
    active_options = {k: v for k, v in options.items() if v is not None and (k!='crf' or quality_mode=='crf') and (k!='video_bitrate' or quality_mode=='bitrate')}
    
    job = create_background_job("transcode_single_advanced", f"Advanced transcoding {video_name}")
    background_tasks.add_task(transcode_single_video_background, job.job_id, video_name, active_options)
    return {"message": f"Advanced transcoding for {video_name} started in background", "job_id": job.job_id, "options": active_options}

def transcode_single_video_background(job_id: str, video_name: str, options: Optional[dict] = None):
    """Background task for transcoding a single video"""
    job = get_background_job(job_id)
    if not job:
        return
    
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT original_path FROM media_files WHERE filename = ? AND media_type = 'video'", (video_name,))
            db_row = cursor.fetchone()
            
            if not db_row:
                job.fail(f"Video '{video_name}' not found in database")
                return
            
            original_file_path = Path(db_row['original_path'])
            if not original_file_path.exists():
                job.fail(f"Original file for '{video_name}' not found at {original_file_path}")
                return
            
            job.update_progress(0, 1, "Starting transcoding...")
        
            output_path = TRANSCODED_DIR / f"{slugify_for_id(original_file_path.stem)}.mp4"
            
            # Check if already exists
            if output_path.exists():
                job.update_progress(1, 1, "File already exists, updating database...")
                db_path = str(output_path.relative_to(BASE_DIR))
                try:
                    cursor.execute("UPDATE media_files SET transcoded_path=?, has_transcoded_version=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_path, video_name))
                    conn.commit() # Commit after update
                    job.complete({"message": f"Transcoded version already exists for {video_name}", "output_path": db_path, "video_name": video_name})
                    return
                except sqlite3.Error as e:
                    job.fail(f"Database update failed: {e}")
                    return
            
            job.update_progress(1, 1, f"Transcoding {video_name}...")
            
            if transcode_video(original_file_path, output_path, options=options):
                db_path = str(output_path.relative_to(BASE_DIR)) if output_path.exists() else None
                if db_path:
                    try:
                        cursor.execute("UPDATE media_files SET transcoded_path=?, has_transcoded_version=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_path, video_name))
                        conn.commit() # Commit after update
                        job.complete({"message": f"Successfully transcoded {video_name}", "output_path": db_path, "video_name": video_name})
                    except sqlite3.Error as e:
                        job.fail(f"Transcoding succeeded but database update failed: {e}")
                else:
                    job.fail("Transcoding completed but output file not found or path issue")
            else:
                job.fail(f"FFmpeg transcoding failed for {video_name}")
                
    except Exception as e:
        logger.error(f"Single video transcoding background task failed: {e}")
        job.fail(str(e))

@app.post("/transcode-all-videos")
async def transcode_all_videos_endpoint(background_tasks: BackgroundTasks):
    """Start transcoding in background"""
    job = create_background_job("transcode", "Transcoding all videos")
    background_tasks.add_task(bulk_transcode_background, job.job_id, None)
    return {"message": "Transcoding started in background", "job_id": job.job_id}

@app.post("/transcode-all-videos-with-options")
async def transcode_all_videos_with_options_endpoint(
    background_tasks: BackgroundTasks,
    resolution: str = Form("720p"), 
    quality_mode: str = Form("crf"), 
    crf: str = Form("23"), 
    video_bitrate: str = Form("2M"), 
    audio_bitrate: str = Form("128k"), 
    preset: str = Form("medium")
):
    """Start transcoding with options in background"""
    options = {
        "resolution": resolution, 
        "quality_mode": quality_mode, 
        "crf": int(crf) if quality_mode == 'crf' else None, 
        "video_bitrate": video_bitrate if quality_mode == 'bitrate' else None, 
        "audio_bitrate": audio_bitrate, 
        "preset": preset
    }
    filtered_options = {k: v for k, v in options.items() if v is not None}
    
    job = create_background_job("transcode", f"Transcoding all videos with custom options")
    background_tasks.add_task(bulk_transcode_background, job.job_id, filtered_options)
    return {"message": "Transcoding with options started in background", "job_id": job.job_id, "options": filtered_options}

def bulk_transcode_background(job_id: str, options: Optional[dict] = None):
    """Background task for bulk transcoding"""
    job = get_background_job(job_id)
    if not job:
        return
    
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename, original_path FROM media_files WHERE media_type='video' AND (has_transcoded_version=FALSE OR transcoded_path IS NULL)")
            videos = cursor.fetchall()
            
            if not videos:
                job.complete({"message": "No videos need transcoding.", "transcoded": 0, "failed": 0, "skipped": 0})
                return
            
            job.update_progress(0, len(videos), "Starting...")
            logger.info(f"Background bulk transcode for {len(videos)} videos, options: {options or 'default'}")
            
            count = 0
            failed = 0
            skipped = 0
            
            for i, video_row in enumerate(videos):
                orig_path = Path(video_row['original_path'])
                filename = video_row['filename']
                
                job.update_progress(i, len(videos), f"Processing {filename}")
                
                if not orig_path.exists():
                    logger.warning(f"Original for {filename} not found.")
                    failed += 1
                    continue
                
                out_path = TRANSCODED_DIR / f"{slugify_for_id(orig_path.stem)}.mp4"
                
                if out_path.exists():  # Already exists, ensure DB is up to date
                    db_path_str = str(out_path.relative_to(BASE_DIR))
                    try:
                        cursor.execute("UPDATE media_files SET transcoded_path=?, has_transcoded_version=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=? AND (has_transcoded_version=FALSE OR transcoded_path IS NULL)", (db_path_str, filename))
                        conn.commit() # Commit after update
                        if cursor.rowcount > 0:
                            logger.info(f"DB updated for existing transcode: {filename}")
                        skipped += 1
                        continue
                    except sqlite3.Error as e:
                        logger.error(f"DB update for existing transcode {filename} failed: {e}")
                        failed += 1
                        continue
            
                if transcode_video(orig_path, out_path, options=options):
                    db_path = str(out_path.relative_to(BASE_DIR)) if out_path.exists() else None
                    if db_path:
                        try:
                            cursor.execute("UPDATE media_files SET transcoded_path=?, has_transcoded_version=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_path, filename))
                            conn.commit() # Commit after update
                            count += 1
                        except sqlite3.Error as e:
                            logger.error(f"DB update for {filename} failed: {e}")
                            failed += 1
                    else:
                        logger.error(f"Transcode {filename} success, but output path issue or does not exist after creation.")
                        failed += 1
                else:
                    logger.error(f"Failed to transcode {filename}.")
                    failed += 1
            
            job.complete({"message": f"Transcoded: {count}, Failed: {failed}, Skipped: {skipped}", "transcoded": count, "failed": failed, "skipped": skipped})
            
    except Exception as e:
        logger.error(f"Transcoding background task failed: {e}")
        job.fail(str(e))

@app.post("/generate-preview/{video_name:path}")
async def generate_specific_preview_endpoint(video_name: str, response: Response, request: Request):
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT original_path FROM media_files WHERE filename=? AND media_type='video'", (video_name,))
        db_row = cursor.fetchone()
        if not db_row: raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found.")
        orig_path = Path(db_row['original_path'])
        if not orig_path.exists(): raise HTTPException(status_code=404, detail=f"Original for '{video_name}' not found.")
        
        preview_p = get_preview_path(orig_path)
        success = create_hover_preview(orig_path, preview_p)
        
        if success:
            db_path = str(preview_p.relative_to(BASE_DIR)) if preview_p.exists() else None
            url = f"/{db_path.replace('\\ ', '/')}" if db_path else None # Corrected path separator for URL
            if db_path:
                try:
                    cursor.execute("UPDATE media_files SET preview_path=?, has_preview=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_path, video_name))
                    response.headers["X-Preview-Url"] = url # Still useful for JS if needed
                    logger.info(f"Preview for {video_name} generated and DB updated.")
                except sqlite3.Error as e: 
                    logger.error(f"DB update for preview {video_name} failed: {e}")
                    # Re-render sidebar with error message
                    updated_video_details, _, _ = get_single_video_details_from_db(video_name)
                    if not updated_video_details: 
                        raise HTTPException(status_code=500, detail="Preview generated, DB update failed, and failed to reload video details.")
                    return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": updated_video_details, "error_message": "Preview generated but DB update failed."})
            else: # Should not happen if create_hover_preview was successful and returned a valid path
                logger.error(f"Preview for {video_name} created, but path issue for DB update.")
                # Re-render sidebar with error message
                updated_video_details, _, _ = get_single_video_details_from_db(video_name)
                if not updated_video_details: 
                    raise HTTPException(status_code=500, detail="Preview generated, path error, and failed to reload video details.")
                return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": updated_video_details, "error_message": "Preview generated but encountered a path issue before DB update."})
            
            # Success: Re-fetch details and re-render the sidebar
            updated_video_details, _, _ = get_single_video_details_from_db(video_name)
            if not updated_video_details: 
                 raise HTTPException(status_code=404, detail="Video details not found after preview generation.")
            return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": updated_video_details})
        else: 
            # If preview generation itself failed, re-render with an error message
            current_video_details, _, _ = get_single_video_details_from_db(video_name)
            if not current_video_details: 
                 raise HTTPException(status_code=500, detail="Failed to generate preview and also failed to reload video details.")
            return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": current_video_details, "error_message": "Failed to generate preview. Check logs."})

@app.post("/generate-all-previews")
async def generate_all_previews_endpoint(background_tasks: BackgroundTasks):
    """Start preview generation in background"""
    job = create_background_job("previews", "Generating hover previews")
    background_tasks.add_task(generate_all_previews_background, job.job_id)
    return {"message": "Preview generation started in background", "job_id": job.job_id}

def generate_all_previews_background(job_id: str):
    """Background task for generating all previews"""
    job = get_background_job(job_id)
    if not job:
        return
    
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename, original_path FROM media_files WHERE media_type='video' AND (has_preview=FALSE OR preview_path IS NULL)")
            videos = cursor.fetchall()
            
            if not videos:
                job.complete({"message": "No previews to generate.", "generated": 0, "failed": 0, "skipped": 0})
                return
            
            job.update_progress(0, len(videos), "Starting...")
            logger.info(f"Background bulk preview gen for {len(videos)} videos.")
            
            gen = 0
            failed = 0
            skipped = 0
            
            for i, video_row in enumerate(videos):
                filename = video_row['filename']
                orig_path = Path(video_row['original_path'])
                
                job.update_progress(i, len(videos), f"Processing {filename}")
                
                if not orig_path.exists():
                    logger.warning(f"Original for {filename} not found.")
                    failed += 1
                    continue
                
                preview_p = get_preview_path(orig_path)
                
                if preview_p.exists():  # Already exists, ensure DB up to date
                    db_path = str(preview_p.relative_to(BASE_DIR))
                    try:
                        cursor.execute("UPDATE media_files SET preview_path=?, has_preview=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=? AND (has_preview=FALSE OR preview_path IS NULL)", (db_path, filename))
                        conn.commit() # Commit after update
                        if cursor.rowcount > 0:
                            logger.info(f"DB updated for existing preview: {filename}")
                        skipped += 1
                        continue
                    except sqlite3.Error as e:
                        logger.error(f"DB update for existing preview {filename} failed: {e}")
                        failed += 1
                        continue
            
                if create_hover_preview(orig_path, preview_p):
                    db_path_str = str(preview_p.relative_to(BASE_DIR)) if preview_p.exists() else None
                    if db_path_str:
                        try:
                            cursor.execute("UPDATE media_files SET preview_path=?, has_preview=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_path_str, filename))
                            conn.commit() # Commit after update
                            gen += 1
                        except sqlite3.Error as e:
                            logger.error(f"DB update for {filename} preview failed: {e}")
                            failed += 1
                    else:
                        logger.error(f"Preview gen {filename} success, output path issue or does not exist after creation.")
                        failed += 1
                else:
                    logger.error(f"Failed to gen preview for {filename}.")
                    failed += 1
            
            job.complete({"message": f"Generated: {gen}, Failed: {failed}, Skipped: {skipped}", "generated": gen, "failed": failed, "skipped": skipped})
            
    except Exception as e:
        logger.error(f"Preview generation background task failed: {e}")
        job.fail(str(e))

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    from config import DEFAULT_MEDIA_SUBDIR # Local import to avoid clutter at top for one-off use
    return templates.TemplateResponse("settings.html", {"request": request, "CWD_DISPLAY_PATH": str(BASE_DIR).replace("\\", "/"), "DEFAULT_MEDIA_SUBDIR_JS": DEFAULT_MEDIA_SUBDIR})

@app.get("/settings/config", response_class=JSONResponse)
async def get_current_config():
    from database import get_setting # Local import
    media_dir_name = get_setting("media_directory_name")
    grid_size = get_setting("media_grid_size")
    per_page = get_setting("per_page")
    autoplay_enabled = get_setting("autoplay_enabled")
    default_muted = get_setting("default_muted")
    autoplay_next = get_setting("autoplay_next")
    auto_replay = get_setting("auto_replay")
    return {
        "media_directory_name": media_dir_name or "media",
        "media_grid_size": grid_size or "medium",
        "per_page": per_page or "20",
        "autoplay_enabled": autoplay_enabled or "true",
        "default_muted": default_muted or "true",
        "autoplay_next": autoplay_next or "false",
        "auto_replay": auto_replay or "false"
    }

@app.post("/settings/config")
async def update_app_config(request: Request):
    from database import update_setting, get_setting # Local import
    try:
        # Handle form data instead of JSON
        form_data = await request.form()
        new_media_dir_name = form_data.get("media_directory_name")
        new_grid_size = form_data.get("media_grid_size")
        new_per_page = form_data.get("per_page")
        new_autoplay_enabled = form_data.get("autoplay_enabled")
        new_default_muted = form_data.get("default_muted")
        new_autoplay_next = form_data.get("autoplay_next")
        new_auto_replay = form_data.get("auto_replay")
        
        requires_restart = False
        messages = []
        
        # Handle media directory name change
        if new_media_dir_name:
            if not isinstance(new_media_dir_name, str) or '/' in new_media_dir_name or '\\' in new_media_dir_name:
                return templates.TemplateResponse("_config_message.html", {
                    "request": request, 
                    "message": "Invalid media directory name. No slashes allowed.", 
                    "message_type": "error"
                })
            if get_setting("media_directory_name") != new_media_dir_name:
                update_setting("media_directory_name", new_media_dir_name)
                messages.append(f"Media directory changed to '{new_media_dir_name}'")
                requires_restart = True
        
        # Handle grid size change
        if new_grid_size:
            if new_grid_size not in ["small", "medium", "large"]:
                return templates.TemplateResponse("_config_message.html", {
                    "request": request, 
                    "message": "Invalid grid size.", 
                    "message_type": "error"
                })
            if get_setting("media_grid_size") != new_grid_size:
                update_setting("media_grid_size", new_grid_size)
                messages.append(f"Grid size changed to '{new_grid_size}'")
        
        # Handle per page change
        if new_per_page:
            if new_per_page not in ["10", "20", "30", "50", "100"]:
                return templates.TemplateResponse("_config_message.html", {
                    "request": request, 
                    "message": "Invalid per page value.", 
                    "message_type": "error"
                })
            if get_setting("per_page") != new_per_page:
                update_setting("per_page", new_per_page)
                messages.append(f"Per page changed to '{new_per_page}'")
        
        # Handle autoplay enabled change
        if new_autoplay_enabled is not None:
            if new_autoplay_enabled not in ["true", "false"]:
                return templates.TemplateResponse("_config_message.html", {
                    "request": request, 
                    "message": "Invalid autoplay setting.", 
                    "message_type": "error"
                })
            if get_setting("autoplay_enabled") != new_autoplay_enabled:
                update_setting("autoplay_enabled", new_autoplay_enabled)
                messages.append(f"Autoplay {'enabled' if new_autoplay_enabled == 'true' else 'disabled'}")
        
        # Handle default muted change
        if new_default_muted is not None:
            if new_default_muted not in ["true", "false"]:
                return templates.TemplateResponse("_config_message.html", {
                    "request": request, 
                    "message": "Invalid default mute setting.", 
                    "message_type": "error"
                })
            if get_setting("default_muted") != new_default_muted:
                update_setting("default_muted", new_default_muted)
                messages.append(f"Default mute {'enabled' if new_default_muted == 'true' else 'disabled'}")
        
        # Handle autoplay next change
        if new_autoplay_next is not None:
            if new_autoplay_next not in ["true", "false"]:
                return templates.TemplateResponse("_config_message.html", {
                    "request": request, 
                    "message": "Invalid autoplay next setting.", 
                    "message_type": "error"
                })
            if get_setting("autoplay_next") != new_autoplay_next:
                update_setting("autoplay_next", new_autoplay_next)
                messages.append(f"Autoplay next video {'enabled' if new_autoplay_next == 'true' else 'disabled'}")
        
        # Handle auto replay change
        if new_auto_replay is not None:
            if new_auto_replay not in ["true", "false"]:
                return templates.TemplateResponse("_config_message.html", {
                    "request": request, 
                    "message": "Invalid auto replay setting.", 
                    "message_type": "error"
                })
            if get_setting("auto_replay") != new_auto_replay:
                update_setting("auto_replay", new_auto_replay)
                messages.append(f"Auto replay {'enabled' if new_auto_replay == 'true' else 'disabled'}")
        
        if not messages:
            return templates.TemplateResponse("_config_message.html", {
                "request": request, 
                "message": "No changes were made to the configuration.", 
                "message_type": "info"
            })
        
        message = ". ".join(messages) + "."
        if requires_restart:
            message += " Please restart the application for media directory changes to take full effect."
        
        return templates.TemplateResponse("_config_message.html", {
            "request": request, 
            "message": message, 
            "message_type": "success"
        })
            
    except Exception as e: 
        logger.error(f"Error updating config: {e}")
        return templates.TemplateResponse("_config_message.html", {
            "request": request, 
            "message": f"Error saving configuration: {str(e)}", 
            "message_type": "error"
        })

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

@app.post("/settings/per-page")
async def update_per_page(per_page: str = Form(...)):
    from database import update_setting, get_setting # Local import
    if per_page not in ["10", "20", "30", "50", "100"]:
        raise HTTPException(status_code=400, detail="Invalid per page value.")
    
    current_per_page = get_setting("per_page")
    if current_per_page != per_page:
        update_setting("per_page", per_page)
        return {"message": f"Per page updated to {per_page}", "per_page": per_page}
    
    return {"message": "Per page unchanged", "per_page": per_page}

@app.post("/settings/autoplay-next")
async def update_autoplay_next(request: Request, autoplay_next: str = Form(...)):
    from database import update_setting, get_setting # Local import
    if autoplay_next not in ["true", "false"]:
        raise HTTPException(status_code=400, detail="Invalid autoplay next value.")
    
    current_autoplay_next = get_setting("autoplay_next")
    if current_autoplay_next != autoplay_next:
        update_setting("autoplay_next", autoplay_next)
        status = "enabled" if autoplay_next == "true" else "disabled"
        return templates.TemplateResponse("_autoplay_next_status.html", {
            "request": request, 
            "message": f"Autoplay next video {status}",
            "enabled": autoplay_next == "true"
        })
    
    return templates.TemplateResponse("_autoplay_next_status.html", {
        "request": request, 
        "message": "Autoplay next setting unchanged",
        "enabled": autoplay_next == "true"
    })

@app.post("/settings/auto-replay")
async def update_auto_replay(request: Request, auto_replay: str = Form(...)):
    from database import update_setting, get_setting # Local import
    if auto_replay not in ["true", "false"]:
        raise HTTPException(status_code=400, detail="Invalid auto replay value.")
    
    current_auto_replay = get_setting("auto_replay")
    if current_auto_replay != auto_replay:
        update_setting("auto_replay", auto_replay)
        status = "enabled" if auto_replay == "true" else "disabled"
        return templates.TemplateResponse("_auto_replay_status.html", {
            "request": request, 
            "message": f"Auto replay {status}",
            "enabled": auto_replay == "true"
        })
    
    return templates.TemplateResponse("_auto_replay_status.html", {
        "request": request, 
        "message": "Auto replay setting unchanged",
        "enabled": auto_replay == "true"
    })

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
            current_video_details, _, _ = get_single_video_details_from_db(video_exists['filename'])
            if not current_video_details: raise HTTPException(status_code=404, detail="Video details not found.")
            return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": current_video_details})

        set_clause = ", ".join([f"{key} = ?" for key in fields_to_update.keys()])
        params = list(fields_to_update.values()) + [video_id_db]
        try:
            cursor.execute(f"UPDATE media_files SET {set_clause}, last_scanned=CURRENT_TIMESTAMP WHERE id = ?", params)
            logger.info(f"Updated metadata for video ID {video_id_db}: {fields_to_update}")
        except sqlite3.Error as e: logger.error(f"DB error for video ID {video_id_db}: {e}"); raise HTTPException(status_code=500, detail="DB error.")
    
    updated_video_details, _, _ = get_single_video_details_from_db(video_exists['filename'])
    if not updated_video_details: raise HTTPException(status_code=404, detail="Video details not found post-update.")
    return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": updated_video_details})

@app.delete("/video/{video_id_db}/tag/{tag_name}")
async def remove_video_tag(video_id_db: int, tag_name: str, request: Request):
    """Remove a specific tag from a video"""
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename, tags FROM media_files WHERE id = ? AND media_type = 'video'", (video_id_db,))
        video_exists = cursor.fetchone()
        if not video_exists: 
            raise HTTPException(status_code=404, detail="Video not found.")

        # Parse current tags
        current_tags_json = video_exists['tags'] or '[]'
        try:
            current_tags = json.loads(current_tags_json)
        except json.JSONDecodeError:
            current_tags = []

        # Remove the specified tag (case-insensitive)
        updated_tags = [tag for tag in current_tags if tag.lower() != tag_name.lower()]
        
        # If no change was made, just return current state
        if len(updated_tags) == len(current_tags):
            current_video_details, _, _ = get_single_video_details_from_db(video_exists['filename'])
            if not current_video_details: 
                raise HTTPException(status_code=404, detail="Video details not found.")
            return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": current_video_details})

        # Update database with new tags
        try:
            cursor.execute("UPDATE media_files SET tags = ?, last_scanned=CURRENT_TIMESTAMP WHERE id = ?", 
                         (json.dumps(updated_tags), video_id_db))
            logger.info(f"Removed tag '{tag_name}' from video ID {video_id_db}")
        except sqlite3.Error as e: 
            logger.error(f"DB error removing tag from video ID {video_id_db}: {e}")
            raise HTTPException(status_code=500, detail="Database error.")
    
    # Return updated sidebar
    updated_video_details, _, _ = get_single_video_details_from_db(video_exists['filename'])
    if not updated_video_details: 
        raise HTTPException(status_code=404, detail="Video details not found post-update.")
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

# Startup scan - REMOVED to rely on DB as source of truth initially
# if MEDIA_DIR.exists() and MEDIA_DIR.is_dir():
#     scan_media_directory_and_update_db()
# elif not MEDIA_DIR.exists():
#     logger.warning(f"Configured media directory {MEDIA_DIR} does not exist. Create or configure in Settings.")

# --- Delete Endpoints ---

@app.post("/delete-thumbnail/{video_name:path}")
async def delete_specific_thumbnail_endpoint(video_name: str, request: Request):
    """Deletes the specific custom thumbnail for a video."""
    error_message = None
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, thumbnail_path FROM media_files WHERE filename = ? AND media_type = 'video'", (video_name,))
        db_row = cursor.fetchone()
        if not db_row:
            raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found.")

        video_id = db_row['id']
        thumbnail_path_str = db_row['thumbnail_path']

        if not thumbnail_path_str:
            logger.info(f"No custom thumbnail found for {video_name} to delete.")
        else:
            thumbnail_path = BASE_DIR / thumbnail_path_str
            if thumbnail_path.exists():
                try:
                    os.remove(thumbnail_path)
                    logger.info(f"Deleted thumbnail file: {thumbnail_path}")
                except OSError as e:
                    logger.error(f"Failed to delete thumbnail file {thumbnail_path}: {e}")
                    error_message = f"Failed to delete thumbnail file: {e}"
            try:
                cursor.execute("UPDATE media_files SET thumbnail_path = NULL, has_specific_thumbnail = FALSE, last_scanned=CURRENT_TIMESTAMP WHERE id = ?", (video_id,))
                logger.info(f"Database updated for thumbnail deletion of {video_name}.")
            except sqlite3.Error as e:
                logger.error(f"Failed to update database for thumbnail deletion of {video_name}: {e}")
                error_message = (error_message + " | " if error_message else "") + "Thumbnail file deleted/checked but database update failed."

    updated_video_details, _, _ = get_single_video_details_from_db(video_name)
    if not updated_video_details:
        raise HTTPException(status_code=404, detail="Video details not found after attempting thumbnail deletion.")
    
    context = {"request": request, "video": updated_video_details}
    if error_message: 
        context["error_message"] = error_message # Pass error to template if any occurred
    return templates.TemplateResponse("_video_metadata_sidebar.html", context)

@app.post("/delete-transcoded/{video_name:path}")
async def delete_specific_transcoded_version_endpoint(video_name: str, request: Request):
    """Deletes the specific transcoded version for a video."""
    error_message = None
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, transcoded_path FROM media_files WHERE filename = ? AND media_type = 'video'", (video_name,))
        db_row = cursor.fetchone()
        if not db_row:
            raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found.")

        video_id = db_row['id']
        transcoded_path_str = db_row['transcoded_path']

        if not transcoded_path_str:
            logger.info(f"No web-optimized version found for {video_name} to delete.")
        else:
            transcoded_path = BASE_DIR / transcoded_path_str
            if transcoded_path.exists():
                try:
                    os.remove(transcoded_path)
                    logger.info(f"Deleted transcoded file: {transcoded_path}")
                except OSError as e:
                    logger.error(f"Failed to delete transcoded file {transcoded_path}: {e}")
                    error_message = f"Failed to delete transcoded file: {e}"
            try:
                cursor.execute("UPDATE media_files SET transcoded_path = NULL, has_transcoded_version = FALSE, last_scanned=CURRENT_TIMESTAMP WHERE id = ?", (video_id,))
                logger.info(f"Database updated for transcoded version deletion of {video_name}.")
            except sqlite3.Error as e:
                logger.error(f"Failed to update database for transcoded version deletion of {video_name}: {e}")
                error_message = (error_message + " | " if error_message else "") + "Transcoded file deleted/checked but database update failed."

    updated_video_details, _, _ = get_single_video_details_from_db(video_name)
    if not updated_video_details:
        raise HTTPException(status_code=404, detail="Video details not found after attempting transcoded file deletion.")
    
    context = {"request": request, "video": updated_video_details}
    if error_message: 
        context["error_message"] = error_message
    return templates.TemplateResponse("_video_metadata_sidebar.html", context)

@app.post("/delete-preview/{video_name:path}")
async def delete_specific_preview_endpoint(video_name: str, request: Request):
    """Deletes the specific hover preview for a video."""
    error_message = None
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, preview_path FROM media_files WHERE filename = ? AND media_type = 'video'", (video_name,))
        db_row = cursor.fetchone()
        if not db_row:
            raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found.")

        video_id = db_row['id']
        preview_path_str = db_row['preview_path']

        if not preview_path_str:
            logger.info(f"No hover preview found for {video_name} to delete.")
        else:
            preview_path = BASE_DIR / preview_path_str
            if preview_path.exists():
                try:
                    os.remove(preview_path)
                    logger.info(f"Deleted preview file: {preview_path}")
                except OSError as e:
                    logger.error(f"Failed to delete preview file {preview_path}: {e}")
                    error_message = f"Failed to delete preview file: {e}"
            try:
                cursor.execute("UPDATE media_files SET preview_path = NULL, has_preview = FALSE, last_scanned=CURRENT_TIMESTAMP WHERE id = ?", (video_id,))
                logger.info(f"Database updated for preview deletion of {video_name}.")
            except sqlite3.Error as e:
                logger.error(f"Failed to update database for preview deletion of {video_name}: {e}")
                error_message = (error_message + " | " if error_message else "") + "Preview file deleted/checked but database update failed."

    updated_video_details, _, _ = get_single_video_details_from_db(video_name)
    if not updated_video_details:
        raise HTTPException(status_code=404, detail="Video details not found after attempting preview file deletion.")

    context = {"request": request, "video": updated_video_details}
    if error_message: 
        context["error_message"] = error_message
    return templates.TemplateResponse("_video_metadata_sidebar.html", context)

@app.delete("/delete-media-file/{video_id_db}") # Using DELETE method
async def delete_media_file_endpoint(video_id_db: int, response: Response):
    """Deletes a media file and all its associated assets."""
    with db_connection() as conn:
        cursor = conn.cursor()
        # Get paths for all associated files
        cursor.execute("SELECT original_path, thumbnail_path, transcoded_path, preview_path FROM media_files WHERE id = ?", (video_id_db,))
        db_row = cursor.fetchone()
        if not db_row:
            raise HTTPException(status_code=404, detail=f"Media file with ID {video_id_db} not found.")

        original_path_str = db_row['original_path']
        thumbnail_path_str = db_row['thumbnail_path']
        transcoded_path_str = db_row['transcoded_path']
        preview_path_str = db_row['preview_path']

        files_to_delete = []
        if original_path_str: files_to_delete.append(BASE_DIR / original_path_str)
        if thumbnail_path_str: files_to_delete.append(BASE_DIR / thumbnail_path_str)
        if transcoded_path_str: files_to_delete.append(BASE_DIR / transcoded_path_str)
        if preview_path_str: files_to_delete.append(BASE_DIR / preview_path_str)

        # Attempt to delete files
        failed_files = []
        for file_path in files_to_delete:
            if file_path.exists():
                try:
                    # Use unlink for Path objects, which is generally preferred over os.remove
                    file_path.unlink(missing_ok=True)
                    logger.info(f"Deleted file: {file_path}")
                except OSError as e:
                    logger.error(f"Failed to delete file {file_path}: {e}")
                    failed_files.append(str(file_path))

        # Even if some files failed to delete, proceed with DB deletion if original was found
        if original_path_str:
            try:
                cursor.execute("DELETE FROM media_files WHERE id = ?", (video_id_db,))
                logger.info(f"Deleted media file record from DB for ID {video_id_db}.")
                conn.commit() # Commit the transaction
            except sqlite3.Error as e:
                logger.error(f"Failed to delete media file record from DB for ID {video_id_db}: {e}")
                # Depending on policy, might want to roll back file deletions here
                raise HTTPException(status_code=500, detail="Failed to delete media file record from database.")
        else:
            # This case should ideally not happen if original_path_str is None but ID was found
             raise HTTPException(status_code=500, detail="Original file path not found in database record.")

        if failed_files:
            # Return a 200 OK response but include a warning about failed file deletions
            response.status_code = 200 # Still consider it a success if DB record is gone
            return {"message": "Media file record deleted, but some associated files could not be deleted.", "failed_files": failed_files}
        else:
            # Set HX-Redirect header for HTMX to navigate to the home page
            response.headers["HX-Redirect"] = "/"
            return {"message": f"Media file and all associated assets deleted for ID {video_id_db}."}

@app.post("/tools/scan-media-directory", response_class=HTMLResponse)
async def scan_media_directory_endpoint(request: Request, background_tasks: BackgroundTasks):
    """Endpoint to trigger a media directory scan and database update."""
    job = create_background_job("media_scan", "Scanning media directory and updating database")
    background_tasks.add_task(run_scan_media_directory_background, job.job_id)
    
    # HTMX can swap this into a target div to show the message and job_id
    return templates.TemplateResponse("_scan_status_message.html", {
        "request": request,
        "message": "Media directory scan started in background.",
        "job_id": job.job_id,
        "job_status_url": f"/jobs/{job.job_id}" # URL to poll for job status
    })

def run_scan_media_directory_background(job_id: str):
    """Background task wrapper for scanning media directory."""
    job = get_background_job(job_id)
    if not job:
        logger.error(f"Job {job_id} not found for media scan.")
        return

    try:
        job.update_progress(0, 1, "Starting media scan...") # total=1 as it's one major task
        # scan_media_directory_and_update_db() # Original function, might need modification for job progress
        
        # For now, let's assume scan_media_directory_and_update_db doesn't have fine-grained progress
        # We'll update the job based on its completion or failure.
        # The original scan_media_directory_and_update_db already logs extensively.
        scan_media_directory_and_update_db() # This is a blocking call

        job.update_progress(1, 1, "Scan completed.")
        job.complete({"message": "Media directory scan and database update finished successfully."})
        logger.info(f"Background media scan job {job_id} completed successfully.")
    except Exception as e:
        logger.error(f"Media scan background task (job {job_id}) failed: {e}")
        job.fail(str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting PABT server. Media directory: {MEDIA_DIR}. Access at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 