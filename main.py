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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Directory Setup & Configuration --- 
BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.json"

DEFAULT_MEDIA_SUBDIR = "media" # Default if not in config

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration: {config}")
                return config
        except json.JSONDecodeError:
            logger.error(f"Error decoding {CONFIG_FILE}. Using default configuration.")
        except Exception as e:
            logger.error(f"Error loading {CONFIG_FILE}: {e}. Using default configuration.")
    return {"media_directory_name": DEFAULT_MEDIA_SUBDIR} # Default structure

def save_config(config: dict):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saved configuration: {config}")
    except Exception as e:
        logger.error(f"Error saving {CONFIG_FILE}: {e}")

# Load configuration and set MEDIA_DIR
app_config = load_config()
# Media directory is now relative to BASE_DIR, based on config or default
MEDIA_DIR_NAME = app_config.get("media_directory_name", DEFAULT_MEDIA_SUBDIR)
MEDIA_DIR = BASE_DIR / MEDIA_DIR_NAME

TRANSCODED_DIR = BASE_DIR / "media_transcoded" # For transcoded videos
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

def get_media_files(search_query: str = None):
    media_files = []
    for file_path_obj in MEDIA_DIR.iterdir():
        if file_path_obj.is_file():
            suffix = file_path_obj.suffix.lower()
            supported_video = ['.mp4', '.avi', '.mov', '.mkv'] # Original supported videos
            supported_audio = ['.mp3', '.wav', '.ogg']
            supported_image = ['.jpg', '.jpeg', '.png', '.gif']

            if not (suffix in supported_video + supported_audio + supported_image):
                continue
            if search_query and search_query.lower() not in file_path_obj.name.lower():
                continue
            
            media_type = 'video' if suffix in supported_video else \
                       'audio' if suffix in supported_audio else 'image'

            display_thumbnail = get_display_thumbnail_url(file_path_obj, media_type)
            duration = get_media_duration(file_path_obj) if media_type in ['video', 'audio'] else None
            
            # Use slugified stem for default transcoded file checks and path construction
            slugified_stem = slugify_for_id(file_path_obj.stem)
            transcoded_file_path_default = TRANSCODED_DIR / f"{slugified_stem}.mp4"
            has_transcoded_version = transcoded_file_path_default.exists() and media_type == 'video'
            
            preview_file_path = get_preview_path(file_path_obj) # Already uses slugification internally
            has_preview = preview_file_path.exists() and media_type == 'video'
            preview_url = f"/static/previews/{preview_file_path.name}" if has_preview else None

            playable_path = f"/media_content/{file_path_obj.name}" # Default to original
            if has_transcoded_version:
                # If default transcoded version exists, it becomes the primary playable version
                playable_path = f"/media_content_transcoded/{transcoded_file_path_default.name}"

            media_files.append({
                "name": file_path_obj.name, # Keep original name for display and non-transcoded operations
                "type": media_type,
                "slugified_stem": slugified_stem, # Add slugified stem for convenience if needed elsewhere
                "size": file_path_obj.stat().st_size,
                "thumbnail": display_thumbnail,
                "has_specific_thumbnail": get_thumbnail_path(file_path_obj).exists(), # get_thumbnail_path now uses slugify
                "duration": duration,
                "path": playable_path,
                "original_path_for_download": f"/media_content/{file_path_obj.name}",
                "has_transcoded_version": has_transcoded_version, # This now refers to the default slugified transcode
                "has_preview": has_preview,
                "preview_url": preview_url
            })
    return media_files

def get_single_video_details(video_name: str):
    original_video_path = MEDIA_DIR / video_name
    if not original_video_path.exists() or not original_video_path.is_file():
        return None

    slugified_stem = slugify_for_id(Path(video_name).stem)
    transcoded_video_path_default = TRANSCODED_DIR / f"{slugified_stem}.mp4"
    has_transcoded_version = transcoded_video_path_default.exists()

    preview_file_path = get_preview_path(original_video_path) # Already uses slugification
    has_preview = preview_file_path.exists()
    preview_url = f"/static/previews/{preview_file_path.name}" if has_preview else None

    playable_path = f"/media_content/{video_name}" # Default to original
    if has_transcoded_version:
        playable_path = f"/media_content_transcoded/{transcoded_video_path_default.name}"

    try:
        clip = VideoFileClip(str(original_video_path.resolve()))
        details = {
            "name": video_name,
            "type": "video",
            "slugified_stem": slugified_stem,
            "path": playable_path,
            "original_path_for_download": f"/media_content/{video_name}",
            "duration": str(timedelta(seconds=int(clip.duration))),
            "resolution": f"{clip.size[0]}x{clip.size[1]}",
            "size": original_video_path.stat().st_size,
            "fps": clip.fps,
            "thumbnail": get_display_thumbnail_url(original_video_path, 'video'), # get_display_thumbnail_url uses get_thumbnail_path, which slugs
            "has_specific_thumbnail": get_thumbnail_path(original_video_path).exists(),
            "has_transcoded_version": has_transcoded_version, # This now refers to the default slugified transcode
            "has_preview": has_preview,
            "preview_url": preview_url
        }
        clip.close()
        return details
    except Exception as e:
        logger.error(f"Error getting details for video {video_name}: {e}")
        return None

# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/files", response_class=HTMLResponse)
async def list_files(request: Request, search: str = None):
    files = get_media_files(search)
    return templates.TemplateResponse("file_list.html", {"request": request, "files": files})

@app.get("/media_content/{file_name:path}") # Added :path to handle potential subdirs if media grows
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
    video_details = get_single_video_details(video_name)
    if not video_details:
        raise HTTPException(status_code=404, detail="Video not found")
    return templates.TemplateResponse("video_player.html", {"request": request, "video": video_details})

@app.post("/generate-thumbnail/{video_name:path}")
async def generate_specific_thumbnail_endpoint(video_name: str, response: Response):
    video_path = MEDIA_DIR / video_name
    if not video_path.exists() or not video_path.is_file() or video_path.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
        raise HTTPException(status_code=404, detail="Video not found or not a supported video type")
    thumbnail_url = _actually_create_thumbnail(video_path, force_creation=True)
    if thumbnail_url:
        response.headers["X-Thumbnail-Url"] = thumbnail_url
        return {"message": f"Thumbnail generated for {video_name}", "thumbnail_url": thumbnail_url}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to generate thumbnail for {video_name}")

@app.post("/generate-all-video-thumbnails")
async def generate_all_thumbnails_endpoint():
    generated_count = 0; failed_count = 0
    video_files = [f for f in MEDIA_DIR.iterdir() if f.is_file() and f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]
    for video_file in video_files:
        if not get_thumbnail_path(video_file).exists(): 
            if _actually_create_thumbnail(video_file, force_creation=True): generated_count += 1
            else: failed_count += 1
    return {"message": f"Thumbnail generation completed. Generated: {generated_count}, Failed: {failed_count}.", "generated": generated_count, "failed": failed_count}

@app.get("/tools/thumbnails", response_class=HTMLResponse)
async def thumbnail_tools_page(request: Request):
    # This page will now be more comprehensive
    return templates.TemplateResponse("tools_page.html", {"request": request})

# --- NEW TRANSCODING ENDPOINTS ---
@app.post("/transcode-video/{video_name:path}")
async def transcode_specific_video_endpoint(video_name: str):
    original_path = MEDIA_DIR / video_name
    slugified_stem = slugify_for_id(original_path.stem)
    transcoded_path = TRANSCODED_DIR / f"{slugified_stem}.mp4"

    if not original_path.exists() or not original_path.is_file():
        raise HTTPException(status_code=404, detail="Original video not found.")
    if original_path.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
        raise HTTPException(status_code=400, detail="File is not a supported video type for transcoding.")

    logger.info(f"Attempting to transcode: {original_path} to {transcoded_path}")
    success = transcode_video(original_path, transcoded_path)
    if success:
        return {"message": f"Successfully started transcoding for {video_name}. Output will be at {transcoded_path.name}"}
    else:
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="FFmpeg not found. Please install FFmpeg and ensure it is in your system PATH.")
        except subprocess.CalledProcessError:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to transcode {video_name}. Check server logs for FFmpeg errors.")

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
    original_path = MEDIA_DIR / video_name
    slugified_stem = slugify_for_id(original_path.stem)
    
    if not original_path.exists() or not original_path.is_file():
        raise HTTPException(status_code=404, detail="Original video not found.")
    if original_path.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
        raise HTTPException(status_code=400, detail="File is not a supported video type for transcoding.")

    options = {
        "resolution": resolution,
        "quality_mode": quality_mode,
        "crf": int(crf) if quality_mode == "crf" else None,
        "video_bitrate": video_bitrate if quality_mode == "bitrate" else None,
        "audio_bitrate": audio_bitrate,
        "preset": preset,
        "profile": profile
    }
    
    quality_suffix = f"_{resolution}_{quality_mode}{crf if quality_mode == 'crf' else video_bitrate}"
    transcoded_path = TRANSCODED_DIR / f"{slugified_stem}{quality_suffix}.mp4"
    
    logger.info(f"Attempting advanced transcode: {original_path} to {transcoded_path} with options: {options}")
    
    success = transcode_video(original_path, transcoded_path, options)
    if success:
        new_file_web_path = f"/media_content_transcoded/{transcoded_path.name}"
        # Return JSON that HTMX can use, or set a custom header
        # For HTMX swapping specific parts, a JSON response handled by client-side JS is often cleaner
        # For this direct update, we can send back a partial HTML or set headers.
        # Let's try returning an HTML response that also triggers a custom event with details.
        
        response_content = f"<div id=\"advanced-transcoding-status-message\" class=\"text-green-400 text-xs mt-2\">Advanced transcoding complete! Output: {transcoded_path.name}. Player will update.</div>"
        response = HTMLResponse(content=response_content)
        response.headers["HX-Trigger" ] = f"newTranscodeAvailable={{ \"newPath\": \"{new_file_web_path}\", \"fileName\": \"{transcoded_path.name}\" }}"
        return response
    else:
        # ... (error handling as before)
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="FFmpeg not found. Please install FFmpeg and ensure it is in your system PATH.")
        except subprocess.CalledProcessError:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to transcode {video_name}. Check server logs for FFmpeg errors.")

@app.post("/transcode-all-videos")
async def transcode_all_videos_endpoint():
    success_count, error_count, errors = await transcode_all_videos()
    status_msg = f"Transcoding complete! {success_count} videos transcoded successfully"
    if error_count > 0:
        status_msg += f", {error_count} failed (see logs)"
    return HTMLResponse(content=status_msg)

@app.post("/transcode-all-videos-with-options")
async def transcode_all_videos_with_options_endpoint(
    resolution: str = Form("720p"),
    quality_mode: str = Form("crf"),
    crf: str = Form("23"),
    video_bitrate: str = Form("2M"),
    audio_bitrate: str = Form("128k"),
    preset: str = Form("medium")
):
    # Convert form parameters to options dictionary
    options = {
        "resolution": resolution,
        "quality_mode": quality_mode,
        "crf": int(crf) if quality_mode == "crf" else None,
        "video_bitrate": video_bitrate if quality_mode == "bitrate" else None,
        "audio_bitrate": audio_bitrate,
        "preset": preset
    }
    
    success_count, error_count, errors = await transcode_all_videos(options)
    status_msg = f"Custom transcoding complete! {success_count} videos transcoded successfully"
    if error_count > 0:
        status_msg += f", {error_count} failed (see logs)"
    return HTMLResponse(content=status_msg)

# Bulk transcoding helper function
async def transcode_all_videos(options=None):
    media_files = get_media_files()
    video_files = [f for f in media_files if f["type"] == "video"]
    
    success_count = 0
    error_count = 0
    errors = []
    
    for video_info in video_files:
        original_path = MEDIA_DIR / video_info["name"]
        slugified_stem = slugify_for_id(original_path.stem)

        # Determine output path:
        # If bulk transcoding with options, it applies those options to the "default" slugified stem.
        # If basic bulk transcoding, it uses default encoding options on the "default" slugified stem.
        output_path_for_bulk = TRANSCODED_DIR / f"{slugified_stem}.mp4"

        # Check if this specific version (default slugified) already exists
        # The video_info['has_transcoded_version'] from get_media_files (once updated) will reflect this.
        # However, since get_media_files is called once at the start, its has_transcoded_version
        # might be stale if a previous iteration of this loop just created the file.
        # So, we check output_path_for_bulk.exists() directly.
        if output_path_for_bulk.exists() and video_info['has_transcoded_version']: # video_info['has_transcoded_version'] for initial skip based on get_media_files
             logger.info(f"Skipping already transcoded (default/bulk): {output_path_for_bulk}")
             # success_count +=1 # Or a skipped_count
             continue
        if output_path_for_bulk.exists() and not options: # If basic bulk and file exists, definitely skip
            logger.info(f"Skipping already transcoded (basic bulk): {output_path_for_bulk}")
            continue
        # If it's bulk *with options*, we might want to re-transcode even if a STEM.mp4 exists,
        # if those options are different. For simplicity now, if STEM.mp4 exists, bulk-with-options
        # will overwrite it with the new options.

        logger.info(f"Queueing for bulk transcode: {original_path} to {output_path_for_bulk} with options: {options if options else 'default'}")
        
        success = transcode_video(original_path, output_path_for_bulk, options) # options are encoding settings
        if success:
            success_count += 1
        else:
            error_count += 1
            errors.append(f"{video_info['name']}: transcoding failed")
            logger.error(f"Failed to transcode {original_path} for bulk operation.")
    
    return success_count, error_count, errors

# --- NEW HOVER PREVIEW ENDPOINTS ---
@app.post("/generate-preview/{video_name:path}")
async def generate_specific_preview_endpoint(video_name: str, response: Response):
    original_path = MEDIA_DIR / video_name
    preview_path = get_preview_path(original_path)

    if not original_path.exists() or not original_path.is_file():
        raise HTTPException(status_code=404, detail="Original video not found.")
    if original_path.suffix.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
        raise HTTPException(status_code=400, detail="File is not a supported video type for preview generation.")

    logger.info(f"Attempting to generate hover preview for: {original_path} to {preview_path}")
    success = create_hover_preview(original_path, preview_path)
    
    if success:
        preview_url = f"/static/previews/{preview_path.name}"
        response.headers["X-Preview-Url"] = preview_url
        return {"message": f"Successfully generated hover preview for {video_name}. Output: {preview_path.name}", "preview_url": preview_url}
    else:
        # FFmpeg error check similar to transcode endpoint
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="FFmpeg not found. Please install FFmpeg and ensure it is in your system PATH.")
        except subprocess.CalledProcessError:
            pass 
        raise HTTPException(status_code=500, detail=f"Failed to generate hover preview for {video_name}. Check server logs.")

@app.post("/generate-all-previews")
async def generate_all_previews_endpoint():
    generated_count = 0
    failed_count = 0
    skipped_count = 0
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    all_video_files = [f for f in MEDIA_DIR.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]

    if not all_video_files:
        return {"message": "No video files found to generate previews for.", "generated": 0, "failed": 0, "skipped": 0}

    for original_path in all_video_files:
        preview_path = get_preview_path(original_path)
        if preview_path.exists():
            skipped_count += 1
            continue
        
        logger.info(f"Queueing hover preview generation for: {original_path} to {preview_path}")
        if create_hover_preview(original_path, preview_path):
            generated_count += 1
        else:
            failed_count += 1
            
    if failed_count == len(all_video_files) and failed_count > 0:
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        except FileNotFoundError:
            return {"message": "FFmpeg not found. All preview generations failed. Please install FFmpeg.", "generated": generated_count, "failed": failed_count, "skipped": skipped_count}
        except subprocess.CalledProcessError:
            pass

    return {
        "message": f"Hover preview generation completed. Generated: {generated_count}, Failed: {failed_count}, Skipped: {skipped_count}.",
        "generated": generated_count,
        "failed": failed_count,
        "skipped": skipped_count
    } 

# --- Configuration Endpoints ---
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
    # Return only the part of the config the user can change, or the whole thing if simple
    return {"media_directory_name": app_config.get("media_directory_name", DEFAULT_MEDIA_SUBDIR)}

@app.post("/settings/config")
async def update_app_config(request: Request):
    form_data = await request.form()
    new_media_dir_name = form_data.get("media_directory_name")

    if not new_media_dir_name or not isinstance(new_media_dir_name, str):
        raise HTTPException(status_code=400, detail="Invalid media_directory_name provided.")
    
    # Basic validation: check if it's just a name and not an absolute path or trying path traversal
    # This assumes the directory is a direct child of BASE_DIR for simplicity.
    # More complex validation would be needed for arbitrary paths.
    if Path(new_media_dir_name).is_absolute() or ".." in new_media_dir_name or "/" in new_media_dir_name or "\\" in new_media_dir_name:
        logger.warning(f"Attempt to set potentially unsafe media directory: {new_media_dir_name}")
        raise HTTPException(status_code=400, detail="Media directory name must be a simple name (no paths or slashes).")

    current_config = load_config()
    current_config["media_directory_name"] = new_media_dir_name.strip()
    save_config(current_config)
    
    # IMPORTANT: This does NOT change MEDIA_DIR for the currently running instance.
    # The application must be restarted for the change to take full effect.
    return HTMLResponse(content=f"""
        <div id=\"config-status-message\" class=\"p-3 mb-4 text-sm rounded-lg bg-sky-500/20 text-sky-300\">
            Configuration saved: Media directory set to '<strong>{new_media_dir_name}</strong>'.<br>
            <strong>Please restart the application for this change to take full effect.</strong>
        </div>
    """) 