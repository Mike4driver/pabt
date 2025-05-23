from pathlib import Path
from PIL import Image
from moviepy.editor import VideoFileClip
from mutagen import File
from datetime import timedelta
import subprocess
import os

from config import (
    logger, slugify_for_id, 
    THUMBNAILS_DIR, PREVIEWS_DIR,
    SUPPORTED_VIDEO_EXTENSIONS, SUPPORTED_AUDIO_EXTENSIONS, SUPPORTED_IMAGE_EXTENSIONS
)

# --- FFmpeg Helper Functions ---
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
    
    default_options = {
        'resolution': 'original',
        'quality_mode': 'crf',
        'crf': 23,
        'video_bitrate': '2M',
        'audio_bitrate': '128k',
        'preset': 'medium',
        'profile': 'high'
    }
    
    if options:
        default_options.update(options)
    opts = default_options
    
    command = ["ffmpeg", "-i", str(input_path.resolve())]
    command.extend(["-c:v", "libx264", "-preset", opts['preset'], "-profile:v", opts['profile']])
    if opts['quality_mode'] == 'crf':
        command.extend(["-crf", str(opts['crf'])])
    else:
        command.extend(["-b:v", opts['video_bitrate']])
    if opts['resolution'] != 'original':
        if opts['resolution'] == '1080p': command.extend(["-vf", "scale=-2:1080"])
        elif opts['resolution'] == '720p': command.extend(["-vf", "scale=-2:720"])
        elif opts['resolution'] == '480p': command.extend(["-vf", "scale=-2:480"])
    command.extend(["-c:a", "aac", "-b:a", opts['audio_bitrate']])
    command.extend(["-movflags", "+faststart"])
    command.append(str(output_path.resolve()))
    
    return run_ffmpeg_command(command)

def create_hover_preview(input_path: Path, output_path: Path):
    if output_path.exists():
        logger.info(f"Hover preview already exists, skipping: {output_path}")
        return True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-i", str(input_path.resolve()),
        "-ss", "2",
        "-t", "5",
        "-vf", "scale=320:-2,crop=iw:min(ih\\,ih/9*16)",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-crf", "28",
        "-an",
        "-movflags", "+faststart",
        str(output_path.resolve())
    ]
    return run_ffmpeg_command(command)

# --- Thumbnail and Duration Logic ---
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
                with Image.open(file_path.resolve()) as img:
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

def get_preview_path(file_path: Path):
    slugified_stem = slugify_for_id(file_path.stem)
    return PREVIEWS_DIR / f"{slugified_stem}_preview.mp4" 