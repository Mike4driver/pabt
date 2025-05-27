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
from typing import Optional, List

# --- Early Database Initialization ---
import database
database.create_tables()

# --- Subsequent Imports ---
from database import get_db_connection, get_setting, update_setting, db_connection
from config import (
    logger, templates, mount_static_files, 
    BASE_DIR, MEDIA_DIR, TRANSCODED_DIR, PREVIEWS_DIR, THUMBNAILS_DIR,
    SUPPORTED_VIDEO_EXTENSIONS, SUPPORTED_AUDIO_EXTENSIONS, SUPPORTED_IMAGE_EXTENSIONS,
    # slugify_for_id, safe_path_join, get_media_type_from_extension # These were moved to utils.py
)
from media_processing import (
    run_ffmpeg_command, transcode_video, create_hover_preview, 
    get_thumbnail_path, get_media_duration, _actually_create_thumbnail,
    get_display_thumbnail_url, get_preview_path
)
from data_access import (
    # scan_media_directory_and_update_db, # This will be in a tools/admin route
    get_media_files_from_db, get_single_video_details_from_db, get_all_tags_from_db, # scan_media_directory_and_update_db is still here
    get_previous_video_in_queue, scan_media_directory_and_update_db
)
from jobs_manager import BackgroundJob, create_background_job, get_background_job, cleanup_old_jobs, background_jobs, job_lock

# Import utilities
from utils import slugify_for_id, safe_path_join, get_media_type_from_extension

# Import routers
from routes import media_routes 
from routes import job_routes # Added job_routes
from routes import processing_routes # Added processing_routes import
from routes import settings_routes # Added settings_routes import
from routes import ml_routes # Added ml_routes import

app = FastAPI()

# --- Mount Static Files --- 
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# --- Include Routers ---
app.include_router(media_routes.router) # Include media routes
app.include_router(job_routes.router, prefix="/jobs", tags=["jobs"]) # Added job_routes
app.include_router(processing_routes.router, prefix="/processing", tags=["processing"]) # Added processing_routes
app.include_router(settings_routes.router, prefix="/settings", tags=["settings"]) # Added settings_routes
app.include_router(ml_routes.router, prefix="/ml", tags=["ml"]) # Added ml_routes


# --- Jinja2 Filters & Globals ---
if hasattr(templates, 'env'): 
    templates.env.filters['slugify_for_id'] = slugify_for_id
    templates.env.globals['get_previous_video_in_queue'] = get_previous_video_in_queue
    templates.env.globals['format_media_duration'] = lambda seconds: str(timedelta(seconds=int(seconds))) if seconds is not None else None

# --- ROUTE DEFINITIONS (MOVED) --- #

# --- Tool Page Route (can be moved later if desired) ---
@app.get("/tools/media-processing", response_class=HTMLResponse, name="tools_page")
async def tools_page(request: Request):
    return templates.TemplateResponse("tools_page.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting PABT server. Media directory from config: {get_setting('media_directory_name')}. Access at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 