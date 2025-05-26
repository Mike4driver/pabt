from fastapi import APIRouter, Request, Query, HTTPException, Response, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pathlib import Path
from typing import Optional
import sqlite3 # Added for DB operations within routes temporarily

from config import templates, MEDIA_DIR, TRANSCODED_DIR, BASE_DIR, logger, get_setting # Assuming these are accessible
from data_access import get_media_files_from_db, get_single_video_details_from_db, get_all_tags_from_db
from utils import safe_path_join, get_media_type_from_extension, slugify_for_id # Placeholder for now, will create utils.py later

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    grid_size = get_setting("media_grid_size") or "medium"
    per_page = get_setting("per_page") or "20"
    return templates.TemplateResponse("index.html", {"request": request, "title": "Media Browser", "grid_size": grid_size, "per_page": per_page})

@router.get("/files", response_class=HTMLResponse)
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
    if per_page is None:
        per_page = int(get_setting("per_page") or "20")
    
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

@router.get("/api/tags", response_class=JSONResponse)
async def get_all_tags_api(): # Renamed to avoid conflict if get_all_tags_from_db is also imported directly
    tags = get_all_tags_from_db()
    return {"tags": tags}

@router.get("/media_content/{file_name:path}")
async def serve_media_file(file_name: str, request: Request): # request param might not be needed
    file_path = safe_path_join(MEDIA_DIR, file_name)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Original file not found: {file_name}")
    
    media_type_str = get_media_type_from_extension(file_path)
    content_type = f"{media_type_str}/{file_path.suffix.lstrip('.').lower()}" if media_type_str != "unknown" else "application/octet-stream"
    
    return FileResponse(file_path, media_type=content_type, filename=file_path.name)

@router.get("/media_content_transcoded/{file_name:path}")
async def serve_transcoded_media_file(file_name: str, request: Request): # request param might not be needed
    file_path = safe_path_join(TRANSCODED_DIR, file_name)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Transcoded file not found: {file_name}")
    return FileResponse(file_path, media_type="video/mp4", filename=file_path.name)

@router.get("/video/{video_name:path}", response_class=HTMLResponse, name="video_player_page")
async def video_player_page(
    request: Request, 
    video_name: str,
    search: str = Query(None), 
    media_type: str = Query(None),
    tags: str = Query(None),
    sort_by: str = Query("date_added"),
    sort_order: str = Query("desc")
):
    tags_filter = None
    if tags:
        tags_filter = [tag.strip() for tag in tags.split(',') if tag.strip()]
    
    video_details, video_queue, next_video = get_single_video_details_from_db(
        video_filename=video_name,
        search_query=search,
        media_type_filter=media_type,
        tags_filter=tags_filter,
        sort_by=sort_by,
        sort_order=sort_order
    )
    if not video_details:
        raise HTTPException(status_code=404, detail="Video not found in database")
    
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
        "auto_replay": auto_replay == "true",
        "search_query": search or "",
        "current_media_type": media_type or "",
        "current_tags": tags or "",
        "current_sort_by": sort_by,
        "current_sort_order": sort_order
    })

@router.post("/video/{video_id_db}/metadata", name="update_video_metadata")
async def update_video_metadata_route(video_id_db: int, request: Request, user_title: Optional[str] = Form(None), tags_str: Optional[str] = Form(None)):
    # This function is a wrapper, actual logic will be in data_access or similar
    # For now, direct implementation for simplicity, then refactor
    from database import db_connection # Local import to avoid circular if data_access uses this router
    import json # For parsing tags
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
            # Re-fetch and return current state if no changes
            current_video_details, _, _ = get_single_video_details_from_db(video_exists['filename'])
            if not current_video_details: raise HTTPException(status_code=404, detail="Video details not found.")
            return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": current_video_details})

        set_clause = ", ".join([f"{key} = ?" for key in fields_to_update.keys()])
        params = list(fields_to_update.values()) + [video_id_db]
        try:
            cursor.execute(f"UPDATE media_files SET {set_clause}, last_scanned=CURRENT_TIMESTAMP WHERE id = ?", params)
            conn.commit() # Ensure commit
            logger.info(f"Updated metadata for video ID {video_id_db}: {fields_to_update}")
        except sqlite3.Error as e: 
            logger.error(f"DB error updating metadata for video ID {video_id_db}: {e}")
            raise HTTPException(status_code=500, detail="DB error updating metadata.")
    
    updated_video_details, _, _ = get_single_video_details_from_db(video_exists['filename'])
    if not updated_video_details: raise HTTPException(status_code=404, detail="Video details not found post-update.")
    return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": updated_video_details})

@router.delete("/video/{video_id_db}/tag/{tag_name}", name="remove_video_tag")
async def remove_video_tag_route(video_id_db: int, tag_name: str, request: Request):
    from database import db_connection # Local import
    import json # For parsing tags
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename, tags FROM media_files WHERE id = ? AND media_type = 'video'", (video_id_db,))
        video_exists = cursor.fetchone()
        if not video_exists: 
            raise HTTPException(status_code=404, detail="Video not found.")

        current_tags_json = video_exists['tags'] or '[]'
        try:
            current_tags = json.loads(current_tags_json)
        except json.JSONDecodeError:
            current_tags = []

        updated_tags = [tag for tag in current_tags if tag.lower() != tag_name.lower()]
        
        if len(updated_tags) == len(current_tags):
            current_video_details, _, _ = get_single_video_details_from_db(video_exists['filename'])
            if not current_video_details: 
                raise HTTPException(status_code=404, detail="Video details not found.")
            return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": current_video_details})

        try:
            cursor.execute("UPDATE media_files SET tags = ?, last_scanned=CURRENT_TIMESTAMP WHERE id = ?", 
                         (json.dumps(updated_tags), video_id_db))
            conn.commit() # Ensure commit
            logger.info(f"Removed tag '{tag_name}' from video ID {video_id_db}")
        except sqlite3.Error as e: 
            logger.error(f"DB error removing tag from video ID {video_id_db}: {e}")
            raise HTTPException(status_code=500, detail="Database error removing tag.")
    
    updated_video_details, _, _ = get_single_video_details_from_db(video_exists['filename'])
    if not updated_video_details: 
        raise HTTPException(status_code=404, detail="Video details not found post-tag-removal.")
    return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": updated_video_details})

@router.get("/video/{current_video_id_db}/next")
async def get_next_video_route(current_video_id_db: int, response: Response):
    from database import db_connection # Local import
    logger.info(f"Received request for next video after ID: {current_video_id_db}")
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename FROM media_files WHERE media_type = 'video' ORDER BY filename ASC")
        all_videos_rows = cursor.fetchall()

    video_filenames_ordered = [row['filename'] for row in all_videos_rows]
    current_video_index = -1

    for i, row in enumerate(all_videos_rows):
        if row['id'] == current_video_id_db:
            current_video_index = i
            break

    if current_video_index != -1 and current_video_index + 1 < len(video_filenames_ordered):
        next_video_filename = video_filenames_ordered[current_video_index + 1]
        logger.info(f"Found next video: {next_video_filename}. Redirecting.")
        response.headers["HX-Redirect"] = f"/video/{next_video_filename}"
        return {"message": f"Redirecting to next video: {next_video_filename}"}
    else:
        logger.warning(f"No next video found after ID {current_video_id_db} or current video not in queue.")
        raise HTTPException(status_code=404, detail="No next video found in the queue.")

@router.delete("/delete-media-file/{video_id_db}", name="delete_media_file_endpoint") 
async def delete_media_file_route(video_id_db: int, response: Response):
    # Actual logic for deletion should be in a data_access or media_processing layer
    from database import db_connection # Local import
    import os # For os.remove if not using Path.unlink fully
    import sqlite3 # ensure sqlite3 is imported if used directly in this route temporarily
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT original_path, thumbnail_path, transcoded_path, preview_path, filename FROM media_files WHERE id = ?", (video_id_db,))
        db_row = cursor.fetchone()
        if not db_row:
            raise HTTPException(status_code=404, detail=f"Media file with ID {video_id_db} not found.")

        original_path_str = db_row['original_path']
        thumbnail_path_str = db_row['thumbnail_path']
        transcoded_path_str = db_row['transcoded_path']
        preview_path_str = db_row['preview_path']
        # video_name = db_row['filename'] # For re-rendering sidebar, if needed

        files_to_delete = []
        if original_path_str: files_to_delete.append(Path(original_path_str))
        if thumbnail_path_str: files_to_delete.append(BASE_DIR / thumbnail_path_str)
        if transcoded_path_str: files_to_delete.append(BASE_DIR / transcoded_path_str)
        if preview_path_str: files_to_delete.append(BASE_DIR / preview_path_str)

        failed_files = []
        for file_path in files_to_delete:
            if file_path.exists():
                try:
                    file_path.unlink(missing_ok=True)
                    logger.info(f"Deleted file: {file_path}")
                except OSError as e:
                    logger.error(f"Failed to delete file {file_path}: {e}")
                    failed_files.append(str(file_path))

        if original_path_str: # Check if there was an original path to begin with
            try:
                cursor.execute("DELETE FROM media_files WHERE id = ?", (video_id_db,))
                conn.commit()
                logger.info(f"Deleted media file record from DB for ID {video_id_db}.")
            except sqlite3.Error as e:
                logger.error(f"Failed to delete media file record from DB for ID {video_id_db}: {e}")
                raise HTTPException(status_code=500, detail="Failed to delete media file record from database.")
        else:
             # This case should ideally not occur if db_row was fetched successfully with an ID.
             # It implies original_path was NULL in DB, which is problematic for a media entry.
             logger.error(f"Original file path not found in database record for ID {video_id_db}, cannot delete DB entry.")
             raise HTTPException(status_code=500, detail="Original file path not found in database record, cannot delete DB entry.")

        if failed_files:
            # Even if some asset files failed to delete, the main DB entry is gone.
            # Redirect to home as the item is no longer accessible via player page.
            response.headers["HX-Redirect"] = "/" 
            # Consider returning a special message to display on the home page about partial deletion.
            # For now, a simple JSON response might be swallowed by HX-Redirect.
            # The important part is the redirect.
            return {"message": "Media file record deleted, but some associated files could not be deleted. Redirecting...", "failed_files": failed_files, "redirect_url": "/"}
        else:
            response.headers["HX-Redirect"] = "/"
            return {"message": f"Media file and all associated assets deleted for ID {video_id_db}. Redirecting...", "redirect_url": "/"} 