import sqlite3
import json
from pathlib import Path
from datetime import timedelta

from database import db_connection # Assuming database.py is in the same root
from config import (
    logger, MEDIA_DIR, BASE_DIR, TRANSCODED_DIR, PREVIEWS_DIR, 
    SUPPORTED_VIDEO_EXTENSIONS, SUPPORTED_AUDIO_EXTENSIONS, SUPPORTED_IMAGE_EXTENSIONS,
    slugify_for_id # Used by functions being moved here
)
from media_processing import (
    get_media_duration, get_display_thumbnail_url, 
    get_thumbnail_path, get_preview_path # Used by scan_media_directory_and_update_db
)

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

    conn = None # Manual connection management for this bulk operation
    try:
        conn = db_connection(). __enter__() # Manually enter context for more control
        cursor = conn.cursor()

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
                if suffix in SUPPORTED_VIDEO_EXTENSIONS: media_type = 'video'
                elif suffix in SUPPORTED_AUDIO_EXTENSIONS: media_type = 'audio'
                elif suffix in SUPPORTED_IMAGE_EXTENSIONS: media_type = 'image'
                else:
                    continue

                logger.info(f"Processing file: {filename} (Type: {media_type})")
                cursor.execute("SELECT id, last_scanned FROM media_files WHERE original_path = ?", (original_path_str,))
                existing_file = cursor.fetchone()

                duration_str = get_media_duration(file_path_obj)
                duration_seconds = None
                if duration_str:
                    parts = list(map(int, duration_str.split(':')))
                    duration_seconds = timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2]).total_seconds()

                width, height, fps = None, None, None
                size_bytes = file_path_obj.stat().st_size
                
                if media_type == 'video':
                    from moviepy.editor import VideoFileClip # Local import to avoid circular dependency if media_processing also imports from data_access
                    try:
                        with VideoFileClip(str(file_path_obj.resolve())) as clip:
                            width, height = clip.size
                            fps = clip.fps
                    except Exception as e:
                        logger.warning(f"Could not get video metadata for {filename}: {e}")
                elif media_type == 'image':
                    from PIL import Image # Local import
                    try:
                        with Image.open(file_path_obj.resolve()) as img:
                            width, height = img.size
                    except Exception as e:
                        logger.warning(f"Could not get image metadata for {filename}: {e}")
                
                actual_thumbnail_p = get_thumbnail_path(file_path_obj)
                has_specific_thumbnail = actual_thumbnail_p.exists()
                db_thumbnail_path = str(actual_thumbnail_p.relative_to(BASE_DIR)) if has_specific_thumbnail else None
                
                transcoded_p = TRANSCODED_DIR / f"{slugify_for_id(file_path_obj.stem)}.mp4"
                has_transcoded_version = transcoded_p.exists()
                db_transcoded_path = str(transcoded_p.relative_to(BASE_DIR)) if has_transcoded_version else None

                preview_p = get_preview_path(file_path_obj)
                has_preview = preview_p.exists()
                db_preview_path = str(preview_p.relative_to(BASE_DIR)) if has_preview else None

                metadata_dict = {"source": "filesystem_scan"}
                if width and height: metadata_dict['resolution'] = f"{width}x{height}"
                if fps: metadata_dict['fps'] = round(fps, 2)
                metadata_json_str = json.dumps(metadata_dict)

                if existing_file:
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
                    logger.debug(f"Adding new DB entry for: {filename}")
                    cursor.execute("""
                        INSERT INTO media_files 
                            (filename, original_path, media_type, user_title, duration, width, height, fps, size_bytes, 
                             last_scanned, thumbnail_path, has_specific_thumbnail, 
                             transcoded_path, has_transcoded_version, preview_path, has_preview, 
                             tags, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (filename, original_path_str, media_type, None,
                          duration_seconds, width, height, fps, size_bytes,
                          db_thumbnail_path, has_specific_thumbnail, 
                          db_transcoded_path, has_transcoded_version, db_preview_path, has_preview, 
                          '[]', metadata_json_str))
                conn.commit() # Commit after each file processing

        db_filenames_set = set(db_files_map.keys())
        orphaned_filenames = db_filenames_set - found_files_in_scan
        if orphaned_filenames:
            logger.info(f"Found orphaned files in DB (will be removed): {orphaned_filenames}")
            for orphaned_file_name in orphaned_filenames:
                file_id_to_delete = db_files_map[orphaned_file_name]
                cursor.execute("DELETE FROM media_files WHERE id = ?", (file_id_to_delete,))
                logger.info(f"Removed orphaned DB entry for: {orphaned_file_name} (ID: {file_id_to_delete})")
            conn.commit()
        logger.info("Media scan and database update completed.")
    except sqlite3.Error as e:
        logger.error(f"SQLite error during media scan: {e}")
        if conn: conn.rollback()
        # raise # Optionally re-raise or handle more gracefully
    except Exception as e:
        logger.error(f"Unexpected error during media scan: {e}")
        if conn: conn.rollback()
        # raise
    finally:
        if conn:
            db_connection().__exit__(None, None, None) # Manually exit context

# --- Data Formatting and Fetching Logic ---
def format_media_duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    return str(timedelta(seconds=int(seconds)))

def get_media_files_from_db(
    search_query: str = None, 
    page: int = 1, 
    per_page: int = 20, 
    media_type_filter: str = None,
    tags_filter: list = None,
    sort_by: str = "date_added",
    sort_order: str = "desc"
):
    """
    Fetches media files from the database with comprehensive search and filtering options.
    
    Args:
        search_query: Text to search in filename, user_title, and tags
        page: Page number for pagination
        per_page: Items per page
        media_type_filter: Filter by media type ('video', 'audio', 'image')
        tags_filter: List of tags to search for (OR operation)
        sort_by: Field to sort by ('date_added', 'filename', 'user_title', 'duration', 'size_bytes')
        sort_order: Sort order ('asc' or 'desc')
    """
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Build WHERE conditions
        where_conditions = []
        count_params = []
        
        # Search query - search in filename, user_title, and tags
        if search_query:
            search_conditions = [
                "filename LIKE ?",
                "user_title LIKE ?",
                "tags LIKE ?"
            ]
            search_param = f"%{search_query}%"
            where_conditions.append(f"({' OR '.join(search_conditions)})")
            count_params.extend([search_param, search_param, search_param])
        
        # Media type filter
        if media_type_filter and media_type_filter in ['video', 'audio', 'image']:
            where_conditions.append("media_type = ?")
            count_params.append(media_type_filter)
        
        # Tags filter (OR operation - match any of the provided tags)
        if tags_filter and isinstance(tags_filter, list) and tags_filter:
            tag_conditions = []
            for tag in tags_filter:
                if tag.strip():  # Only add non-empty tags
                    tag_conditions.append("tags LIKE ?")
                    count_params.append(f'%"{tag.strip()}"%')
            if tag_conditions:
                where_conditions.append(f"({' OR '.join(tag_conditions)})")
        
        # Build the WHERE clause
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # Get total count for pagination
        count_query = f"SELECT COUNT(*) as total FROM media_files {where_clause}"
        cursor.execute(count_query, count_params)
        total_count = cursor.fetchone()['total']
        
        # Calculate offset and limit
        offset = (page - 1) * per_page
        
        # Build ORDER BY clause
        valid_sort_fields = {
            'date_added': 'date_added',
            'filename': 'filename',
            'user_title': 'COALESCE(user_title, filename)',  # Use filename if user_title is NULL
            'duration': 'duration',
            'size_bytes': 'size_bytes',
            'media_type': 'media_type'
        }
        
        sort_field = valid_sort_fields.get(sort_by, 'date_added')
        sort_direction = 'ASC' if sort_order.lower() == 'asc' else 'DESC'
        order_clause = f"ORDER BY {sort_field} {sort_direction}"
        
        # Main query with pagination
        query = f"SELECT * FROM media_files {where_clause} {order_clause} LIMIT ? OFFSET ?"
        params = count_params + [per_page, offset]
        
        cursor.execute(query, params)
        db_rows = cursor.fetchall()

    media_list = []
    for row in db_rows:
        thumbnail_url = None
        if row['has_specific_thumbnail'] and row['thumbnail_path']:
            thumbnail_url = f"/{row['thumbnail_path'].replace('\\\\ ', '/')}" 
        else:
            if row['media_type'] == 'video': thumbnail_url = "/static/icons/generic-video-icon.svg"
            elif row['media_type'] == 'image': thumbnail_url = "/static/icons/generic-image-icon.svg"
            elif row['media_type'] == 'audio': thumbnail_url = "/static/icons/generic-audio-icon.svg"
        
        playable_path_url = f"/media_content/{row['filename']}"
        if row['has_transcoded_version'] and row['transcoded_path']:
            playable_path_url = f"/media_content_transcoded/{Path(row['transcoded_path']).name}"

        preview_url = f"/{row['preview_path'].replace('\\\\ ', '/')}" if row['has_preview'] and row['preview_path'] else None
        display_name = row['user_title'] if row['user_title'] else row['filename']
        tags_list = json.loads(row['tags']) if row['tags'] else []
        metadata = json.loads(row['metadata_json']) if row['metadata_json'] else {}

        media_list.append({
            "id_db": row['id'],
            "name": row['filename'],
            "display_title": display_name,
            "user_title": row['user_title'],
            "path": playable_path_url,
            "original_path_for_download": f"/media_content/{row['filename']}",
            "type": row['media_type'],
            "thumbnail": thumbnail_url,
            "duration": format_media_duration(row['duration']),
            "id": slugify_for_id(row['filename']),
            "has_specific_thumbnail": bool(row['has_specific_thumbnail']),
            "has_transcoded_version": bool(row['has_transcoded_version']),
            "has_preview": bool(row['has_preview']),
            "preview_url": preview_url,
            "tags": tags_list,
            "size_bytes": row['size_bytes'],
            "resolution": metadata.get('resolution', f"{row['width']}x{row['height']}" if row['width'] and row['height'] else None),
            "fps": metadata.get('fps', row['fps'] if row['fps'] else None),
            "original_full_path": row['original_path']
        })
    
    # Calculate pagination info
    total_pages = (total_count + per_page - 1) // per_page  # Ceiling division
    has_prev = page > 1
    has_next = page < total_pages
    
    return {
        'media_files': media_list,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total_count': total_count,
            'total_pages': total_pages,
            'has_prev': has_prev,
            'has_next': has_next,
            'prev_page': page - 1 if has_prev else None,
            'next_page': page + 1 if has_next else None
        }
    }

def get_single_video_details_from_db(video_filename: str):
    """Fetches detailed information for a single video from the database by its filename and also gets the full video queue."""
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Fetch details for the current video
        cursor.execute("SELECT * FROM media_files WHERE filename = ? AND media_type = 'video'", (video_filename,))
        row = cursor.fetchone()

        if not row: return None, [], None # Return None for video, empty queue, None for next video

        # Fetch all video files for the queue, ordered by filename
        cursor.execute("SELECT id, filename, user_title FROM media_files WHERE media_type = 'video' ORDER BY filename ASC")
        all_videos_rows = cursor.fetchall()

    video_list = []
    current_video_index = -1
    for i, v_row in enumerate(all_videos_rows):
        video_list.append({
            "id_db": v_row['id'],
            "name": v_row['filename'],
            "display_title": v_row['user_title'] if v_row['user_title'] else v_row['filename']
        })
        if v_row['id'] == row['id']:
            current_video_index = i

    next_video = None
    if current_video_index != -1 and current_video_index + 1 < len(video_list):
        next_video = video_list[current_video_index + 1]

    # Format current video details
    thumbnail_url = f"/{row['thumbnail_path'].replace('\\ ', '/')}" if row['has_specific_thumbnail'] and row['thumbnail_path'] else "/static/icons/generic-video-icon.svg"
    playable_path_url = f"/media_content/{row['filename']}"
    if row['has_transcoded_version'] and row['transcoded_path']:
        playable_path_url = f"/media_content_transcoded/{Path(row['transcoded_path']).name}"
    
    preview_url = f"/{row['preview_path'].replace('\\ ', '/')}" if row['has_preview'] and row['preview_path'] else None
    display_name = row['user_title'] if row['user_title'] else row['filename']
    tags_list = json.loads(row['tags']) if row['tags'] else []
    metadata = json.loads(row['metadata_json']) if row['metadata_json'] else {}

    current_video_details = {
        "id_db": row['id'],
        "name": row['filename'],
        "display_title": display_name,
        "user_title": row['user_title'],
        "path": playable_path_url,
        "original_path_for_download": f"/media_content/{row['filename']}",
        "type": row['media_type'],
        "thumbnail": thumbnail_url,
        "duration": format_media_duration(row['duration']),
        "id": slugify_for_id(row['filename']),
        "has_specific_thumbnail": bool(row['has_specific_thumbnail']),
        "has_transcoded_version": bool(row['has_transcoded_version']),
        "has_preview": bool(row['has_preview']),
        "preview_url": preview_url,
        "tags": tags_list,
        "size": row['size_bytes'],
        "resolution": metadata.get('resolution', f"{row['width']}x{row['height']}" if row['width'] and row['height'] else 'N/A'),
        "fps": metadata.get('fps', row['fps'] if row['fps'] else 'N/A'),
        "original_full_path": row['original_path']
    }

    return current_video_details, video_list, next_video # Return current video, queue, and next video 

def get_all_tags_from_db():
    """Get all unique tags from the database for filter suggestions."""
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT tags FROM media_files WHERE tags IS NOT NULL AND tags != '[]'")
        rows = cursor.fetchall()
    
    all_tags = set()
    for row in rows:
        try:
            tags_list = json.loads(row['tags'])
            all_tags.update(tags_list)
        except (json.JSONDecodeError, TypeError):
            continue
    
    return sorted(list(all_tags)) 