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

@router.get("/api/semantic-search/debug", response_class=JSONResponse)
async def semantic_search_debug():
    """Debug endpoint to check semantic search capabilities"""
    try:
        from ml_processing import chroma_manager
        from data_access import get_media_files_from_db
        
        # Check ChromaDB status
        chroma_available = chroma_manager.is_available()
        chroma_stats = chroma_manager.get_collection_stats()
        
        # Check for videos with ML analysis
        all_videos = get_media_files_from_db(media_type_filter="video", page=1, per_page=1000)
        videos_with_ml = [v for v in all_videos['media_files'] if v.get('has_ml_analysis')]
        
        # Check sentence transformers
        try:
            from sentence_transformers import SentenceTransformer
            st_available = True
            try:
                # Try to load the default model
                model = SentenceTransformer("clip-ViT-B-32")
                model_loadable = True
            except Exception as e:
                model_loadable = False
                model_error = str(e)
        except ImportError as e:
            st_available = False
            model_loadable = False
            model_error = str(e)
        
        debug_info = {
            "chromadb": {
                "available": chroma_available,
                "stats": chroma_stats
            },
            "sentence_transformers": {
                "available": st_available,
                "model_loadable": model_loadable,
                "model_error": model_error if not model_loadable else None
            },
            "videos": {
                "total_videos": len(all_videos['media_files']),
                "videos_with_ml_analysis": len(videos_with_ml),
                "ml_analysis_video_ids": [v['id_db'] for v in videos_with_ml]
            },
            "recommendations": []
        }
        
        # Add recommendations based on findings
        if not chroma_available:
            debug_info["recommendations"].append("ChromaDB is not available. Check if it's properly initialized.")
        
        if chroma_stats.get("total_frames", 0) == 0:
            debug_info["recommendations"].append("No frame embeddings found in ChromaDB. Process videos with ML analysis first.")
        
        if not st_available:
            debug_info["recommendations"].append("Sentence Transformers not available. Install with: pip install sentence-transformers")
        
        if not model_loadable:
            debug_info["recommendations"].append("Cannot load CLIP model. Check internet connection or model cache.")
        
        if len(videos_with_ml) == 0:
            debug_info["recommendations"].append("No videos have been processed with ML analysis. Go to ML Processing page to analyze videos.")
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Error in semantic search debug: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/semantic-search", response_class=JSONResponse)
async def semantic_search_media(
    query_text: str = Form(...),
    page: int = Form(1),
    per_page: int = Form(20),
    similarity_threshold: float = Form(0.7)
):
    """Search for videos using natural language queries against frame embeddings"""
    try:
        from ml_processing import semantic_search_videos
        
        # Perform semantic search
        semantic_results = semantic_search_videos(
            query_text=query_text,
            n_results=per_page * 3,  # Get more results to account for pagination
            similarity_threshold=similarity_threshold
        )
        
        # Apply pagination to semantic results
        total_count = len(semantic_results)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_results = semantic_results[start_idx:end_idx]
        
        # Create pagination info
        pagination = {
            "page": page,
            "per_page": per_page,
            "total_count": total_count,
            "total_pages": (total_count + per_page - 1) // per_page,
            "has_prev": page > 1,
            "has_next": page * per_page < total_count,
            "prev_page": page - 1 if page > 1 else None,
            "next_page": page + 1 if page * per_page < total_count else None
        }
        
        return {
            "media_files": paginated_results,
            "pagination": pagination,
            "search_type": "semantic",
            "query_text": query_text,
            "similarity_threshold": similarity_threshold,
            "query_info": {
                "total_frames_searched": len(semantic_results) * 5 if semantic_results else 0,  # Estimate
                "threshold_used": similarity_threshold,
                "model_used": "clip-ViT-B-32"  # Default, could be detected
            }
        }
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    sort_order: str = Query("desc"),
    t: float = Query(None, description="Start time in seconds")
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
        "current_sort_order": sort_order,
        "start_time": t
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

@router.post("/api/extract-frames/{video_name:path}", response_class=JSONResponse)
async def extract_video_frames(
    video_name: str,
    timestamps: str = Form(...),  # Comma-separated timestamps in seconds
    max_frames: int = Form(10)
):
    """Extract frames from video at specific timestamps for semantic search"""
    try:
        import tempfile
        import subprocess
        from pathlib import Path
        import base64
        
        # Get video path
        video_path = safe_path_join(MEDIA_DIR, video_name)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Get video duration first
        try:
            duration_cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'csv=p=0', str(video_path)
            ]
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=10)
            if duration_result.returncode == 0:
                video_duration = float(duration_result.stdout.strip())
            else:
                video_duration = None
        except:
            video_duration = None
        
        # Parse timestamps
        try:
            timestamp_list = [float(t.strip()) for t in timestamps.split(',') if t.strip()]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format")
        
        # Filter timestamps to be within video duration
        if video_duration:
            original_count = len(timestamp_list)
            timestamp_list = [t for t in timestamp_list if 0 <= t < video_duration]
            if len(timestamp_list) < original_count:
                logger.warning(f"Filtered {original_count - len(timestamp_list)} timestamps outside video duration ({video_duration}s)")
        
        # Limit number of frames
        timestamp_list = timestamp_list[:max_frames]
        
        if not timestamp_list:
            raise HTTPException(status_code=400, detail="No valid timestamps within video duration")
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            extracted_frames = []
            
            for i, timestamp in enumerate(timestamp_list):
                frame_filename = f"frame_{i:03d}.jpg"
                frame_path = temp_path / frame_filename
                
                # Extract frame at specific timestamp using FFmpeg
                cmd = [
                    'ffmpeg',
                    '-i', str(video_path),
                    '-ss', str(timestamp),  # Seek to timestamp
                    '-vframes', '1',  # Extract only 1 frame
                    '-q:v', '2',  # High quality
                    '-y',  # Overwrite output file
                    str(frame_path)
                ]
                
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode == 0 and frame_path.exists():
                        # Read frame and convert to base64
                        with open(frame_path, 'rb') as f:
                            frame_data = f.read()
                            frame_base64 = base64.b64encode(frame_data).decode('utf-8')
                        
                        extracted_frames.append({
                            "timestamp": timestamp,
                            "frame_index": i,
                            "frame_data": f"data:image/jpeg;base64,{frame_base64}",
                            "size": len(frame_data)
                        })
                    else:
                        logger.warning(f"Failed to extract frame at {timestamp}s: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    logger.error(f"Timeout extracting frame at {timestamp}s")
                except Exception as e:
                    logger.error(f"Error extracting frame at {timestamp}s: {e}")
        
        return {
            "video_name": video_name,
            "extracted_frames": extracted_frames,
            "total_frames": len(extracted_frames),
            "requested_timestamps": timestamp_list
        }
        
    except Exception as e:
        logger.error(f"Error extracting frames from {video_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/semantic-search-by-frame", response_class=JSONResponse)
async def semantic_search_by_frame(
    frame_data: str = Form(...),  # Base64 encoded image data
    similarity_threshold: float = Form(0.35),
    max_results: int = Form(20)
):
    """Perform semantic search using a frame image"""
    try:
        import base64
        import tempfile
        from PIL import Image
        import io
        
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            if frame_data.startswith('data:image'):
                frame_data = frame_data.split(',')[1]
            
            image_bytes = base64.b64decode(frame_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
        
        # Generate embedding for the frame
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load CLIP model (should match the model used for stored embeddings)
            model_name = "clip-ViT-B-32"
            model = SentenceTransformer(model_name)
            
            # Generate embedding
            frame_embedding = model.encode(image).tolist()
            
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"ML libraries not available: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating embedding: {e}")
        
        # Perform similarity search
        from ml_processing import chroma_manager
        
        if not chroma_manager.is_available():
            raise HTTPException(status_code=500, detail="ChromaDB not available")
        
        # Search for similar frames
        similar_frames = chroma_manager.find_similar_frames(
            query_embedding=frame_embedding,
            n_results=max_results * 3,  # Get more frames to aggregate by video
            video_id=None  # Search across all videos
        )
        
        # Group results by video and calculate video-level scores
        video_scores = {}
        video_best_frames = {}  # Store best frame info for each video
        frames_above_threshold = 0
        
        for frame in similar_frames:
            video_id = frame["video_id"]
            similarity_score = frame["similarity_score"]
            
            # Only include frames above the similarity threshold
            if similarity_score >= similarity_threshold:
                frames_above_threshold += 1
                if video_id not in video_scores:
                    video_scores[video_id] = []
                    video_best_frames[video_id] = []
                
                video_scores[video_id].append(similarity_score)
                video_best_frames[video_id].append({
                    "similarity_score": similarity_score,
                    "timestamp": frame["timestamp"],
                    "frame_number": frame["frame_number"]
                })
        
        # Calculate aggregated scores for each video
        video_results = []
        for video_id, scores in video_scores.items():
            if not scores:
                continue
            
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            frame_count = len(scores)
            
            # Find the best matching frame for this video
            best_frame = None
            if video_id in video_best_frames:
                # Sort frames by similarity score and get the best one
                sorted_frames = sorted(video_best_frames[video_id], 
                                     key=lambda x: x["similarity_score"], reverse=True)
                best_frame = sorted_frames[0]
            
            # Weighted score
            weighted_score = (max_score * 0.6) + (avg_score * 0.3) + (min(frame_count / 10, 1.0) * 0.1)
            
            result_data = {
                "video_id": video_id,
                "max_similarity": max_score,
                "avg_similarity": avg_score,
                "weighted_score": weighted_score,
                "matching_frames": frame_count
            }
            
            # Add best frame timestamp if available
            if best_frame:
                result_data["best_frame_timestamp"] = best_frame["timestamp"]
                result_data["best_frame_number"] = best_frame["frame_number"]
            
            video_results.append(result_data)
        
        # Sort by weighted score
        video_results.sort(key=lambda x: x["weighted_score"], reverse=True)
        video_results = video_results[:max_results]
        
        # Get video metadata
        from data_access import get_media_files_from_db
        
        all_videos_result = get_media_files_from_db(
            media_type_filter="video",
            page=1,
            per_page=1000
        )
        
        video_lookup = {video['id_db']: video for video in all_videos_result['media_files']}
        
        # Enrich results with video metadata
        enriched_results = []
        for result in video_results:
            video_id = result["video_id"]
            video_metadata = video_lookup.get(video_id)
            
            if video_metadata:
                enriched_result = {
                    **video_metadata,
                    "semantic_search_score": result["weighted_score"],
                    "max_similarity": result["max_similarity"],
                    "avg_similarity": result["avg_similarity"],
                    "matching_frames": result["matching_frames"],
                    "search_type": "frame_based_semantic"
                }
                
                # Add best frame timestamp if available
                if "best_frame_timestamp" in result:
                    enriched_result["best_frame_timestamp"] = result["best_frame_timestamp"]
                if "best_frame_number" in result:
                    enriched_result["best_frame_number"] = result["best_frame_number"]
                
                enriched_results.append(enriched_result)
        
        return {
            "search_results": enriched_results,
            "total_results": len(enriched_results),
            "frames_above_threshold": frames_above_threshold,
            "similarity_threshold": similarity_threshold,
            "search_type": "frame_based_semantic"
        }
        
    except Exception as e:
        logger.error(f"Error in frame-based semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 