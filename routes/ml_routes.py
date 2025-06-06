from fastapi import APIRouter, Request, HTTPException, BackgroundTasks, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
from typing import Optional, List
import json

from config import templates, MEDIA_DIR, BASE_DIR, logger
from data_access import get_media_files_from_db, get_single_video_details_from_db
from jobs_manager import create_background_job
from ml_processing import (
    process_video_frames_with_embeddings, cleanup_video_analysis,
    find_similar_frames, get_video_embeddings_from_chroma, 
    get_chroma_collection_stats, migrate_existing_embeddings_to_chroma
)

router = APIRouter()

@router.get("/ml-processing", response_class=HTMLResponse, name="ml_processing_page")
async def ml_processing_page(request: Request):
    """Main ML processing page"""
    return templates.TemplateResponse("ml_processing.html", {
        "request": request,
        "title": "ML Video Processing"
    })

@router.get("/api/videos-for-ml", response_class=JSONResponse)
async def get_videos_for_ml(
    sort_by: str = Query("date_added"),
    sort_order: str = Query("desc")
):
    """Get list of videos available for ML processing"""
    try:
        result = get_media_files_from_db(
            media_type_filter="video",
            page=1,
            per_page=1000,  # Get all videos
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        videos = []
        for video in result['media_files']:
            # Get raw duration in seconds from database instead of formatted string
            raw_duration = None
            try:
                # Query database directly for raw duration value
                from database import db_connection
                with db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT duration FROM media_files WHERE id = ?", (video['id_db'],))
                    row = cursor.fetchone()
                    if row and row['duration']:
                        raw_duration = row['duration']
            except Exception as e:
                logger.warning(f"Could not get raw duration for video {video['id_db']}: {e}")
            
            videos.append({
                'id': video['id_db'],
                'filename': video['name'],
                'display_title': video['display_title'],
                'duration': raw_duration,  # Pass raw seconds instead of formatted string
                'size_mb': round(video['size_bytes'] / (1024 * 1024), 1) if video['size_bytes'] else 0,
                'thumbnail': video['thumbnail'],  # Include thumbnail URL
                'has_ml_analysis': video.get('has_ml_analysis', False),
                'ml_analysis_info': video.get('ml_analysis_info', {})
            })
        
        return {"videos": videos}
    except Exception as e:
        logger.error(f"Error getting videos for ML: {e}")
        raise HTTPException(status_code=500, detail="Failed to get videos")

@router.post("/process-video-frames", response_class=JSONResponse)
async def process_video_frames_endpoint(
    background_tasks: BackgroundTasks,
    video_id: int = Form(...),
    frame_interval: int = Form(100),
    model_name: str = Form("clip-ViT-B-32")
):
    """Start video frame processing job"""
    try:
        # Get video details
        video_files = get_media_files_from_db(page=1, per_page=1000)
        video = None
        for v in video_files['media_files']:
            if v['id_db'] == video_id:
                video = v
                break
        
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Create background job
        job = create_background_job(
            "video_frame_processing",
            f"Processing frames for {video['display_title']} (every {frame_interval} frames)"
        )
        
        # Start processing in background
        background_tasks.add_task(
            process_video_frames_with_embeddings,
            job,
            video,
            frame_interval,
            model_name
        )
        
        return {
            "job_id": job.job_id,
            "message": f"Started frame processing for {video['display_title']}"
        }
        
    except Exception as e:
        logger.error(f"Error starting video frame processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/video/{video_id}/frame-analysis", response_class=JSONResponse)
async def get_video_frame_analysis(video_id: int):
    """Get frame analysis results for a video"""
    try:
        # Check if analysis exists
        analysis_dir = BASE_DIR / "ml_analysis" / f"video_{video_id}"
        embeddings_file = analysis_dir / "embeddings.json"
        
        if not embeddings_file.exists():
            return {"has_analysis": False, "message": "No frame analysis found for this video"}
        
        # Load analysis data
        with open(embeddings_file, 'r') as f:
            analysis_data = json.load(f)
        
        return {
            "has_analysis": True,
            "frame_count": len(analysis_data.get("frames", [])),
            "model_used": analysis_data.get("model_name", "unknown"),
            "processed_at": analysis_data.get("processed_at", "unknown"),
            "frames": analysis_data.get("frames", [])[:10]  # Return first 10 frames as preview
        }
        
    except Exception as e:
        logger.error(f"Error getting frame analysis for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/video/{video_id}/ml-analysis", response_class=HTMLResponse)
async def delete_video_ml_analysis(video_id: int, request: Request):
    """Delete ML analysis data for a video"""
    error_message = None
    
    try:
        # Get video details first
        from database import db_connection
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename FROM media_files WHERE id = ? AND media_type = 'video'", (video_id,))
            db_row = cursor.fetchone()
            
            if not db_row:
                raise HTTPException(status_code=404, detail=f"Video with ID {video_id} not found")
            
            video_filename = db_row['filename']
        
        # Check if analysis exists
        analysis_dir = BASE_DIR / "ml_analysis" / f"video_{video_id}"
        if not analysis_dir.exists():
            error_message = "No ML analysis found for this video"
        else:
            # Delete the analysis data
            success = cleanup_video_analysis(video_id)
            
            if success:
                logger.info(f"Successfully deleted ML analysis for video {video_id}")
            else:
                error_message = "Failed to delete ML analysis data"
        
        # Get updated video details and return the sidebar template
        updated_video_details, _, _ = get_single_video_details_from_db(video_filename)
        context = {"request": request, "video": updated_video_details}
        if error_message:
            context["error_message"] = error_message
        
        return templates.TemplateResponse("_video_metadata_sidebar.html", context)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting ML analysis for video {video_id}: {e}")
        # Try to get video details for error display
        try:
            from database import db_connection
            with db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT filename FROM media_files WHERE id = ?", (video_id,))
                db_row = cursor.fetchone()
                if db_row:
                    updated_video_details, _, _ = get_single_video_details_from_db(db_row['filename'])
                    context = {"request": request, "video": updated_video_details, "error_message": f"Error deleting ML analysis: {str(e)}"}
                    return templates.TemplateResponse("_video_metadata_sidebar.html", context)
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/chroma-stats", response_class=JSONResponse)
async def get_chroma_stats():
    """Get ChromaDB collection statistics"""
    try:
        stats = get_chroma_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting ChromaDB stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/migrate-embeddings", response_class=JSONResponse)
async def migrate_embeddings_to_chroma():
    """Migrate existing JSON embeddings to ChromaDB"""
    try:
        result = migrate_existing_embeddings_to_chroma()
        return result
    except Exception as e:
        logger.error(f"Error migrating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/video/{video_id}/embeddings", response_class=JSONResponse)
async def get_video_embeddings(video_id: int):
    """Get video embeddings from ChromaDB"""
    try:
        embeddings_data = get_video_embeddings_from_chroma(video_id)
        if embeddings_data:
            return {
                "has_embeddings": True,
                "data": embeddings_data
            }
        else:
            return {
                "has_embeddings": False,
                "message": "No embeddings found for this video"
            }
    except Exception as e:
        logger.error(f"Error getting video embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/similarity-search", response_class=JSONResponse)
async def similarity_search(
    query_embedding: List[float] = Form(...),
    n_results: int = Form(10),
    video_id: Optional[int] = Form(None)
):
    """Find similar frames using vector similarity search"""
    try:
        # Convert string representation to list if needed
        if isinstance(query_embedding, str):
            import json
            query_embedding = json.loads(query_embedding)
        
        similar_frames = find_similar_frames(
            query_embedding=query_embedding,
            n_results=n_results,
            video_id=video_id
        )
        
        return {
            "similar_frames": similar_frames,
            "query_info": {
                "n_results": n_results,
                "video_id": video_id,
                "embedding_dimension": len(query_embedding)
            }
        }
    except Exception as e:
        logger.error(f"Error performing similarity search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/semantic-search", response_class=JSONResponse)
async def semantic_search(
    query_text: str = Form(...),
    n_results: int = Form(20),
    similarity_threshold: float = Form(0.7)
):
    """Search for videos using natural language queries against frame embeddings"""
    try:
        from ml_processing import semantic_search_videos
        
        results = semantic_search_videos(
            query_text=query_text,
            n_results=n_results,
            similarity_threshold=similarity_threshold
        )
        
        return {
            "videos": results,
            "query_info": {
                "query_text": query_text,
                "n_results": n_results,
                "similarity_threshold": similarity_threshold,
                "total_matches": len(results)
            }
        }
    except Exception as e:
        logger.error(f"Error performing semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 