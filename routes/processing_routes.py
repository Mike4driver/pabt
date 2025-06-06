from fastapi import APIRouter, Request, HTTPException, Response, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
from typing import Optional
import sqlite3
import os

from config import (
    templates, logger, BASE_DIR, TRANSCODED_DIR, 
    # MEDIA_DIR will be needed by some functions if they interact with original files directly,
    # but ideally original path is fetched from DB.
)
from database import db_connection, get_setting # get_setting might be for options
from data_access import get_single_video_details_from_db, scan_media_directory_and_update_db
from media_processing import (
    _actually_create_thumbnail, get_thumbnail_path, 
    transcode_video, slugify_for_id, 
    create_hover_preview, get_preview_path
)
from jobs_manager import create_background_job, get_background_job

router = APIRouter()

# --- Thumbnail Endpoints & Background Tasks --- 
@router.post("/generate-thumbnail/{video_name:path}", name="generate_specific_thumbnail_endpoint")
async def generate_specific_thumbnail_endpoint_route(video_name: str, response: Response, request: Request):
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
                    conn.commit()
                    logger.info(f"Database updated for new thumbnail of {video_name}.")
                except sqlite3.Error as e:
                    logger.error(f"Failed to update database for thumbnail {video_name}: {e}")
                    updated_video_details, _, _ = get_single_video_details_from_db(video_name)
                    if not updated_video_details: 
                        raise HTTPException(status_code=500, detail="Thumbnail generated, DB update failed, and failed to reload video details.")
                    return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": updated_video_details, "error_message": "Thumbnail generated but DB update failed."})
            
            response.headers["X-Thumbnail-Url"] = thumbnail_url_rel
            updated_video_details, _, _ = get_single_video_details_from_db(video_name)
            if not updated_video_details: 
                 raise HTTPException(status_code=404, detail="Video details not found after thumbnail generation.")
            return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": updated_video_details})
        else:
            current_video_details, _, _ = get_single_video_details_from_db(video_name)
            if not current_video_details: 
                 raise HTTPException(status_code=500, detail="Failed to generate thumbnail and also failed to reload video details.")
            return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": current_video_details, "error_message": "Failed to generate thumbnail. Check logs."})

@router.post("/generate-all-video-thumbnails")
async def generate_all_thumbnails_endpoint_route(background_tasks: BackgroundTasks):
    job = create_background_job("thumbnails", "Generating video thumbnails")
    background_tasks.add_task(generate_all_thumbnails_background_task, job.job_id)
    return {"message": "Thumbnail generation started in background", "job_id": job.job_id}

def generate_all_thumbnails_background_task(job_id: str):
    job = get_background_job(job_id)
    if not job: return
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename, original_path FROM media_files WHERE media_type = 'video' AND (has_specific_thumbnail = FALSE OR thumbnail_path IS NULL)")
            videos_to_process = cursor.fetchall()
            if not videos_to_process: 
                job.complete({"message": "No video thumbnails to generate.", "generated": 0, "failed": 0}); return
            job.update_progress(0, len(videos_to_process), "Starting...")
            generated_count, failed_count = 0, 0
            for i, video_row in enumerate(videos_to_process):
                video_filename, original_file_path_str = video_row['filename'], video_row['original_path']
                original_file_path = Path(original_file_path_str)
                job.update_progress(i, len(videos_to_process), f"Processing {video_filename}")
                if not original_file_path.exists(): logger.warning(f"Original for {video_filename} not found."); failed_count += 1; continue
                if _actually_create_thumbnail(original_file_path, force_creation=True):
                    actual_thumbnail_p = get_thumbnail_path(original_file_path)
                    db_thumb_path = str(actual_thumbnail_p.relative_to(BASE_DIR)) if actual_thumbnail_p.exists() else None
                    if db_thumb_path:
                        try: cursor.execute("UPDATE media_files SET thumbnail_path=?, has_specific_thumbnail=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_thumb_path, video_filename)); generated_count += 1
                        except sqlite3.Error as e: logger.error(f"DB update failed for thumb {video_filename}: {e}"); failed_count += 1
                    else: logger.warning(f"Thumb created for {video_filename} but path issue."); failed_count += 1
                else: logger.error(f"Failed to gen thumb for {video_filename}."); failed_count += 1
            conn.commit()
            job.complete({"message": f"Generated: {generated_count}, Failed: {failed_count}", "generated": generated_count, "failed": failed_count})
    except Exception as e: logger.error(f"Thumbnail gen background task failed: {e}"); job.fail(str(e))

# --- Transcoding Endpoints & Background Tasks --- 
@router.post("/transcode-video/{video_name:path}", name="transcode_specific_video_endpoint")
async def transcode_specific_video_endpoint_route(video_name: str, background_tasks: BackgroundTasks):
    job = create_background_job("transcode_single", f"Transcoding {video_name}")
    background_tasks.add_task(transcode_single_video_background_task, job.job_id, video_name, None)
    return {"message": f"Transcoding {video_name} started", "job_id": job.job_id}

@router.post("/transcode-video-advanced/{video_name:path}", name="transcode_specific_video_advanced_endpoint")
async def transcode_specific_video_advanced_endpoint_route(
    video_name: str, background_tasks: BackgroundTasks,
    resolution: str = Form("720p"), quality_mode: str = Form("crf"), crf: str = Form("23"), 
    video_bitrate: str = Form("2M"), audio_bitrate: str = Form("128k"), 
    preset: str = Form("medium"), profile: str = Form("high")
):
    options = {
        "resolution": resolution, "quality_mode": quality_mode, "crf": int(crf), 
        "video_bitrate": video_bitrate, "audio_bitrate": audio_bitrate, 
        "preset": preset, "profile": profile }
    active_options = {k: v for k, v in options.items() if v is not None and (k!='crf' or quality_mode=='crf') and (k!='video_bitrate' or quality_mode=='bitrate')}
    job = create_background_job("transcode_single_advanced", f"Advanced transcoding {video_name}")
    background_tasks.add_task(transcode_single_video_background_task, job.job_id, video_name, active_options)
    return {"message": f"Advanced transcoding for {video_name} started", "job_id": job.job_id, "options": active_options}

def transcode_single_video_background_task(job_id: str, video_name: str, options: Optional[dict] = None):
    job = get_background_job(job_id)
    if not job: return
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT original_path FROM media_files WHERE filename = ? AND media_type = 'video'", (video_name,))
            db_row = cursor.fetchone()
            if not db_row: job.fail(f"Video '{video_name}' not found"); return
            original_file_path = Path(db_row['original_path'])
            if not original_file_path.exists(): job.fail(f"Original for '{video_name}' not found"); return
            job.update_progress(0, 1, "Starting transcoding...")
            output_path = TRANSCODED_DIR / f"{slugify_for_id(original_file_path.stem)}.mp4"
            if output_path.exists():
                job.update_progress(1, 1, "File already exists, updating DB...")
                db_path = str(output_path.relative_to(BASE_DIR))
                try: cursor.execute("UPDATE media_files SET transcoded_path=?, has_transcoded_version=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_path, video_name)); conn.commit(); job.complete({"message": "Already exists", "output_path": db_path}); return
                except sqlite3.Error as e: job.fail(f"DB update failed: {e}"); return
            job.update_progress(1, 1, f"Transcoding {video_name}...")
            if transcode_video(original_file_path, output_path, options=options):
                db_path = str(output_path.relative_to(BASE_DIR)) if output_path.exists() else None
                if db_path: 
                    try: cursor.execute("UPDATE media_files SET transcoded_path=?, has_transcoded_version=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_path, video_name)); conn.commit(); job.complete({"message": "Success", "output_path": db_path})
                    except sqlite3.Error as e: job.fail(f"DB update failed post-transcode: {e}")
                else: job.fail("Transcode complete, output not found")
            else: job.fail(f"FFmpeg transcode failed for {video_name}")
    except Exception as e: logger.error(f"Single video transcode task failed: {e}"); job.fail(str(e))

@router.post("/transcode-all-videos")
async def transcode_all_videos_endpoint_route(background_tasks: BackgroundTasks):
    job = create_background_job("transcode_all", "Transcoding all videos")
    background_tasks.add_task(bulk_transcode_background_task, job.job_id, None)
    return {"message": "Transcoding all videos started", "job_id": job.job_id}

@router.post("/transcode-all-videos-with-options")
async def transcode_all_videos_with_options_endpoint_route(background_tasks: BackgroundTasks, resolution: str = Form("720p"), quality_mode: str = Form("crf"), crf: str = Form("23"), video_bitrate: str = Form("2M"), audio_bitrate: str = Form("128k"), preset: str = Form("medium")):
    options = {"resolution": resolution, "quality_mode": quality_mode, "crf": int(crf) if quality_mode == 'crf' else None, "video_bitrate": video_bitrate if quality_mode == 'bitrate' else None, "audio_bitrate": audio_bitrate, "preset": preset}
    filtered_options = {k: v for k, v in options.items() if v is not None}
    job = create_background_job("transcode_all_options", f"Transcoding all videos with options")
    background_tasks.add_task(bulk_transcode_background_task, job.job_id, filtered_options)
    return {"message": "Transcoding all with options started", "job_id": job.job_id, "options": filtered_options}

def bulk_transcode_background_task(job_id: str, options: Optional[dict] = None):
    job = get_background_job(job_id)
    if not job: return
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename, original_path FROM media_files WHERE media_type='video' AND (has_transcoded_version=FALSE OR transcoded_path IS NULL)")
            videos = cursor.fetchall()
            if not videos: job.complete({"message": "No videos need transcoding.", "transcoded": 0, "failed": 0, "skipped": 0}); return
            job.update_progress(0, len(videos), "Starting...")
            count, failed, skipped = 0,0,0
            for i, video_row in enumerate(videos):
                orig_path_str, filename = video_row['original_path'], video_row['filename']
                orig_path = Path(orig_path_str)
                job.update_progress(i, len(videos), f"Processing {filename}")
                if not orig_path.exists(): logger.warning(f"Original for {filename} not found."); failed += 1; continue
                out_path = TRANSCODED_DIR / f"{slugify_for_id(orig_path.stem)}.mp4"
                if out_path.exists():
                    db_path_str = str(out_path.relative_to(BASE_DIR))
                    try: 
                        cursor.execute("UPDATE media_files SET transcoded_path=?, has_transcoded_version=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=? AND (has_transcoded_version=FALSE OR transcoded_path IS NULL)", (db_path_str, filename)); conn.commit()
                        if cursor.rowcount > 0: logger.info(f"DB updated for existing transcode: {filename}")
                        skipped += 1; continue
                    except sqlite3.Error as e: logger.error(f"DB update for existing transcode {filename} failed: {e}"); failed += 1; continue
                if transcode_video(orig_path, out_path, options=options):
                    db_path = str(out_path.relative_to(BASE_DIR)) if out_path.exists() else None
                    if db_path: 
                        try: cursor.execute("UPDATE media_files SET transcoded_path=?, has_transcoded_version=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_path, filename)); conn.commit(); count += 1
                        except sqlite3.Error as e: logger.error(f"DB update for {filename} failed: {e}"); failed += 1
                    else: logger.error(f"Transcode {filename} success, output path issue."); failed += 1
                else: logger.error(f"Failed to transcode {filename}."); failed += 1
            job.complete({"message": f"Transcoded: {count}, Failed: {failed}, Skipped: {skipped}", "transcoded": count, "failed": failed, "skipped": skipped})
    except Exception as e: logger.error(f"Bulk transcode task failed: {e}"); job.fail(str(e))

# --- Preview Endpoints & Background Tasks ---
@router.post("/generate-preview/{video_name:path}", name="generate_specific_preview_endpoint")
async def generate_specific_preview_endpoint_route(video_name: str, response: Response, request: Request):
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
            url = f"/{db_path.replace('\\', '/')}" if db_path else None 
            if db_path:
                try: 
                    cursor.execute("UPDATE media_files SET preview_path=?, has_preview=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_path, video_name)); conn.commit()
                    response.headers["X-Preview-Url"] = url 
                    logger.info(f"Preview for {video_name} generated and DB updated.")
                except sqlite3.Error as e: 
                    logger.error(f"DB update for preview {video_name} failed: {e}")
                    # Error handling omitted for brevity, should mirror thumbnail endpoint
                    raise HTTPException(status_code=500, detail="DB update failed after preview generation.")
            else: logger.error(f"Preview for {video_name} created, path issue."); raise HTTPException(status_code=500, detail="Path issue after preview generation.")
            updated_video_details, _, _ = get_single_video_details_from_db(video_name)
            return templates.TemplateResponse("_video_metadata_sidebar.html", {"request": request, "video": updated_video_details})
        else: 
            # Error handling omitted for brevity
            raise HTTPException(status_code=500, detail="Failed to generate preview.")

@router.post("/generate-all-previews")
async def generate_all_previews_endpoint_route(background_tasks: BackgroundTasks):
    job = create_background_job("previews_all", "Generating all hover previews")
    background_tasks.add_task(generate_all_previews_background_task, job.job_id)
    return {"message": "Preview generation for all videos started", "job_id": job.job_id}

def generate_all_previews_background_task(job_id: str):
    job = get_background_job(job_id)
    if not job: return
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename, original_path FROM media_files WHERE media_type='video' AND (has_preview=FALSE OR preview_path IS NULL)")
            videos = cursor.fetchall()
            if not videos: job.complete({"message": "No previews to generate.", "generated": 0, "failed": 0, "skipped": 0}); return
            job.update_progress(0, len(videos), "Starting...")
            gen, failed, skipped = 0,0,0
            for i, video_row in enumerate(videos):
                filename, orig_path_str = video_row['filename'], video_row['original_path']
                orig_path = Path(orig_path_str)
                job.update_progress(i, len(videos), f"Processing {filename}")
                if not orig_path.exists(): logger.warning(f"Original for {filename} not found."); failed += 1; continue
                preview_p = get_preview_path(orig_path)
                if preview_p.exists(): 
                    db_path = str(preview_p.relative_to(BASE_DIR))
                    try: 
                        cursor.execute("UPDATE media_files SET preview_path=?, has_preview=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=? AND (has_preview=FALSE OR preview_path IS NULL)", (db_path, filename)); conn.commit()
                        if cursor.rowcount > 0: logger.info(f"DB updated for existing preview: {filename}")
                        skipped += 1; continue
                    except sqlite3.Error as e: logger.error(f"DB update for existing preview {filename} failed: {e}"); failed += 1; continue
                if create_hover_preview(orig_path, preview_p):
                    db_path_str = str(preview_p.relative_to(BASE_DIR)) if preview_p.exists() else None
                    if db_path_str:
                        try: cursor.execute("UPDATE media_files SET preview_path=?, has_preview=TRUE, last_scanned=CURRENT_TIMESTAMP WHERE filename=?", (db_path_str, filename)); conn.commit(); gen += 1
                        except sqlite3.Error as e: logger.error(f"DB update for {filename} preview failed: {e}"); failed += 1
                    else: logger.error(f"Preview gen {filename} success, output path issue."); failed += 1
                else: logger.error(f"Failed to gen preview for {filename}."); failed += 1
            job.complete({"message": f"Generated: {gen}, Failed: {failed}, Skipped: {skipped}", "generated": gen, "failed": failed, "skipped": skipped})
    except Exception as e: logger.error(f"Preview gen background task failed: {e}"); job.fail(str(e))

# --- Media Scanning Endpoint & Background Task ---
@router.post("/scan-media-directory", response_class=HTMLResponse)
async def scan_media_directory_endpoint_route(request: Request, background_tasks: BackgroundTasks):
    job = create_background_job("media_scan", "Scanning media directory and updating database")
    background_tasks.add_task(run_scan_media_directory_background_task, job.job_id) 
    return templates.TemplateResponse("_scan_status_message.html", {
        "request": request, "message": "Media directory scan started.",
        "job_id": job.job_id, "job_status_url": f"/jobs/{job.job_id}" 
    })

def run_scan_media_directory_background_task(job_id: str):
    job = get_background_job(job_id)
    if not job: logger.error(f"Job {job_id} not found for media scan."); return
    try:
        job.update_progress(0, 1, "Starting media scan...")
        scan_media_directory_and_update_db() # Imported from data_access
        job.update_progress(1, 1, "Scan completed.")
        job.complete({"message": "Media directory scan and database update finished successfully."})
    except Exception as e: logger.error(f"Media scan task (job {job_id}) failed: {e}"); job.fail(str(e))

# --- Delete Asset Endpoints ---
@router.post("/delete-thumbnail/{video_name:path}", name="delete_specific_thumbnail_endpoint")
async def delete_specific_thumbnail_route(video_name: str, request: Request):
    error_message = None
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, thumbnail_path FROM media_files WHERE filename = ? AND media_type = 'video'", (video_name,))
        db_row = cursor.fetchone(); video_id, thumbnail_path_str = (db_row['id'], db_row['thumbnail_path']) if db_row else (None, None)
        if not video_id: raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found.")
        if thumbnail_path_str:
            thumbnail_path = BASE_DIR / thumbnail_path_str
            if thumbnail_path.exists():
                try: os.remove(thumbnail_path); logger.info(f"Deleted thumbnail: {thumbnail_path}")
                except OSError as e: logger.error(f"Failed to delete thumb file {thumbnail_path}: {e}"); error_message = str(e)
            try: cursor.execute("UPDATE media_files SET thumbnail_path = NULL, has_specific_thumbnail = FALSE, last_scanned=CURRENT_TIMESTAMP WHERE id = ?", (video_id,)); conn.commit()
            except sqlite3.Error as e: error_message = (error_message or "") + f" DB update failed: {e}"
    updated_video_details, _, _ = get_single_video_details_from_db(video_name)
    context = {"request": request, "video": updated_video_details}
    if error_message: context["error_message"] = error_message
    return templates.TemplateResponse("_video_metadata_sidebar.html", context)

@router.post("/delete-transcoded/{video_name:path}", name="delete_specific_transcoded_version_endpoint")
async def delete_specific_transcoded_route(video_name: str, request: Request):
    error_message = None
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, transcoded_path FROM media_files WHERE filename = ? AND media_type = 'video'", (video_name,))
        db_row = cursor.fetchone(); video_id, transcoded_path_str = (db_row['id'], db_row['transcoded_path']) if db_row else (None, None)
        if not video_id: raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found.")
        if transcoded_path_str:
            transcoded_path = BASE_DIR / transcoded_path_str
            if transcoded_path.exists():
                try: os.remove(transcoded_path); logger.info(f"Deleted transcoded: {transcoded_path}")
                except OSError as e: logger.error(f"Failed to delete transcoded file {transcoded_path}: {e}"); error_message = str(e)
            try: cursor.execute("UPDATE media_files SET transcoded_path = NULL, has_transcoded_version = FALSE, last_scanned=CURRENT_TIMESTAMP WHERE id = ?", (video_id,)); conn.commit()
            except sqlite3.Error as e: error_message = (error_message or "") + f" DB update failed: {e}"
    updated_video_details, _, _ = get_single_video_details_from_db(video_name)
    context = {"request": request, "video": updated_video_details}
    if error_message: context["error_message"] = error_message
    return templates.TemplateResponse("_video_metadata_sidebar.html", context)

@router.post("/delete-preview/{video_name:path}", name="delete_specific_preview_endpoint")
async def delete_specific_preview_route(video_name: str, request: Request):
    error_message = None
    with db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, preview_path FROM media_files WHERE filename = ? AND media_type = 'video'", (video_name,))
        db_row = cursor.fetchone(); video_id, preview_path_str = (db_row['id'], db_row['preview_path']) if db_row else (None, None)
        if not video_id: raise HTTPException(status_code=404, detail=f"Video '{video_name}' not found.")
        if preview_path_str:
            preview_path_file = BASE_DIR / preview_path_str # Renamed to avoid conflict with get_preview_path function
            if preview_path_file.exists():
                try: os.remove(preview_path_file); logger.info(f"Deleted preview: {preview_path_file}")
                except OSError as e: logger.error(f"Failed to delete preview file {preview_path_file}: {e}"); error_message = str(e)
            try: cursor.execute("UPDATE media_files SET preview_path = NULL, has_preview = FALSE, last_scanned=CURRENT_TIMESTAMP WHERE id = ?", (video_id,)); conn.commit()
            except sqlite3.Error as e: error_message = (error_message or "") + f" DB update failed: {e}"
    updated_video_details, _, _ = get_single_video_details_from_db(video_name)
    context = {"request": request, "video": updated_video_details}
    if error_message: context["error_message"] = error_message
    return templates.TemplateResponse("_video_metadata_sidebar.html", context)

# --- ML Analysis Endpoints & Background Tasks ---
@router.post("/process-all-videos-ml-analysis")
async def process_all_videos_ml_analysis_endpoint(
    background_tasks: BackgroundTasks,
    frame_interval: int = Form(100),
    model_name: str = Form("clip-ViT-B-32"),
    skip_existing: bool = Form(True)
):
    """Start ML analysis processing for all videos"""
    try:
        from ml_processing import process_all_videos_for_ml_analysis
        
        # Create background job
        job = create_background_job(
            "ml_analysis_all",
            f"ML Analysis for all videos (every {frame_interval} frames, model: {model_name})"
        )
        
        # Start processing in background
        background_tasks.add_task(
            process_all_videos_for_ml_analysis,
            job,
            frame_interval,
            model_name,
            skip_existing
        )
        
        return {
            "job_id": job.job_id,
            "message": f"Started ML analysis for all videos (frame interval: {frame_interval}, model: {model_name})"
        }
        
    except Exception as e:
        logger.error(f"Error starting bulk ML analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 