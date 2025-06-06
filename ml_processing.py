import subprocess
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import tempfile
import shutil

from config import BASE_DIR, MEDIA_DIR, logger
from jobs_manager import BackgroundJob
from chroma_db import chroma_manager

# ML Analysis directory
ML_ANALYSIS_DIR = BASE_DIR / "ml_analysis"
ML_ANALYSIS_DIR.mkdir(exist_ok=True)

def process_video_frames_with_embeddings(
    job: BackgroundJob,
    video: Dict[str, Any],
    frame_interval: int = 100,
    model_name: str = "clip-ViT-B-32"
):
    """
    Process video frames and generate embeddings using sentence transformers
    """
    try:
        job.update_progress(0, 100, "Initializing frame processing...")
        
        # Setup paths
        video_path = MEDIA_DIR / video['name']
        video_id = video['id_db']
        analysis_dir = ML_ANALYSIS_DIR / f"video_{video_id}"
        analysis_dir.mkdir(exist_ok=True)
        frames_dir = analysis_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        logger.info(f"Starting frame processing for video {video['name']} (ID: {video_id})")
        
        # Step 1: Extract frames using FFmpeg
        job.update_progress(10, 100, "Extracting frames from video...")
        frame_files = extract_frames_with_ffmpeg(video_path, frames_dir, frame_interval)
        
        if not frame_files:
            job.fail("No frames were extracted from the video")
            return
        
        logger.info(f"Extracted {len(frame_files)} frames")
        
        # Step 2: Generate embeddings
        job.update_progress(30, 100, f"Generating embeddings for {len(frame_files)} frames...")
        embeddings_data = generate_frame_embeddings(frame_files, model_name, job)
        
        # Step 3: Save results
        job.update_progress(90, 100, "Saving analysis results...")
        
        # Prepare final data structure
        analysis_data = {
            "video_id": video_id,
            "video_filename": video['name'],
            "video_title": video['display_title'],
            "frame_interval": frame_interval,
            "model_name": model_name,
            "processed_at": datetime.now().isoformat(),
            "total_frames": len(frame_files),
            "frames": embeddings_data
        }
        
        # Save embeddings data to JSON (for backward compatibility)
        embeddings_file = analysis_dir / "embeddings.json"
        with open(embeddings_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        # Store embeddings in ChromaDB
        job.update_progress(95, 100, "Storing embeddings in vector database...")
        chroma_success = chroma_manager.store_video_embeddings(video_id, analysis_data)
        if chroma_success:
            logger.info(f"Successfully stored embeddings in ChromaDB for video {video_id}")
        else:
            logger.warning(f"Failed to store embeddings in ChromaDB for video {video_id}")
        
        # Save metadata
        metadata_file = analysis_dir / "metadata.json"
        metadata = {
            "video_id": video_id,
            "video_filename": video['name'],
            "processed_at": datetime.now().isoformat(),
            "frame_count": len(frame_files),
            "model_name": model_name,
            "frame_interval": frame_interval,
            "chroma_stored": chroma_success
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        job.update_progress(100, 100, "Frame processing completed successfully")
        job.complete({
            "message": f"Successfully processed {len(frame_files)} frames",
            "frame_count": len(frame_files),
            "analysis_dir": str(analysis_dir),
            "video_id": video_id
        })
        
        logger.info(f"Frame processing completed for video {video['name']}")
        
    except Exception as e:
        error_msg = f"Error processing video frames: {str(e)}"
        logger.error(error_msg)
        job.fail(error_msg)

def extract_frames_with_ffmpeg(video_path: Path, output_dir: Path, frame_interval: int) -> List[Path]:
    """
    Extract frames from video using FFmpeg at specified intervals
    """
    try:
        # Clear existing frames
        for existing_frame in output_dir.glob("*.jpg"):
            existing_frame.unlink()
        
        # FFmpeg command to extract every Nth frame
        # Using select filter to get every frame_interval-th frame
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vf', f'select=not(mod(n\\,{frame_interval}))',
            '-vsync', 'vfr',
            '-q:v', '2',  # High quality JPEG
            str(output_dir / 'frame_%06d.jpg')
        ]
        
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        
        # Run FFmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise Exception(f"FFmpeg failed: {result.stderr}")
        
        # Get list of extracted frames
        frame_files = sorted(list(output_dir.glob("frame_*.jpg")))
        logger.info(f"Successfully extracted {len(frame_files)} frames")
        
        return frame_files
        
    except subprocess.TimeoutExpired:
        raise Exception("FFmpeg frame extraction timed out")
    except Exception as e:
        logger.error(f"Error extracting frames: {e}")
        raise

def generate_frame_embeddings(frame_files: List[Path], model_name: str, job: BackgroundJob) -> List[Dict[str, Any]]:
    """
    Generate embeddings for extracted frames using sentence transformers
    """
    try:
        # Import sentence transformers (lazy import to avoid startup issues if not installed)
        try:
            from sentence_transformers import SentenceTransformer
            from PIL import Image
            import torch
        except ImportError as e:
            raise Exception(f"Required ML libraries not installed: {e}. Please install: pip install sentence-transformers torch pillow")
        
        # Load the model
        logger.info(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        
        embeddings_data = []
        total_frames = len(frame_files)
        
        for i, frame_path in enumerate(frame_files):
            try:
                # Update progress
                progress = 30 + int((i / total_frames) * 50)  # 30-80% of total progress
                job.update_progress(progress, 100, f"Processing frame {i+1}/{total_frames}: {frame_path.name}")
                
                # Load and process image
                image = Image.open(frame_path)
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Generate embedding
                embedding = model.encode(image)
                
                # Convert numpy array to list for JSON serialization
                if hasattr(embedding, 'tolist'):
                    embedding_list = embedding.tolist()
                else:
                    embedding_list = list(embedding)
                
                # Extract frame number from filename
                frame_number = int(frame_path.stem.split('_')[1])
                
                frame_data = {
                    "frame_number": frame_number,
                    "filename": frame_path.name,
                    "frame_path": str(frame_path.relative_to(BASE_DIR)),  # Store relative path for ChromaDB
                    "embedding": embedding_list,
                    "embedding_dimension": len(embedding_list),
                    "timestamp": frame_number * (1/30)  # Assuming 30fps, this is approximate
                }
                
                embeddings_data.append(frame_data)
                
            except Exception as e:
                logger.warning(f"Error processing frame {frame_path}: {e}")
                continue
        
        logger.info(f"Generated embeddings for {len(embeddings_data)} frames")
        return embeddings_data
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

def get_video_analysis_summary(video_id: int) -> Dict[str, Any]:
    """
    Get summary of analysis for a video
    """
    try:
        analysis_dir = ML_ANALYSIS_DIR / f"video_{video_id}"
        metadata_file = analysis_dir / "metadata.json"
        
        if not metadata_file.exists():
            return {"has_analysis": False}
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return {
            "has_analysis": True,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error getting analysis summary for video {video_id}: {e}")
        return {"has_analysis": False, "error": str(e)}

def cleanup_video_analysis(video_id: int) -> bool:
    """
    Clean up analysis data for a video (both file system and ChromaDB)
    """
    try:
        success = True
        
        # Clean up ChromaDB embeddings
        chroma_success = chroma_manager.delete_video_embeddings(video_id)
        if chroma_success:
            logger.info(f"Cleaned up ChromaDB embeddings for video {video_id}")
        else:
            logger.warning(f"Failed to clean up ChromaDB embeddings for video {video_id}")
            success = False
        
        # Clean up file system
        analysis_dir = ML_ANALYSIS_DIR / f"video_{video_id}"
        if analysis_dir.exists():
            shutil.rmtree(analysis_dir)
            logger.info(f"Cleaned up file system analysis data for video {video_id}")
        
        return success
    except Exception as e:
        logger.error(f"Error cleaning up analysis for video {video_id}: {e}")
        return False

def has_ml_analysis(video_id: int) -> bool:
    """
    Check if a video has ML analysis data
    """
    try:
        analysis_dir = ML_ANALYSIS_DIR / f"video_{video_id}"
        embeddings_file = analysis_dir / "embeddings.json"
        return embeddings_file.exists()
    except Exception as e:
        logger.error(f"Error checking ML analysis for video {video_id}: {e}")
        return False

def get_ml_analysis_info(video_id: int) -> Dict[str, Any]:
    """
    Get ML analysis information for a video (checks both ChromaDB and file system)
    """
    try:
        # First check ChromaDB
        chroma_data = chroma_manager.get_video_embeddings(video_id)
        if chroma_data:
            return {
                "has_analysis": True,
                "frame_count": chroma_data.get("frame_count", 0),
                "model_name": chroma_data.get("model_name", "unknown"),
                "processed_at": chroma_data.get("processed_at", "unknown"),
                "source": "chromadb"
            }
        
        # Fallback to file system
        analysis_dir = ML_ANALYSIS_DIR / f"video_{video_id}"
        embeddings_file = analysis_dir / "embeddings.json"
        metadata_file = analysis_dir / "metadata.json"
        
        if not embeddings_file.exists():
            return {"has_analysis": False}
        
        # Get basic info from metadata if available
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return {
                "has_analysis": True,
                "frame_count": metadata.get("frame_count", 0),
                "model_name": metadata.get("model_name", "unknown"),
                "processed_at": metadata.get("processed_at", "unknown"),
                "source": "filesystem"
            }
        else:
            # Fallback to embeddings file
            with open(embeddings_file, 'r') as f:
                analysis_data = json.load(f)
            return {
                "has_analysis": True,
                "frame_count": len(analysis_data.get("frames", [])),
                "model_name": analysis_data.get("model_name", "unknown"),
                "processed_at": analysis_data.get("processed_at", "unknown"),
                "source": "filesystem"
            }
    except Exception as e:
        logger.error(f"Error getting ML analysis info for video {video_id}: {e}")
        return {"has_analysis": False}

def find_similar_frames(query_embedding: List[float], n_results: int = 10, 
                       video_id: int = None) -> List[Dict[str, Any]]:
    """
    Find similar frames using ChromaDB vector similarity search
    
    Args:
        query_embedding: The embedding vector to search for
        n_results: Number of similar results to return
        video_id: Optional video ID to limit search to specific video
        
    Returns:
        List of similar frames with metadata and similarity scores
    """
    try:
        if not chroma_manager.is_available():
            logger.error("ChromaDB not available for similarity search")
            return []
        
        similar_frames = chroma_manager.find_similar_frames(
            query_embedding=query_embedding,
            n_results=n_results,
            video_id=video_id
        )
        
        logger.info(f"Found {len(similar_frames)} similar frames")
        return similar_frames
        
    except Exception as e:
        logger.error(f"Error finding similar frames: {e}")
        return []

def get_video_embeddings_from_chroma(video_id: int) -> Dict[str, Any]:
    """
    Get video embeddings from ChromaDB
    
    Args:
        video_id: Database ID of the video
        
    Returns:
        Dictionary containing embeddings data or None if not found
    """
    try:
        return chroma_manager.get_video_embeddings(video_id)
    except Exception as e:
        logger.error(f"Error getting video embeddings from ChromaDB: {e}")
        return None

def get_chroma_collection_stats() -> Dict[str, Any]:
    """
    Get statistics about the ChromaDB collection
    
    Returns:
        Dictionary containing collection statistics
    """
    try:
        return chroma_manager.get_collection_stats()
    except Exception as e:
        logger.error(f"Error getting ChromaDB collection stats: {e}")
        return {"available": False, "error": str(e)}

def migrate_existing_embeddings_to_chroma() -> Dict[str, Any]:
    """
    Migrate existing JSON embeddings to ChromaDB
    
    Returns:
        Dictionary with migration results
    """
    try:
        migrated_count = 0
        failed_count = 0
        errors = []
        
        # Find all existing analysis directories
        for analysis_dir in ML_ANALYSIS_DIR.glob("video_*"):
            if not analysis_dir.is_dir():
                continue
            
            try:
                video_id = int(analysis_dir.name.split("_")[1])
                embeddings_file = analysis_dir / "embeddings.json"
                
                if not embeddings_file.exists():
                    continue
                
                # Check if already in ChromaDB
                if chroma_manager.get_video_embeddings(video_id):
                    logger.info(f"Video {video_id} embeddings already in ChromaDB, skipping")
                    continue
                
                # Load embeddings data
                with open(embeddings_file, 'r') as f:
                    analysis_data = json.load(f)
                
                # Store in ChromaDB
                success = chroma_manager.store_video_embeddings(video_id, analysis_data)
                if success:
                    migrated_count += 1
                    logger.info(f"Migrated embeddings for video {video_id} to ChromaDB")
                else:
                    failed_count += 1
                    errors.append(f"Failed to migrate video {video_id}")
                    
            except Exception as e:
                failed_count += 1
                error_msg = f"Error migrating {analysis_dir.name}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        result = {
            "migrated_count": migrated_count,
            "failed_count": failed_count,
            "errors": errors,
            "success": failed_count == 0
        }
        
        logger.info(f"Migration completed: {migrated_count} migrated, {failed_count} failed")
        return result
        
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        return {
            "migrated_count": 0,
            "failed_count": 0,
            "errors": [str(e)],
            "success": False
        }

def semantic_search_videos(query_text: str, n_results: int = 20, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    Search for videos using natural language queries against frame embeddings
    
    Args:
        query_text: Natural language search query
        n_results: Maximum number of results to return
        similarity_threshold: Minimum similarity score (0-1) to include results
        
    Returns:
        List of videos with matching frames and similarity scores
    """
    try:
        logger.info(f"🔍 Starting semantic search: query='{query_text}', n_results={n_results}, threshold={similarity_threshold}")
        
        # Check ChromaDB availability
        if not chroma_manager.is_available():
            logger.error("❌ ChromaDB not available for semantic search")
            return []
        
        # Get ChromaDB stats for debugging
        stats = chroma_manager.get_collection_stats()
        logger.info(f"📊 ChromaDB Stats: {stats}")
        
        if stats.get("total_frames", 0) == 0:
            logger.warning("⚠️ No frame embeddings found in ChromaDB. Videos need to be processed with ML analysis first.")
            return []
        
        # Import sentence transformers for text encoding
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            logger.error(f"❌ Sentence transformers not available: {e}")
            return []
        
        # Load a CLIP model for text encoding (should match the model used for frame embeddings)
        # We'll use a default model, but ideally this should match the model used for the frames
        model_name = "clip-ViT-B-32"  # Default model
        
        # Try to get the most commonly used model from ChromaDB stats
        try:
            if stats.get("available") and stats.get("video_ids"):
                # Get a sample of metadata to determine the most common model
                sample_results = chroma_manager.collection.get(
                    limit=10,
                    include=["metadatas"]
                )
                if sample_results.get("metadatas"):
                    # Use the model from the first available metadata
                    detected_model = sample_results["metadatas"][0].get("model_name", "clip-ViT-B-32")
                    if detected_model:
                        model_name = detected_model
                        logger.info(f"🎯 Detected model from ChromaDB: {model_name}")
        except Exception as e:
            logger.warning(f"⚠️ Could not determine model from ChromaDB, using default: {e}")
        
        logger.info(f"🤖 Loading model: {model_name}")
        try:
            model = SentenceTransformer(model_name)
            logger.info(f"✅ Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            return []
        
        # Generate embedding for the search query
        logger.info(f"🔤 Encoding query text: '{query_text}'")
        try:
            query_embedding = model.encode(query_text).tolist()
            logger.info(f"✅ Query encoded to {len(query_embedding)}-dimensional embedding")
        except Exception as e:
            logger.error(f"❌ Failed to encode query: {e}")
            return []
        
        # Search for similar frames
        logger.info(f"🔍 Searching for similar frames (requesting {n_results * 5} frames for aggregation)")
        similar_frames = chroma_manager.find_similar_frames(
            query_embedding=query_embedding,
            n_results=n_results * 5,  # Get more frames to aggregate by video
            video_id=None  # Search across all videos
        )
        
        logger.info(f"📊 Found {len(similar_frames)} similar frames from ChromaDB")
        
        # Group results by video and calculate video-level scores
        video_scores = {}
        video_frame_counts = {}
        frames_above_threshold = 0
        
        logger.info(f"🎯 Filtering frames with similarity >= {similarity_threshold}")
        
        for frame in similar_frames:
            video_id = frame["video_id"]
            similarity_score = frame["similarity_score"]
            
            logger.debug(f"  Frame {frame['frame_number']} from video {video_id}: similarity={similarity_score:.3f}")
            
            # Only include frames above the similarity threshold
            if similarity_score >= similarity_threshold:
                frames_above_threshold += 1
                if video_id not in video_scores:
                    video_scores[video_id] = []
                    video_frame_counts[video_id] = 0
                
                video_scores[video_id].append(similarity_score)
                video_frame_counts[video_id] += 1
        
        logger.info(f"📊 Frames above threshold: {frames_above_threshold}/{len(similar_frames)}")
        logger.info(f"📊 Videos with matching frames: {len(video_scores)}")
        
        # Calculate aggregated scores for each video and find best matching frame
        video_results = []
        video_best_frames = {}  # Store best frame info for each video
        
        # First pass: collect all frame data for each video
        for frame in similar_frames:
            video_id = frame["video_id"]
            similarity_score = frame["similarity_score"]
            
            if similarity_score >= similarity_threshold:
                if video_id not in video_best_frames:
                    video_best_frames[video_id] = []
                
                video_best_frames[video_id].append({
                    "similarity_score": similarity_score,
                    "timestamp": frame["timestamp"],
                    "frame_number": frame["frame_number"]
                })
        
        for video_id, scores in video_scores.items():
            if not scores:
                continue
            
            # Calculate various score metrics
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
            
            # Weighted score: combines max score with frame count and average
            # Higher weight for videos with more matching frames
            weighted_score = (max_score * 0.6) + (avg_score * 0.3) + (min(frame_count / 10, 1.0) * 0.1)
            
            logger.debug(f"  Video {video_id}: {frame_count} frames, max={max_score:.3f}, avg={avg_score:.3f}, weighted={weighted_score:.3f}")
            if best_frame:
                logger.debug(f"    Best frame: timestamp={best_frame['timestamp']:.1f}s, frame={best_frame['frame_number']}")
            
            result_data = {
                "video_id": video_id,
                "max_similarity": max_score,
                "avg_similarity": avg_score,
                "weighted_score": weighted_score,
                "matching_frames": frame_count,
                "similarity_scores": scores[:5]  # Top 5 frame scores for reference
            }
            
            # Add best frame timestamp if available
            if best_frame:
                result_data["best_frame_timestamp"] = best_frame["timestamp"]
                result_data["best_frame_number"] = best_frame["frame_number"]
            
            video_results.append(result_data)
        
        logger.info(f"📊 Calculated scores for {len(video_results)} videos")
        
        # Sort by weighted score (descending)
        video_results.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        # Limit results
        video_results = video_results[:n_results]
        logger.info(f"📊 Limited to top {len(video_results)} videos")
        
        # Get video metadata from database
        logger.info(f"🗃️ Fetching video metadata from database")
        from data_access import get_media_files_from_db
        
        # Get all videos to match with our results
        all_videos_result = get_media_files_from_db(
            media_type_filter="video",
            page=1,
            per_page=1000  # Get all videos
        )
        
        logger.info(f"📊 Found {len(all_videos_result['media_files'])} videos in database")
        
        # Create a lookup dictionary for video metadata
        video_lookup = {video['id_db']: video for video in all_videos_result['media_files']}
        
        # Enrich results with video metadata
        enriched_results = []
        missing_videos = []
        
        for result in video_results:
            video_id = result["video_id"]
            video_metadata = video_lookup.get(video_id)
            
            if video_metadata:
                enriched_result = {
                    **video_metadata,  # Include all video metadata
                    "semantic_search_score": result["weighted_score"],
                    "max_similarity": result["max_similarity"],
                    "avg_similarity": result["avg_similarity"],
                    "matching_frames": result["matching_frames"],
                    "search_type": "semantic"
                }
                
                # Add best frame timestamp if available
                if "best_frame_timestamp" in result:
                    enriched_result["best_frame_timestamp"] = result["best_frame_timestamp"]
                if "best_frame_number" in result:
                    enriched_result["best_frame_number"] = result["best_frame_number"]
                
                enriched_results.append(enriched_result)
                logger.debug(f"  ✅ Enriched video {video_id}: {video_metadata.get('display_title', video_metadata.get('name', 'Unknown'))}")
                if "best_frame_timestamp" in result:
                    logger.debug(f"    📍 Best frame timestamp: {result['best_frame_timestamp']:.1f}s")
            else:
                missing_videos.append(video_id)
                logger.warning(f"  ⚠️ Video {video_id} found in ChromaDB but not in database")
        
        if missing_videos:
            logger.warning(f"⚠️ {len(missing_videos)} videos found in ChromaDB but missing from database: {missing_videos}")
        
        logger.info(f"✅ Semantic search for '{query_text}' found {len(enriched_results)} videos")
        return enriched_results
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return []

def process_all_videos_for_ml_analysis(
    job: BackgroundJob,
    frame_interval: int = 100,
    model_name: str = "clip-ViT-B-32",
    skip_existing: bool = True
):
    """
    Process all videos in the media library for ML analysis
    
    Args:
        job: Background job instance for progress tracking
        frame_interval: Interval between frames to extract (default: every 100 frames)
        model_name: ML model to use for embeddings (default: clip-ViT-B-32)
        skip_existing: Whether to skip videos that already have ML analysis
    """
    try:
        job.update_progress(0, 100, "Initializing bulk ML analysis...")
        logger.info("Starting bulk ML analysis for all videos")
        
        # Get all videos from database
        from data_access import get_media_files_from_db
        
        all_videos_result = get_media_files_from_db(
            media_type_filter="video",
            page=1,
            per_page=10000  # Get all videos
        )
        
        all_videos = all_videos_result['media_files']
        logger.info(f"Found {len(all_videos)} videos in database")
        
        if not all_videos:
            job.complete({
                "message": "No videos found in database",
                "processed": 0,
                "skipped": 0,
                "failed": 0
            })
            return
        
        # Filter videos based on skip_existing setting
        videos_to_process = []
        skipped_count = 0
        
        job.update_progress(5, 100, "Checking existing ML analysis...")
        
        for video in all_videos:
            video_id = video['id_db']
            
            if skip_existing and has_ml_analysis(video_id):
                logger.debug(f"Skipping video {video_id} - already has ML analysis")
                skipped_count += 1
                continue
            
            # Check if video file exists
            video_path = MEDIA_DIR / video['name']
            if not video_path.exists():
                logger.warning(f"Video file not found: {video_path}")
                skipped_count += 1
                continue
            
            videos_to_process.append(video)
        
        logger.info(f"Processing {len(videos_to_process)} videos, skipping {skipped_count}")
        
        if not videos_to_process:
            job.complete({
                "message": f"No videos to process. Skipped {skipped_count} videos (already processed or missing files)",
                "processed": 0,
                "skipped": skipped_count,
                "failed": 0
            })
            return
        
        # Process each video
        processed_count = 0
        failed_count = 0
        
        for i, video in enumerate(videos_to_process):
            video_id = video['id_db']
            video_name = video['name']
            display_title = video.get('display_title', video_name)
            
            progress_percent = int(10 + (i / len(videos_to_process)) * 85)  # 10-95%
            job.update_progress(
                progress_percent, 
                100, 
                f"Processing {display_title} ({i+1}/{len(videos_to_process)})"
            )
            
            logger.info(f"Processing video {i+1}/{len(videos_to_process)}: {display_title}")
            
            try:
                # Create a sub-job for this video (we'll track progress internally)
                video_path = MEDIA_DIR / video_name
                analysis_dir = ML_ANALYSIS_DIR / f"video_{video_id}"
                analysis_dir.mkdir(exist_ok=True)
                frames_dir = analysis_dir / "frames"
                frames_dir.mkdir(exist_ok=True)
                
                # Step 1: Extract frames
                logger.debug(f"Extracting frames from {video_name}")
                frame_files = extract_frames_with_ffmpeg(video_path, frames_dir, frame_interval)
                
                if not frame_files:
                    logger.error(f"No frames extracted from {video_name}")
                    failed_count += 1
                    continue
                
                logger.debug(f"Extracted {len(frame_files)} frames from {video_name}")
                
                # Step 2: Generate embeddings (create a minimal job for progress tracking)
                class MinimalJob:
                    def update_progress(self, current, total, message):
                        pass  # We'll handle progress at the video level
                
                minimal_job = MinimalJob()
                embeddings_data = generate_frame_embeddings(frame_files, model_name, minimal_job)
                
                # Step 3: Save results
                analysis_data = {
                    "video_id": video_id,
                    "video_filename": video_name,
                    "video_title": display_title,
                    "frame_interval": frame_interval,
                    "model_name": model_name,
                    "processed_at": datetime.now().isoformat(),
                    "total_frames": len(frame_files),
                    "frames": embeddings_data
                }
                
                # Save embeddings data to JSON
                embeddings_file = analysis_dir / "embeddings.json"
                with open(embeddings_file, 'w') as f:
                    json.dump(analysis_data, f, indent=2)
                
                # Store embeddings in ChromaDB
                chroma_success = chroma_manager.store_video_embeddings(video_id, analysis_data)
                if chroma_success:
                    logger.debug(f"Successfully stored embeddings in ChromaDB for video {video_id}")
                else:
                    logger.warning(f"Failed to store embeddings in ChromaDB for video {video_id}")
                
                # Save metadata
                metadata_file = analysis_dir / "metadata.json"
                metadata = {
                    "video_id": video_id,
                    "video_filename": video_name,
                    "processed_at": datetime.now().isoformat(),
                    "frame_count": len(frame_files),
                    "model_name": model_name,
                    "frame_interval": frame_interval,
                    "chroma_stored": chroma_success
                }
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                processed_count += 1
                logger.info(f"Successfully processed {display_title} - {len(frame_files)} frames")
                
            except Exception as e:
                logger.error(f"Failed to process video {display_title}: {e}")
                failed_count += 1
                continue
        
        # Complete the job
        job.update_progress(100, 100, "ML analysis completed")
        
        result_message = f"Processed {processed_count} videos successfully"
        if failed_count > 0:
            result_message += f", {failed_count} failed"
        if skipped_count > 0:
            result_message += f", {skipped_count} skipped"
        
        job.complete({
            "message": result_message,
            "processed": processed_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "total_videos": len(all_videos),
            "frame_interval": frame_interval,
            "model_name": model_name
        })
        
        logger.info(f"Bulk ML analysis completed: {result_message}")
        
    except Exception as e:
        error_msg = f"Error in bulk ML analysis: {str(e)}"
        logger.error(error_msg)
        job.fail(error_msg) 