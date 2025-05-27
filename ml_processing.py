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
        
        # Save embeddings data
        embeddings_file = analysis_dir / "embeddings.json"
        with open(embeddings_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        # Save metadata
        metadata_file = analysis_dir / "metadata.json"
        metadata = {
            "video_id": video_id,
            "video_filename": video['name'],
            "processed_at": datetime.now().isoformat(),
            "frame_count": len(frame_files),
            "model_name": model_name,
            "frame_interval": frame_interval
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
                    "embedding": embedding_list,
                    "embedding_dimension": len(embedding_list),
                    "timestamp_seconds": frame_number * (1/30)  # Assuming 30fps, this is approximate
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
    Clean up analysis data for a video
    """
    try:
        analysis_dir = ML_ANALYSIS_DIR / f"video_{video_id}"
        if analysis_dir.exists():
            shutil.rmtree(analysis_dir)
            logger.info(f"Cleaned up analysis data for video {video_id}")
            return True
        return False
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
    Get ML analysis information for a video
    """
    try:
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
                "processed_at": metadata.get("processed_at", "unknown")
            }
        else:
            # Fallback to embeddings file
            with open(embeddings_file, 'r') as f:
                analysis_data = json.load(f)
            return {
                "has_analysis": True,
                "frame_count": len(analysis_data.get("frames", [])),
                "model_name": analysis_data.get("model_name", "unknown"),
                "processed_at": analysis_data.get("processed_at", "unknown")
            }
    except Exception as e:
        logger.error(f"Error getting ML analysis info for video {video_id}: {e}")
        return {"has_analysis": False} 