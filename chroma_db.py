import chromadb
from chromadb.config import Settings
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

from config import BASE_DIR, logger

# ChromaDB configuration
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
CHROMA_DB_PATH.mkdir(exist_ok=True)

class ChromaDBManager:
    """Manages ChromaDB operations for video frame embeddings"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=str(CHROMA_DB_PATH),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection for video frame embeddings
            self.collection = self.client.get_or_create_collection(
                name="video_frame_embeddings",
                metadata={
                    "description": "Video frame embeddings generated by CLIP models",
                    "created_at": datetime.now().isoformat()
                }
            )
            
            logger.info(f"ChromaDB initialized successfully. Collection count: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.client = None
            self.collection = None
    
    def is_available(self) -> bool:
        """Check if ChromaDB is available and initialized"""
        return self.client is not None and self.collection is not None
    
    def store_video_embeddings(self, video_id: int, embeddings_data: Dict) -> bool:
        """
        Store video frame embeddings in ChromaDB
        
        Args:
            video_id: Database ID of the video
            embeddings_data: Dictionary containing frame embeddings and metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_available():
            logger.error("ChromaDB not available for storing embeddings")
            return False
        
        try:
            frames = embeddings_data.get("frames", [])
            if not frames:
                logger.warning(f"No frames found in embeddings data for video {video_id}")
                return False
            
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for frame in frames:
                frame_id = f"video_{video_id}_frame_{frame['frame_number']}"
                
                # Convert embedding to list if it's a numpy array
                embedding = frame['embedding']
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                # Prepare metadata
                metadata = {
                    "video_id": video_id,
                    "frame_number": frame['frame_number'],
                    "timestamp": frame['timestamp'],
                    "frame_path": frame['frame_path'],
                    "model_name": embeddings_data.get("model_name", "unknown"),
                    "processed_at": embeddings_data.get("processed_at", datetime.now().isoformat()),
                    "embedding_dimension": len(embedding)
                }
                
                # Create a document description for the frame
                document = f"Frame {frame['frame_number']} from video {video_id} at {frame['timestamp']}s"
                
                ids.append(frame_id)
                embeddings.append(embedding)
                metadatas.append(metadata)
                documents.append(document)
            
            # Store in ChromaDB (upsert to handle updates)
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            logger.info(f"Stored {len(frames)} frame embeddings for video {video_id} in ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embeddings for video {video_id} in ChromaDB: {e}")
            return False
    
    def get_video_embeddings(self, video_id: int) -> Optional[Dict]:
        """
        Retrieve all embeddings for a specific video
        
        Args:
            video_id: Database ID of the video
            
        Returns:
            Dict containing embeddings data or None if not found
        """
        if not self.is_available():
            logger.error("ChromaDB not available for retrieving embeddings")
            return None
        
        try:
            # Query for all frames of this video
            results = self.collection.get(
                where={"video_id": video_id},
                include=["embeddings", "metadatas", "documents"]
            )
            
            if not results['ids']:
                return None
            
            # Reconstruct the embeddings data structure
            frames = []
            for i, frame_id in enumerate(results['ids']):
                frame_data = {
                    "frame_number": results['metadatas'][i]['frame_number'],
                    "timestamp": results['metadatas'][i]['timestamp'],
                    "frame_path": results['metadatas'][i]['frame_path'],
                    "embedding": results['embeddings'][i]
                }
                frames.append(frame_data)
            
            # Sort frames by frame number
            frames.sort(key=lambda x: x['frame_number'])
            
            # Get metadata from first frame (should be consistent across all frames)
            first_metadata = results['metadatas'][0]
            
            return {
                "video_id": video_id,
                "model_name": first_metadata.get('model_name', 'unknown'),
                "processed_at": first_metadata.get('processed_at'),
                "frame_count": len(frames),
                "frames": frames
            }
            
        except Exception as e:
            logger.error(f"Error retrieving embeddings for video {video_id} from ChromaDB: {e}")
            return None
    
    def delete_video_embeddings(self, video_id: int) -> bool:
        """
        Delete all embeddings for a specific video
        
        Args:
            video_id: Database ID of the video
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_available():
            logger.error("ChromaDB not available for deleting embeddings")
            return False
        
        try:
            # Get all frame IDs for this video
            results = self.collection.get(
                where={"video_id": video_id},
                include=["metadatas"]
            )
            
            if results['ids']:
                # Delete all frames for this video
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} frame embeddings for video {video_id} from ChromaDB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embeddings for video {video_id} from ChromaDB: {e}")
            return False
    
    def find_similar_frames(self, query_embedding: List[float], n_results: int = 10, 
                          video_id: Optional[int] = None) -> List[Dict]:
        """
        Find similar frames using vector similarity search
        
        Args:
            query_embedding: The embedding vector to search for
            n_results: Number of similar results to return
            video_id: Optional video ID to limit search to specific video
            
        Returns:
            List of similar frames with metadata and distances
        """
        if not self.is_available():
            logger.error("❌ ChromaDB not available for similarity search")
            return []
        
        try:
            logger.info(f"🔍 ChromaDB similarity search: n_results={n_results}, video_filter={video_id}, embedding_dim={len(query_embedding)}")
            
            # Prepare query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["embeddings", "metadatas", "documents", "distances"]
            }
            
            # Add video filter if specified
            if video_id is not None:
                query_params["where"] = {"video_id": video_id}
                logger.info(f"🎯 Filtering to video_id: {video_id}")
            
            # Perform similarity search
            logger.debug(f"🔍 Executing ChromaDB query with params: {query_params}")
            results = self.collection.query(**query_params)
            
            logger.info(f"📊 ChromaDB raw results: ids={len(results.get('ids', [[]])[0])}, distances={len(results.get('distances', [[]])[0])}")
            
            # Format results
            similar_frames = []
            if results['ids'] and results['ids'][0]:  # Check if we have results
                logger.info(f"✅ Processing {len(results['ids'][0])} results from ChromaDB")
                for i in range(len(results['ids'][0])):
                    distance = results['distances'][0][i]
                    
                    # Convert distance to similarity score
                    # ChromaDB uses L2 distance, so we need to convert it properly
                    # For normalized vectors, L2 distance relates to cosine similarity as:
                    # cosine_similarity = 1 - (L2_distance^2 / 2)
                    # But since our embeddings aren't normalized, we'll use a different approach
                    
                    # For large distances (unnormalized embeddings), use exponential decay
                    # This maps large distances to small similarities
                    similarity_score = max(0.0, 1.0 / (1.0 + distance / 100.0))
                    
                    frame_data = {
                        "frame_id": results['ids'][0][i],
                        "video_id": results['metadatas'][0][i]['video_id'],
                        "frame_number": results['metadatas'][0][i]['frame_number'],
                        "timestamp": results['metadatas'][0][i]['timestamp'],
                        "frame_path": results['metadatas'][0][i]['frame_path'],
                        "model_name": results['metadatas'][0][i]['model_name'],
                        "distance": distance,
                        "similarity_score": similarity_score,
                        "document": results['documents'][0][i]
                    }
                    similar_frames.append(frame_data)
                    logger.debug(f"  Frame {frame_data['frame_number']} (video {frame_data['video_id']}): distance={distance:.3f}, similarity={similarity_score:.3f}")
            else:
                logger.warning("⚠️ No results returned from ChromaDB query")
            
            logger.info(f"✅ ChromaDB search completed: {len(similar_frames)} frames found")
            return similar_frames
            
        except Exception as e:
            logger.error(f"❌ Error performing similarity search in ChromaDB: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the ChromaDB collection"""
        if not self.is_available():
            return {"available": False, "error": "ChromaDB not available"}
        
        try:
            count = self.collection.count()
            
            # Get unique video count
            results = self.collection.get(include=["metadatas"])
            unique_videos = set()
            if results['metadatas']:
                unique_videos = set(metadata['video_id'] for metadata in results['metadatas'])
            
            return {
                "available": True,
                "total_frames": count,
                "unique_videos": len(unique_videos),
                "video_ids": sorted(list(unique_videos))
            }
            
        except Exception as e:
            logger.error(f"Error getting ChromaDB collection stats: {e}")
            return {"available": False, "error": str(e)}
    
    def reset_collection(self) -> bool:
        """Reset the entire collection (delete all data)"""
        if not self.is_available():
            logger.error("ChromaDB not available for reset")
            return False
        
        try:
            # Delete the collection
            self.client.delete_collection("video_frame_embeddings")
            
            # Recreate the collection
            self.collection = self.client.get_or_create_collection(
                name="video_frame_embeddings",
                metadata={
                    "description": "Video frame embeddings generated by CLIP models",
                    "created_at": datetime.now().isoformat()
                }
            )
            
            logger.info("ChromaDB collection reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting ChromaDB collection: {e}")
            return False

# Global instance
chroma_manager = ChromaDBManager() 