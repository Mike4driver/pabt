#!/usr/bin/env python3
"""
Debug script to check actual similarity scores
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def debug_similarity_scores():
    """Check what similarity scores we're actually getting"""
    try:
        from chroma_db import chroma_manager
        from sentence_transformers import SentenceTransformer
        
        print("🔍 Debugging Similarity Scores")
        print("=" * 50)
        
        # Load model and encode a test query
        model = SentenceTransformer("clip-ViT-B-32")
        query_text = "person"
        query_embedding = model.encode(query_text).tolist()
        
        print(f"Query: '{query_text}'")
        print(f"Query embedding dimension: {len(query_embedding)}")
        print(f"Query embedding sample: {query_embedding[:5]}...")
        
        # Get raw results from ChromaDB
        results = chroma_manager.collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            include=["embeddings", "metadatas", "documents", "distances"]
        )
        
        print(f"\nRaw ChromaDB Results:")
        print(f"Number of results: {len(results['ids'][0])}")
        
        if results['distances'] and results['distances'][0]:
            distances = results['distances'][0]
            print(f"\nDistance Analysis:")
            print(f"Min distance: {min(distances):.6f}")
            print(f"Max distance: {max(distances):.6f}")
            print(f"Average distance: {sum(distances)/len(distances):.6f}")
            
            print(f"\nSimilarity Analysis (old: 1 - distance):")
            old_similarities = [1 - d for d in distances]
            print(f"Old Min similarity: {min(old_similarities):.6f}")
            print(f"Old Max similarity: {max(old_similarities):.6f}")
            print(f"Old Average similarity: {sum(old_similarities)/len(old_similarities):.6f}")
            
            print(f"\nSimilarity Analysis (new: exponential decay):")
            new_similarities = [max(0.0, 1.0 / (1.0 + d / 100.0)) for d in distances]
            print(f"New Min similarity: {min(new_similarities):.6f}")
            print(f"New Max similarity: {max(new_similarities):.6f}")
            print(f"New Average similarity: {sum(new_similarities)/len(new_similarities):.6f}")
            
            print(f"\nTop 10 Results:")
            for i in range(min(10, len(distances))):
                distance = distances[i]
                old_similarity = 1 - distance
                new_similarity = max(0.0, 1.0 / (1.0 + distance / 100.0))
                video_id = results['metadatas'][0][i]['video_id']
                frame_num = results['metadatas'][0][i]['frame_number']
                print(f"  {i+1}. Video {video_id}, Frame {frame_num}: distance={distance:.6f}")
                print(f"      Old similarity={old_similarity:.6f}, New similarity={new_similarity:.6f}")
            
            # Check if any are above common thresholds
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            print(f"\nThreshold Analysis (new similarities):")
            for threshold in thresholds:
                count = sum(1 for s in new_similarities if s >= threshold)
                print(f"  >= {threshold}: {count}/{len(new_similarities)} results")
        
        # Also check a sample frame embedding for comparison
        print(f"\nSample Frame Embedding Analysis:")
        sample_results = chroma_manager.collection.get(
            limit=1,
            include=["embeddings", "metadatas"]
        )
        
        if sample_results['embeddings'] and len(sample_results['embeddings']) > 0:
            frame_embedding = sample_results['embeddings'][0]
            print(f"Frame embedding dimension: {len(frame_embedding)}")
            print(f"Frame embedding sample: {frame_embedding[:5]}...")
            
            # Calculate manual cosine similarity
            import numpy as np
            
            query_vec = np.array(query_embedding)
            frame_vec = np.array(frame_embedding)
            
            # Normalize vectors
            query_norm = query_vec / np.linalg.norm(query_vec)
            frame_norm = frame_vec / np.linalg.norm(frame_vec)
            
            # Calculate cosine similarity
            cosine_sim = np.dot(query_norm, frame_norm)
            print(f"Manual cosine similarity: {cosine_sim:.6f}")
            
            # Calculate L2 distance (what ChromaDB likely uses)
            l2_distance = np.linalg.norm(query_norm - frame_norm)
            print(f"Manual L2 distance: {l2_distance:.6f}")
            print(f"Manual similarity (1 - L2): {1 - l2_distance:.6f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_similarity_scores() 