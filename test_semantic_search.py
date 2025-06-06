#!/usr/bin/env python3
"""
Diagnostic script to test semantic search functionality
Run this to identify issues with semantic search setup
"""

import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_chromadb():
    """Test ChromaDB availability and content"""
    print("🔍 Testing ChromaDB...")
    try:
        from chroma_db import chroma_manager
        
        # Check if ChromaDB is available
        available = chroma_manager.is_available()
        print(f"  ChromaDB Available: {available}")
        
        if available:
            # Get stats
            stats = chroma_manager.get_collection_stats()
            print(f"  Collection Stats: {stats}")
            
            if stats.get("total_frames", 0) > 0:
                print(f"  ✅ Found {stats['total_frames']} frames from {stats['unique_videos']} videos")
                return True
            else:
                print(f"  ⚠️ No frame embeddings found in ChromaDB")
                return False
        else:
            print(f"  ❌ ChromaDB not available")
            return False
            
    except Exception as e:
        print(f"  ❌ Error testing ChromaDB: {e}")
        return False

def test_sentence_transformers():
    """Test Sentence Transformers availability"""
    print("\n🤖 Testing Sentence Transformers...")
    try:
        from sentence_transformers import SentenceTransformer
        print("  ✅ Sentence Transformers imported successfully")
        
        # Try to load the model
        try:
            model = SentenceTransformer("clip-ViT-B-32")
            print("  ✅ CLIP model loaded successfully")
            
            # Test encoding
            test_text = "a person walking"
            embedding = model.encode(test_text)
            print(f"  ✅ Text encoding successful: {len(embedding)}-dimensional vector")
            return True, model
            
        except Exception as e:
            print(f"  ❌ Error loading CLIP model: {e}")
            return False, None
            
    except ImportError as e:
        print(f"  ❌ Sentence Transformers not available: {e}")
        print("  💡 Install with: pip install sentence-transformers")
        return False, None

def test_video_database():
    """Test video database for ML analysis"""
    print("\n🗃️ Testing Video Database...")
    try:
        from data_access import get_media_files_from_db
        
        # Get all videos
        all_videos = get_media_files_from_db(media_type_filter="video", page=1, per_page=1000)
        total_videos = len(all_videos['media_files'])
        print(f"  Total videos in database: {total_videos}")
        
        # Check for ML analysis
        videos_with_ml = [v for v in all_videos['media_files'] if v.get('has_ml_analysis')]
        print(f"  Videos with ML analysis: {len(videos_with_ml)}")
        
        if videos_with_ml:
            print("  ✅ Found videos with ML analysis:")
            for video in videos_with_ml[:5]:  # Show first 5
                print(f"    - ID {video['id_db']}: {video.get('display_title', video.get('name', 'Unknown'))}")
            if len(videos_with_ml) > 5:
                print(f"    ... and {len(videos_with_ml) - 5} more")
            return True
        else:
            print("  ⚠️ No videos have been processed with ML analysis")
            print("  💡 Go to ML Processing page to analyze videos first")
            return False
            
    except Exception as e:
        print(f"  ❌ Error testing video database: {e}")
        return False

def test_semantic_search():
    """Test the actual semantic search function"""
    print("\n🔍 Testing Semantic Search Function...")
    try:
        from ml_processing import semantic_search_videos
        
        # Test with a simple query
        test_query = "person"
        print(f"  Testing query: '{test_query}'")
        
        results = semantic_search_videos(
            query_text=test_query,
            n_results=5,
            similarity_threshold=0.3  # Low threshold for testing
        )
        
        print(f"  Results found: {len(results)}")
        
        if results:
            print("  ✅ Semantic search working! Sample results:")
            for i, result in enumerate(results[:3]):
                print(f"    {i+1}. {result.get('display_title', result.get('name', 'Unknown'))}")
                print(f"       Score: {result.get('semantic_search_score', 0):.3f}")
                print(f"       Matching frames: {result.get('matching_frames', 0)}")
            return True
        else:
            print("  ⚠️ No results found - this could indicate:")
            print("    - Threshold too high")
            print("    - Query doesn't match video content")
            print("    - Issue with embeddings")
            return False
            
    except Exception as e:
        print(f"  ❌ Error testing semantic search: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🚀 PABT Semantic Search Diagnostic Tool")
    print("=" * 50)
    
    # Run tests
    chromadb_ok = test_chromadb()
    st_ok, model = test_sentence_transformers()
    video_db_ok = test_video_database()
    
    # Only test semantic search if prerequisites are met
    if chromadb_ok and st_ok and video_db_ok:
        search_ok = test_semantic_search()
    else:
        search_ok = False
        print("\n⚠️ Skipping semantic search test due to missing prerequisites")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    print(f"ChromaDB:           {'✅ OK' if chromadb_ok else '❌ FAIL'}")
    print(f"Sentence Transformers: {'✅ OK' if st_ok else '❌ FAIL'}")
    print(f"Video Database:     {'✅ OK' if video_db_ok else '❌ FAIL'}")
    print(f"Semantic Search:    {'✅ OK' if search_ok else '❌ FAIL'}")
    
    if all([chromadb_ok, st_ok, video_db_ok, search_ok]):
        print("\n🎉 All tests passed! Semantic search should work.")
    else:
        print("\n🔧 Issues found. Follow the recommendations above to fix them.")
        
        # Specific recommendations
        print("\n💡 RECOMMENDATIONS:")
        if not chromadb_ok:
            print("  1. Check ChromaDB initialization and permissions")
        if not st_ok:
            print("  2. Install sentence-transformers: pip install sentence-transformers")
        if not video_db_ok:
            print("  3. Process videos with ML analysis first")
        if not search_ok and chromadb_ok and st_ok and video_db_ok:
            print("  4. Try lowering the similarity threshold or different search terms")

if __name__ == "__main__":
    main() 