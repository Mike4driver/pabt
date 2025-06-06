#!/usr/bin/env python3
"""
Test script for frame search functionality
"""

import tempfile
import subprocess
from pathlib import Path
import base64

def test_frame_extraction():
    """Test frame extraction using FFmpeg"""
    print("🧪 Testing Frame Extraction Functionality")
    print("=" * 50)
    
    # Check if FFmpeg is available
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ FFmpeg is available")
        else:
            print("❌ FFmpeg not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ FFmpeg not found. Please install FFmpeg.")
        return False
    
    # Check if we have any video files in the media directory
    from config import MEDIA_DIR
    video_files = []
    for ext in ['.mp4', '.avi', '.mkv', '.mov', '.wmv']:
        video_files.extend(MEDIA_DIR.glob(f"*{ext}"))
        video_files.extend(MEDIA_DIR.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print("❌ No video files found in media directory")
        return False
    
    test_video = video_files[0]
    print(f"📹 Testing with video: {test_video.name}")
    
    # Get video duration first
    try:
        duration_cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', str(test_video)
        ]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=10)
        if duration_result.returncode == 0:
            duration = float(duration_result.stdout.strip())
            print(f"📏 Video duration: {duration:.1f} seconds")
        else:
            duration = 60  # Default fallback
    except:
        duration = 60  # Default fallback
    
    # Test frame extraction at appropriate timestamps
    if duration > 60:
        timestamps = [10, 30, 60]  # Extract frames at 10s, 30s, 60s
    elif duration > 10:
        timestamps = [1, duration/2, duration-1]  # Beginning, middle, near end
    else:
        timestamps = [1, duration/2]  # Just beginning and middle for very short videos
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        extracted_frames = []
        
        for i, timestamp in enumerate(timestamps):
            frame_filename = f"frame_{i:03d}.jpg"
            frame_path = temp_path / frame_filename
            
            # Extract frame at specific timestamp using FFmpeg
            cmd = [
                'ffmpeg',
                '-i', str(test_video),
                '-ss', str(timestamp),  # Seek to timestamp
                '-vframes', '1',  # Extract only 1 frame
                '-q:v', '2',  # High quality
                '-y',  # Overwrite output file
                str(frame_path)
            ]
            
            try:
                print(f"  🎬 Extracting frame at {timestamp}s...")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0 and frame_path.exists():
                    # Read frame and get size
                    frame_size = frame_path.stat().st_size
                    print(f"    ✅ Frame extracted successfully ({frame_size} bytes)")
                    
                    # Test base64 encoding
                    with open(frame_path, 'rb') as f:
                        frame_data = f.read()
                        frame_base64 = base64.b64encode(frame_data).decode('utf-8')
                        print(f"    ✅ Base64 encoding successful ({len(frame_base64)} chars)")
                    
                    extracted_frames.append({
                        "timestamp": timestamp,
                        "frame_index": i,
                        "frame_data": f"data:image/jpeg;base64,{frame_base64}",
                        "size": len(frame_data)
                    })
                else:
                    print(f"    ❌ Failed to extract frame at {timestamp}s")
                    if result.stderr:
                        print(f"    Error: {result.stderr}")
                        
            except subprocess.TimeoutExpired:
                print(f"    ❌ Timeout extracting frame at {timestamp}s")
            except Exception as e:
                print(f"    ❌ Error extracting frame at {timestamp}s: {e}")
    
    print(f"\n📊 Results:")
    print(f"  - Requested frames: {len(timestamps)}")
    print(f"  - Successfully extracted: {len(extracted_frames)}")
    
    if len(extracted_frames) > 0:
        print("✅ Frame extraction test PASSED")
        return True
    else:
        print("❌ Frame extraction test FAILED")
        return False

def test_sentence_transformers():
    """Test if sentence transformers can load and encode images"""
    print("\n🧪 Testing Sentence Transformers")
    print("=" * 50)
    
    try:
        from sentence_transformers import SentenceTransformer
        from PIL import Image
        import io
        import numpy as np
        
        print("✅ Sentence Transformers imported successfully")
        
        # Try to load the CLIP model
        model_name = "clip-ViT-B-32"
        print(f"🤖 Loading model: {model_name}")
        
        model = SentenceTransformer(model_name)
        print("✅ Model loaded successfully")
        
        # Create a test image
        test_image = Image.new('RGB', (224, 224), color='red')
        print("🖼️ Created test image")
        
        # Generate embedding
        embedding = model.encode(test_image)
        print(f"✅ Embedding generated: {len(embedding)} dimensions")
        
        # Test conversion to list
        embedding_list = embedding.tolist()
        print(f"✅ Embedding converted to list: {len(embedding_list)} items")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install with: pip install sentence-transformers torch pillow")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_chromadb():
    """Test ChromaDB functionality"""
    print("\n🧪 Testing ChromaDB")
    print("=" * 50)
    
    try:
        from ml_processing import chroma_manager
        
        if chroma_manager.is_available():
            print("✅ ChromaDB is available")
            
            stats = chroma_manager.get_collection_stats()
            print(f"📊 Collection stats: {stats}")
            
            if stats.get('total_frames', 0) > 0:
                print("✅ ChromaDB has frame embeddings")
                return True
            else:
                print("⚠️ ChromaDB is empty (no frame embeddings)")
                print("💡 Process some videos with ML analysis first")
                return True  # ChromaDB works, just empty
        else:
            print("❌ ChromaDB not available")
            return False
            
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Frame Search Functionality Test")
    print("=" * 50)
    
    tests = [
        ("Frame Extraction", test_frame_extraction),
        ("Sentence Transformers", test_sentence_transformers),
        ("ChromaDB", test_chromadb)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Frame search should work correctly.")
    else:
        print("⚠️ Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main() 