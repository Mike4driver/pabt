import pytest
import tempfile
import sqlite3
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from PIL import Image
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import create_tables, get_db_connection
from main import app


@pytest.fixture(scope="session")
def test_config():
    """Configuration for tests"""
    return {
        "database_name": ":memory:",
        "media_dir": "test_media",
        "transcoded_dir": "test_transcoded",
        "previews_dir": "test_previews",
        "thumbnails_dir": "test_thumbnails",
        "test_port": 8001
    }


@pytest.fixture(scope="function")
def temp_dirs(test_config):
    """Create temporary directories for testing"""
    temp_base = Path(tempfile.mkdtemp())
    
    dirs = {
        "base": temp_base,
        "media": temp_base / test_config["media_dir"],
        "transcoded": temp_base / test_config["transcoded_dir"],
        "previews": temp_base / test_config["previews_dir"],
        "thumbnails": temp_base / test_config["thumbnails_dir"],
        "static": temp_base / "static"
    }
    
    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create static subdirectories
    (dirs["static"] / "previews").mkdir(exist_ok=True)
    (dirs["static"] / "thumbnails").mkdir(exist_ok=True)
    (dirs["static"] / "icons").mkdir(exist_ok=True)
    
    yield dirs
    
    # Cleanup
    shutil.rmtree(temp_base, ignore_errors=True)


@pytest.fixture(scope="function")
def test_db(temp_dirs):
    """Create a test database with tables"""
    db_path = temp_dirs["base"] / "test_media.db"
    
    # Patch the database name globally for this test
    with patch('database.DATABASE_NAME', str(db_path)):
        create_tables()
        
        # Add some test settings
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", 
                      ("media_directory_name", "test_media"))
        cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", 
                      ("media_grid_size", "medium"))
        cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", 
                      ("per_page", "20"))
        conn.commit()
        conn.close()
        
        yield str(db_path)


@pytest.fixture(scope="function")
def test_media_files(temp_dirs):
    """Create test media files"""
    media_dir = temp_dirs["media"]
    
    # Create test video file (placeholder)
    test_video = media_dir / "test_video.mp4"
    test_video.write_bytes(b"fake video content for testing")
    
    # Create test audio file
    test_audio = media_dir / "test_audio.mp3"
    test_audio.write_bytes(b"fake audio content for testing")
    
    # Create test image
    test_image = media_dir / "test_image.jpg"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(test_image, 'JPEG')
    
    # Create nested directory with file
    nested_dir = media_dir / "subfolder"
    nested_dir.mkdir()
    nested_video = nested_dir / "nested_video.mp4"
    nested_video.write_bytes(b"fake nested video content")
    
    return {
        "video": test_video,
        "audio": test_audio,
        "image": test_image,
        "nested_video": nested_video,
        "media_dir": media_dir
    }


@pytest.fixture(scope="function")
def mock_ffmpeg():
    """Mock FFmpeg subprocess calls"""
    with patch('subprocess.run') as mock_run:
        # Mock successful FFmpeg execution
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "duration=120.5"
        mock_run.return_value.stderr = ""
        yield mock_run


@pytest.fixture(scope="function")
def mock_ml_models():
    """Mock ML model dependencies"""
    with patch('sentence_transformers.SentenceTransformer') as mock_st, \
         patch('chromadb.Client') as mock_chroma:
        
        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4] * 128]  # 512-dim embedding
        mock_st.return_value = mock_model
        
        # Mock ChromaDB
        mock_collection = Mock()
        mock_collection.add.return_value = None
        mock_collection.query.return_value = {
            'ids': [['doc1', 'doc2']],
            'distances': [[0.1, 0.3]],
            'metadatas': [[{'video_id': 1, 'timestamp': 10.0}, {'video_id': 2, 'timestamp': 20.0}]]
        }
        mock_chroma_client = Mock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_chroma_client
        
        yield {
            'sentence_transformer': mock_st,
            'chromadb': mock_chroma,
            'model': mock_model,
            'collection': mock_collection
        }


@pytest.fixture(scope="function")
def client(test_db, temp_dirs, mock_ffmpeg):
    """FastAPI test client with mocked dependencies"""
    with patch('config.BASE_DIR', temp_dirs["base"]), \
         patch('config.MEDIA_DIR', temp_dirs["media"]), \
         patch('config.TRANSCODED_DIR', temp_dirs["transcoded"]), \
         patch('config.PREVIEWS_DIR', temp_dirs["previews"]), \
         patch('config.THUMBNAILS_DIR', temp_dirs["thumbnails"]), \
         patch('database.DATABASE_NAME', test_db):
        
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture(scope="function")
def selenium_driver():
    """Selenium WebDriver for integration tests"""
    pytest.importorskip("selenium")
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.service import Service
    
    # Chrome options for headless testing
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Create driver with WebDriver Manager
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.implicitly_wait(10)
    
    yield driver
    
    driver.quit()


@pytest.fixture(scope="function")
def sample_video_data():
    """Sample video metadata for testing"""
    return {
        "filename": "sample_video.mp4",
        "original_path": "/test/media/sample_video.mp4",
        "media_type": "video",
        "user_title": "Sample Test Video",
        "duration": 120.5,
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "size_bytes": 1024 * 1024 * 10,  # 10MB
        "tags": '["test", "sample", "video"]',
        "metadata_json": '{"codec": "h264", "bitrate": "1000k"}'
    }


@pytest.fixture(scope="function")
def populated_test_db(test_db, sample_video_data):
    """Test database with sample data"""
    with patch('database.DATABASE_NAME', test_db):
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert sample video
        cursor.execute("""
            INSERT INTO media_files 
            (filename, original_path, media_type, user_title, duration, width, height, 
             fps, size_bytes, tags, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sample_video_data["filename"],
            sample_video_data["original_path"],
            sample_video_data["media_type"],
            sample_video_data["user_title"],
            sample_video_data["duration"],
            sample_video_data["width"],
            sample_video_data["height"],
            sample_video_data["fps"],
            sample_video_data["size_bytes"],
            sample_video_data["tags"],
            sample_video_data["metadata_json"]
        ))
        
        # Insert additional test videos
        test_videos = [
            ("video1.mp4", "Test Video 1", '["action", "drama"]'),
            ("video2.avi", "Test Video 2", '["comedy", "test"]'),
            ("video3.mov", "Test Video 3", '["documentary"]')
        ]
        
        for filename, title, tags in test_videos:
            cursor.execute("""
                INSERT INTO media_files 
                (filename, original_path, media_type, user_title, duration, tags)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (filename, f"/test/media/{filename}", "video", title, 60.0, tags))
        
        conn.commit()
        conn.close()
        
        yield test_db


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test"""
    yield
    # Cleanup any temporary files that might have been created
    temp_patterns = [
        "test_*.db",
        "test_*.jpg",
        "test_*.mp4",
        "test_*.mp3"
    ]
    
    for pattern in temp_patterns:
        for file in Path(".").glob(pattern):
            try:
                file.unlink()
            except:
                pass