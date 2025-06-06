import pytest
import json
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
from pathlib import Path

from main import app


@pytest.mark.unit
class TestMediaRoutes:
    """Unit tests for media routes"""
    
    def test_home_page(self, client):
        """Test home page loads correctly"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_files_list_no_params(self, client, populated_test_db):
        """Test files list without parameters"""
        response = client.get("/files")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_files_list_with_search(self, client, populated_test_db):
        """Test files list with search parameter"""
        response = client.get("/files?search=test")
        assert response.status_code == 200
        # Should contain search results
        content = response.content.decode()
        assert "test" in content.lower()
    
    def test_files_list_with_pagination(self, client, populated_test_db):
        """Test files list with pagination"""
        response = client.get("/files?page=1&per_page=2")
        assert response.status_code == 200
        
        response = client.get("/files?page=2&per_page=2")
        assert response.status_code == 200
    
    def test_files_list_with_media_type_filter(self, client, populated_test_db):
        """Test files list with media type filter"""
        response = client.get("/files?media_type=video")
        assert response.status_code == 200
        
        # Test invalid media type
        response = client.get("/files?media_type=invalid")
        assert response.status_code == 200  # Should still work, just return no results
    
    def test_files_list_with_tags_filter(self, client, populated_test_db):
        """Test files list with tags filter"""
        response = client.get("/files?tags=test,drama")
        assert response.status_code == 200
    
    def test_files_list_with_sorting(self, client, populated_test_db):
        """Test files list with sorting options"""
        # Test different sort options
        sort_options = [
            ("date_added", "desc"),
            ("date_added", "asc"),
            ("filename", "asc"),
            ("duration", "desc")
        ]
        
        for sort_by, sort_order in sort_options:
            response = client.get(f"/files?sort_by={sort_by}&sort_order={sort_order}")
            assert response.status_code == 200
    
    def test_api_tags_endpoint(self, client, populated_test_db):
        """Test API tags endpoint"""
        response = client.get("/api/tags")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert "tags" in data
        assert isinstance(data["tags"], list)
    
    def test_semantic_search_debug(self, client, mock_ml_models):
        """Test semantic search debug endpoint"""
        response = client.get("/api/semantic-search/debug")
        assert response.status_code == 200
        
        data = response.json()
        assert "chromadb" in data
        assert "sentence_transformers" in data
        assert "videos" in data
        assert "recommendations" in data
    
    @patch('routes.media_routes.semantic_search_videos')
    def test_semantic_search_api(self, mock_search, client, mock_ml_models):
        """Test semantic search API endpoint"""
        # Mock the semantic search function
        mock_search.return_value = [
            {
                "id_db": 1,
                "filename": "test_video.mp4",
                "user_title": "Test Video",
                "similarity_score": 0.8
            }
        ]
        
        response = client.post("/api/semantic-search", data={
            "query_text": "test query",
            "page": 1,
            "per_page": 20,
            "similarity_threshold": 0.7
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "media_files" in data
        assert "pagination" in data
        assert "search_type" in data
        assert data["search_type"] == "semantic"
    
    def test_video_player_page(self, client, populated_test_db, test_media_files):
        """Test video player page"""
        response = client.get("/video/sample_video.mp4")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_video_player_page_with_timestamp(self, client, populated_test_db):
        """Test video player page with timestamp parameter"""
        response = client.get("/video/sample_video.mp4?t=30.5")
        assert response.status_code == 200
        content = response.content.decode()
        # Should contain the timestamp in the response
        assert "30.5" in content
    
    def test_video_player_page_not_found(self, client):
        """Test video player page with non-existent video"""
        response = client.get("/video/nonexistent.mp4")
        assert response.status_code == 404
    
    def test_serve_media_file_success(self, client, test_media_files):
        """Test serving media files"""
        with patch('routes.media_routes.MEDIA_DIR', test_media_files["media_dir"]):
            response = client.get("/media_content/test_video.mp4")
            assert response.status_code == 200
            assert response.content == b"fake video content for testing"
    
    def test_serve_media_file_not_found(self, client):
        """Test serving non-existent media file"""
        response = client.get("/media_content/nonexistent.mp4")
        assert response.status_code == 404
    
    def test_update_video_metadata(self, client, populated_test_db):
        """Test updating video metadata"""
        response = client.post("/video/1/metadata", data={
            "user_title": "Updated Title",
            "tags_str": "tag1, tag2, tag3"
        })
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_update_video_metadata_not_found(self, client):
        """Test updating metadata for non-existent video"""
        response = client.post("/video/999/metadata", data={
            "user_title": "Updated Title"
        })
        
        assert response.status_code == 404
    
    def test_remove_video_tag(self, client, populated_test_db):
        """Test removing a tag from video"""
        response = client.delete("/video/1/tag/test")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_remove_video_tag_not_found(self, client):
        """Test removing tag from non-existent video"""
        response = client.delete("/video/999/tag/test")
        assert response.status_code == 404
    
    def test_get_next_video(self, client, populated_test_db):
        """Test getting next video in queue"""
        response = client.get("/video/1/next")
        # Should redirect to next video
        assert response.status_code == 200
        assert "HX-Redirect" in response.headers or response.status_code == 404
    
    def test_delete_media_file(self, client, populated_test_db, test_media_files):
        """Test deleting a media file"""
        response = client.delete("/delete-media-file/1")
        # Should redirect after deletion
        assert response.status_code == 200
        assert "HX-Redirect" in response.headers
    
    def test_delete_media_file_not_found(self, client):
        """Test deleting non-existent media file"""
        response = client.delete("/delete-media-file/999")
        assert response.status_code == 404


@pytest.mark.unit
class TestAPIEndpoints:
    """Unit tests for API endpoints"""
    
    @patch('routes.media_routes.subprocess.run')
    def test_extract_video_frames_success(self, mock_subprocess, client, test_media_files):
        """Test successful frame extraction"""
        # Mock FFmpeg subprocess calls
        mock_subprocess.side_effect = [
            # Duration check
            Mock(returncode=0, stdout="120.5", stderr=""),
            # Frame extraction calls
            Mock(returncode=0, stdout="", stderr=""),
            Mock(returncode=0, stdout="", stderr=""),
        ]
        
        with patch('routes.media_routes.MEDIA_DIR', test_media_files["media_dir"]), \
             patch('routes.media_routes.base64.b64encode') as mock_b64:
            
            mock_b64.return_value = b"fake_base64_data"
            
            response = client.post("/api/extract-frames/test_video.mp4", data={
                "timestamps": "10.5,30.0,60.5",
                "max_frames": 3
            })
            
            assert response.status_code == 200
            data = response.json()
            assert "frames" in data
            assert len(data["frames"]) <= 3
    
    def test_extract_video_frames_not_found(self, client):
        """Test frame extraction with non-existent video"""
        response = client.post("/api/extract-frames/nonexistent.mp4", data={
            "timestamps": "10.5",
            "max_frames": 1
        })
        
        assert response.status_code == 404
    
    def test_extract_video_frames_invalid_timestamps(self, client, test_media_files):
        """Test frame extraction with invalid timestamps"""
        with patch('routes.media_routes.MEDIA_DIR', test_media_files["media_dir"]):
            response = client.post("/api/extract-frames/test_video.mp4", data={
                "timestamps": "invalid,timestamps",
                "max_frames": 1
            })
            
            assert response.status_code == 400
    
    @patch('routes.media_routes.semantic_search_videos')
    def test_semantic_search_by_frame(self, mock_search, client, mock_ml_models):
        """Test semantic search by frame data"""
        mock_search.return_value = [
            {
                "id_db": 1,
                "filename": "similar_video.mp4",
                "similarity_score": 0.9
            }
        ]
        
        # Create a fake base64 image
        fake_frame_data = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAA"
        
        response = client.post("/api/semantic-search-by-frame", data={
            "frame_data": fake_frame_data,
            "similarity_threshold": 0.35,
            "max_results": 20
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "similar_videos" in data
        assert "query_info" in data


@pytest.mark.unit
class TestRouteErrorHandling:
    """Unit tests for route error handling"""
    
    def test_invalid_video_id_format(self, client):
        """Test handling of invalid video ID format"""
        response = client.post("/video/invalid_id/metadata", data={
            "user_title": "Test"
        })
        # FastAPI should handle type conversion errors
        assert response.status_code in [400, 422]  # Bad request or validation error
    
    def test_malformed_json_in_request(self, client):
        """Test handling of malformed JSON data"""
        response = client.post("/api/semantic-search", 
                             data={"query_text": "test"},
                             headers={"Content-Type": "application/json"})
        # Should handle the content-type mismatch gracefully
        assert response.status_code in [200, 400, 422]
    
    @patch('routes.media_routes.get_single_video_details_from_db')
    def test_database_error_handling(self, mock_db_func, client):
        """Test handling of database errors"""
        # Mock database function to raise an exception
        mock_db_func.side_effect = Exception("Database error")
        
        response = client.get("/video/test.mp4")
        assert response.status_code == 500
    
    def test_large_timestamp_values(self, client, populated_test_db):
        """Test handling of very large timestamp values"""
        response = client.get("/video/sample_video.mp4?t=999999999")
        # Should not crash, timestamp should be handled gracefully
        assert response.status_code == 200
    
    def test_negative_timestamp_values(self, client, populated_test_db):
        """Test handling of negative timestamp values"""
        response = client.get("/video/sample_video.mp4?t=-10")
        # Should handle negative timestamps gracefully
        assert response.status_code == 200


@pytest.mark.unit
class TestRoutePermissions:
    """Unit tests for route access control (if implemented)"""
    
    def test_tools_page_access(self, client):
        """Test access to tools page"""
        response = client.get("/tools/media-processing")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]