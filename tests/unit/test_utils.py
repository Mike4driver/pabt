import pytest
from pathlib import Path
from unittest.mock import patch

from utils import slugify_for_id, safe_path_join, get_media_type_from_extension


@pytest.mark.unit
class TestUtilityFunctions:
    """Unit tests for utility functions"""
    
    def test_slugify_for_id_basic(self):
        """Test basic slugify functionality"""
        # Normal filename
        result = slugify_for_id("My Video File.mp4")
        assert result == "my-video-file-mp4"
        
        # Already clean filename
        result = slugify_for_id("clean-filename")
        assert result == "clean-filename"
        
        # Filename with underscores
        result = slugify_for_id("file_with_underscores.mp4")
        assert result == "file_with_underscores-mp4"
    
    def test_slugify_for_id_special_characters(self):
        """Test slugify with special characters"""
        # Special characters
        result = slugify_for_id("File with (special) [chars] & symbols!.mp4")
        assert result.startswith("file-with-special-chars-symbols-mp4")
        
        # Multiple spaces and hyphens
        result = slugify_for_id("Multiple   spaces---and--hyphens.mp4")
        assert result == "multiple-spaces-and-hyphens-mp4"
        
        # Leading/trailing hyphens
        result = slugify_for_id("---leading-and-trailing---")
        assert result == "leading-and-trailing"
    
    def test_slugify_for_id_edge_cases(self):
        """Test slugify edge cases"""
        # Empty string
        result = slugify_for_id("")
        assert result.startswith("id-")
        assert len(result) > 3  # Should have hash suffix
        
        # Only special characters
        result = slugify_for_id("!@#$%^&*()")
        assert result.startswith("id-")
        assert len(result) > 3
        
        # Numbers only
        result = slugify_for_id("12345")
        assert result.startswith("id-12345-")
        assert len(result) > 9  # Should have hash suffix
        
        # Starting with number
        result = slugify_for_id("1video.mp4")
        assert result.startswith("id-1video-mp4-")
    
    def test_slugify_for_id_unicode(self):
        """Test slugify with unicode characters"""
        # Unicode characters
        result = slugify_for_id("Vidéo Française.mp4")
        # Should remove accented characters and handle them appropriately
        assert "video" in result.lower()
        assert "mp4" in result
    
    def test_slugify_for_id_consistency(self):
        """Test that slugify produces consistent results for same input"""
        filename = "Test Video File.mp4"
        result1 = slugify_for_id(filename)
        result2 = slugify_for_id(filename)
        assert result1 == result2
    
    def test_safe_path_join_normal(self):
        """Test safe path joining with normal filenames"""
        base = Path("/media")
        filename = "video.mp4"
        result = safe_path_join(base, filename)
        assert result == Path("/media/video.mp4")
    
    def test_safe_path_join_security(self):
        """Test path traversal protection"""
        base = Path("/media")
        
        # Directory traversal attempts
        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/absolute/path/attack",
            "\\absolute\\path\\attack",
            "subfolder/../../../secret.txt"
        ]
        
        for attempt in traversal_attempts:
            result = safe_path_join(base, attempt)
            # Should only use the basename, preventing traversal
            assert not str(result).startswith("/etc")
            assert not str(result).startswith("/absolute")
            assert not str(result).startswith("\\absolute")
            assert result.parent == base
    
    def test_safe_path_join_edge_cases(self):
        """Test safe path joining edge cases"""
        base = Path("/media")
        
        # Empty filename
        result = safe_path_join(base, "")
        assert result == base / ""
        
        # Filename with path separators
        result = safe_path_join(base, "folder/file.mp4")
        assert result == base / "file.mp4"  # Should only take basename
        
        # Windows-style separators
        result = safe_path_join(base, "folder\\file.mp4")
        assert result == base / "file.mp4"
    
    def test_get_media_type_from_extension_video(self):
        """Test video file type detection"""
        video_files = [
            "video.mp4",
            "movie.avi", 
            "clip.mov",
            "content.mkv",
            "VIDEO.MP4",  # Test case insensitivity
        ]
        
        for filename in video_files:
            path = Path(filename)
            assert get_media_type_from_extension(path) == "video"
    
    def test_get_media_type_from_extension_audio(self):
        """Test audio file type detection"""
        audio_files = [
            "song.mp3",
            "track.wav",
            "audio.ogg",
            "MUSIC.MP3",  # Test case insensitivity
        ]
        
        for filename in audio_files:
            path = Path(filename)
            assert get_media_type_from_extension(path) == "audio"
    
    def test_get_media_type_from_extension_image(self):
        """Test image file type detection"""
        image_files = [
            "photo.jpg",
            "picture.jpeg",
            "image.png",
            "animation.gif",
            "PHOTO.JPG",  # Test case insensitivity
        ]
        
        for filename in image_files:
            path = Path(filename)
            assert get_media_type_from_extension(path) == "image"
    
    def test_get_media_type_from_extension_unknown(self):
        """Test unknown file type detection"""
        unknown_files = [
            "document.txt",
            "archive.zip",
            "data.csv",
            "file.unknown",
            "no_extension",
        ]
        
        for filename in unknown_files:
            path = Path(filename)
            assert get_media_type_from_extension(path) == "unknown"
    
    def test_get_media_type_from_extension_no_extension(self):
        """Test file with no extension"""
        path = Path("filename_without_extension")
        assert get_media_type_from_extension(path) == "unknown"
    
    def test_get_media_type_from_extension_pathlib_integration(self):
        """Test integration with pathlib Path objects"""
        # Test with full path
        full_path = Path("/media/videos/movie.mp4")
        assert get_media_type_from_extension(full_path) == "video"
        
        # Test with relative path
        rel_path = Path("./audio/song.mp3")
        assert get_media_type_from_extension(rel_path) == "audio"


@pytest.mark.unit
class TestUtilityIntegration:
    """Integration tests for utility functions working together"""
    
    def test_safe_path_with_slugified_filename(self):
        """Test using safe_path_join with slugified filenames"""
        base = Path("/media")
        unsafe_filename = "../dangerous file (1).mp4"
        
        # First slugify for ID generation
        safe_id = slugify_for_id(unsafe_filename)
        
        # Then use safe path join
        result = safe_path_join(base, safe_id)
        
        # Should be safe and properly formatted
        assert result.parent == base
        assert ".." not in str(result)
        assert "dangerous" in str(result).lower()
    
    def test_media_type_with_safe_paths(self):
        """Test media type detection with safe path operations"""
        base = Path("/media")
        test_files = [
            ("../../../video.mp4", "video"),
            ("\\..\\audio.mp3", "audio"),
            ("subfolder/../image.jpg", "image")
        ]
        
        for unsafe_filename, expected_type in test_files:
            safe_path = safe_path_join(base, unsafe_filename)
            detected_type = get_media_type_from_extension(safe_path)
            assert detected_type == expected_type
    
    def test_utility_functions_with_unicode(self):
        """Test utility functions with unicode filenames"""
        base = Path("/média")
        unicode_filename = "Vidéo française (test).mp4"
        
        # Test safe path join with unicode
        safe_path = safe_path_join(base, unicode_filename)
        assert safe_path.parent == base
        
        # Test media type detection with unicode
        media_type = get_media_type_from_extension(safe_path)
        assert media_type == "video"
        
        # Test slugify with unicode
        slug = slugify_for_id(unicode_filename)
        assert slug is not None
        assert len(slug) > 0