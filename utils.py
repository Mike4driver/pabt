import os
from pathlib import Path
import re

# Assuming SUPPORTED_VIDEO_EXTENSIONS, etc., are defined in config.py or passed as arguments
# For now, let's hardcode them here or assume they will be imported if this utils.py grows
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"} # Example
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".aac", ".flac"} # Example
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp"} # Example

def slugify_for_id(value: str) -> str:
    import hashlib
    
    # Store original value for hash generation if needed
    original_value = value
    
    # Remove characters that are not alphanumeric, underscores, or hyphens
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    # Replace whitespace and sequences of hyphens with a single hyphen
    value = re.sub(r'[-\s]+', '-', value)
    # Remove leading/trailing hyphens
    value = value.strip('-')
    
    # If the result is empty or doesn't start with a letter, create a hash-based ID
    if not value or not value[0].isalpha():
        # Create a hash of the original filename for uniqueness
        hash_suffix = hashlib.md5(original_value.encode('utf-8')).hexdigest()[:8]
        if value:
            value = f"id-{value}-{hash_suffix}"
        else:
            value = f"id-{hash_suffix}"
    
    return value

def safe_path_join(base_path: Path, filename: str) -> Path:
    """Safely join a base path with a filename, preventing directory traversal."""
    # Normalize the filename to prevent directory traversal
    safe_filename = os.path.basename(filename)
    # It's crucial that base_path is a trusted, absolute path or resolved correctly.
    # Ensure the combined path does not escape the intended directory if base_path is relative
    # However, if base_path is already verified (e.g., MEDIA_DIR), this is simpler.
    return base_path / safe_filename

def get_media_type_from_extension(file_path: Path) -> str:
    """Determine media type from file extension."""
    suffix = file_path.suffix.lower()
    if suffix in SUPPORTED_VIDEO_EXTENSIONS:
        return "video"
    elif suffix in SUPPORTED_AUDIO_EXTENSIONS:
        return "audio"
    elif suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    return "unknown" 