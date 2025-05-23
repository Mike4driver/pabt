# Media Browser and Transcoding Toolkit

A web-based application for browsing, managing, and processing media files (videos, images, audio). It features video transcoding, hover previews, thumbnail generation, and a configurable media library. Built with FastAPI, HTMX, and Tailwind CSS.

## Features

*   **Media Browser:**
    *   Lists files from a configurable media directory.
    *   Displays thumbnails (generates if missing for images and videos).
    *   Shows file metadata (size, duration, resolution, FPS).
    *   Search functionality.
    *   Hover video previews for video files.
*   **Video Player:**
    *   Plays original or transcoded video versions.
    *   Download original video.
    *   On-demand thumbnail generation.
    *   Basic "Web-Optimized" transcoding for the current video.
    *   Advanced transcoding modal with options for resolution, quality (CRF/bitrate), encoding preset, and H.264 profile. Transcoded versions become immediately playable.
    *   On-demand hover preview generation.
*   **Media Processing Tools Page:**
    *   Bulk thumbnail generation for all videos.
    *   Bulk video transcoding with selectable options (resolution, quality mode, CRF/bitrate, audio bitrate, preset).
    *   Bulk hover preview generation for all videos.
*   **Settings Page:**
    *   Configure the `media_directory_name` (relative to the application's root).
    *   Changes require an application restart.
*   **Backend Technology:**
    *   FastAPI for the web framework.
    *   FFmpeg (via `subprocess`) for all video processing tasks.
    *   Pillow for image manipulation (thumbnails).
    *   MoviePy & Mutagen for media metadata.
*   **Frontend Technology:**
    *   HTMX for dynamic page updates without full reloads.
    *   Tailwind CSS for styling.

## Setup

1.  **Clone the Repository (if you haven't):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install FFmpeg:**
    FFmpeg is crucial for video transcoding and preview generation. Ensure it's installed and accessible in your system's PATH.
    *   **Windows:** Download from [FFmpeg's official site](https://ffmpeg.org/download.html) and add the `bin` directory to your PATH.
    *   **macOS (using Homebrew):** `brew install ffmpeg`
    *   **Linux (using apt):** `sudo apt update && sudo apt install ffmpeg`

3.  **Create a Python Virtual Environment (recommended):**
    ```bash
    python -m venv venv
    ```
    Activate it:
    *   Windows: `venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`

4.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Create Your Media Directory:**
    *   By default, the application looks for a directory named `media` in the project root.
    *   You can change this directory name via the "Settings" page in the application after the first run. If you change it, you'll need to create this directory yourself.
    *   Example: If you set the media directory name to `my_videos` in the settings, create a folder named `my_videos` in the project root.
    *   Place your video, image, and audio files into this directory.

6.  **Initial Configuration (Optional - `config.json`):**
    The application uses a `config.json` file in the project root to store the name of your media directory.
    *   If this file doesn't exist on first run, it will be created with `{"media_directory_name": "media"}`.
    *   You can manually create/edit `config.json` before the first run if you prefer, e.g.:
        ```json
        {
            "media_directory_name": "my_custom_media_folder"
        }
        ```
        Ensure `my_custom_media_folder` exists in the project root.

## Running the Application

Start the FastAPI server using Uvicorn:
```bash
uvicorn main:app --reload
```
The application will typically be available at `http://127.0.0.1:8000`.

## Directory Structure

*   `main.py`: The main FastAPI application.
*   `media/`: Default directory for your media files (configurable).
*   `media_transcoded/`: Stores transcoded video files. Automatically created.
*   `static/`: For static assets.
    *   `icons/`: SVG icons.
    *   `previews/`: Stores generated hover previews. Automatically created.
    *   `thumbnails/`: Stores generated thumbnails. Automatically created.
*   `templates/`: HTML templates for the frontend.
*   `config.json`: Stores application configuration (e.g., media directory name).
*   `requirements.txt`: Python dependencies.
*   `README.md`: This file.

## Notes

*   **FFmpeg Dependency:** The application heavily relies on FFmpeg. If FFmpeg is not found in your system's PATH, transcoding and preview generation features will fail.
*   **Configuration Changes:** When changing the media directory via the Settings page, an application restart is required for the changes to take full effect.
*   **File Paths & Slugification:** Filenames are slugified for IDs and generated file paths to prevent issues with special characters. 