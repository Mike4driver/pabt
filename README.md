# PABT - Personal Audio/Video Browser and Transcoding Toolkit

A web-based application for browsing, managing, and processing media files (videos, images, audio). It features video transcoding, hover previews, thumbnail generation, background job processing, and a configurable media library. Built with FastAPI, HTMX, and Tailwind CSS.

## Features

### Media Browser
*   **File Management:**
    *   Lists files from a configurable media directory stored in database settings.
    *   Displays thumbnails (generates if missing for images and videos).
    *   Shows file metadata (size, duration, resolution, FPS).
    *   Search functionality with filtering by media type and tags.
    *   Hover video previews for video files.
    *   User-defined titles and tagging system for media files.
    *   File deletion capabilities.

### Video Player
*   **Playback Features:**
    *   Plays original or transcoded video versions.
    *   Download original video files.
    *   Configurable autoplay, muting, auto-replay, and autoplay-next settings.
    *   Queue-based navigation with next/previous video functionality.
    *   Full keyboard control with comprehensive keyboard shortcuts.
*   **Processing Features:**
    *   On-demand thumbnail generation.
    *   Basic "Web-Optimized" transcoding for the current video.
    *   Advanced transcoding modal with options for resolution, quality (CRF/bitrate), encoding preset, and H.264 profile.
    *   On-demand hover preview generation.
    *   Transcoded versions become immediately playable.

### Media Processing Tools
*   **Bulk Operations:**
    *   Bulk thumbnail generation for all videos.
    *   Bulk video transcoding with selectable options (resolution, quality mode, CRF/bitrate, audio bitrate, preset).
    *   Bulk hover preview generation for all videos.
*   **Background Job System:**
    *   All processing operations run as background jobs.
    *   Real-time job progress tracking with status updates.
    *   Job history and cleanup for completed/failed jobs.

### Machine Learning Video Processing
*   **Frame Extraction & Analysis:**
    *   Extract frames from videos at configurable intervals (every 30th, 60th, 100th, etc. frame).
    *   Generate embeddings for extracted frames using CLIP models (ViT-B/32, ViT-L/14, ViT-B/16).
    *   Store frame embeddings in both JSON format and ChromaDB vector database for optimal performance.
    *   Background processing with real-time progress tracking.
    *   Support for multiple ML models via sentence-transformers library.
*   **AI-Powered Semantic Search:**
    *   Search videos using natural language queries (e.g., "videos with people", "outdoor scenes", "cars driving").
    *   Automatic detection of semantic search queries vs. traditional filename searches.
    *   Vector similarity search using ChromaDB for fast, accurate results.
    *   Visual similarity scoring and ranking of search results.
    *   Seamless integration with existing search interface.
*   **Vector Database Integration:**
    *   ChromaDB for high-performance vector storage and similarity search.
    *   Dual storage system: ChromaDB for performance + JSON for compatibility.
    *   Migration tools for existing embeddings data.
    *   Collection statistics and health monitoring.
*   **ML Dependencies:**
    *   Uses sentence-transformers for generating image embeddings.
    *   Supports CLIP models for vision-language understanding.
    *   Requires PyTorch for model inference.
    *   ChromaDB for vector database operations.

### Settings & Configuration
*   **Database-Driven Settings:**
    *   Configure the `media_directory_name` (relative to the application's root).
    *   Media grid size preferences (small, medium, large).
    *   Pagination settings (items per page).
    *   Video player preferences (autoplay, muting, auto-replay, autoplay-next).
*   **Persistent Configuration:**
    *   Settings stored in SQLite database.
    *   Automatic migration from legacy `config.json` files.

### Technical Architecture
*   **Backend Technology:**
    *   FastAPI with modular router-based architecture.
    *   SQLite database for media metadata and application settings.
    *   FFmpeg (via `subprocess`) for all video processing tasks.
    *   Pillow for image manipulation (thumbnails).
    *   MoviePy & Mutagen for media metadata extraction.
    *   Background job system with threading for long-running operations.
*   **Frontend Technology:**
    *   HTMX for dynamic page updates without full reloads.
    *   Tailwind CSS for modern, responsive styling.
    *   Real-time progress updates for background operations.

## Setup

### Option 1: Docker (Recommended)

The easiest way to run PABT is using Docker, which includes all dependencies:

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create necessary directories:**
    ```bash
    mkdir -p media media_transcoded ml_analysis data static/previews static/thumbnails
    ```

3.  **Run with Docker Compose:**
    ```bash
    docker-compose up --build
    ```

4.  **Access the application:**
    Open your browser and go to `http://localhost:8000`

For detailed Docker setup instructions, see [DOCKER.md](DOCKER.md).

### Option 2: Manual Installation

1.  **Clone the Repository:**
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
    *   You can change this directory name via the "Settings" page in the application after the first run.
    *   Example: If you set the media directory name to `my_videos` in the settings, create a folder named `my_videos` in the project root.
    *   Place your video, image, and audio files into this directory.

6.  **Database Initialization:**
    The application automatically creates and initializes the SQLite database (`media_library.db`) on first run.
    *   Settings are stored in the database instead of `config.json`.
    *   Legacy `config.json` files are automatically migrated to the database.
    *   You can manually initialize the database by running: `python database.py`

## Running the Application

Start the FastAPI server using Uvicorn:
```bash
python main.py
```
Or alternatively:
```bash
uvicorn main:app --reload
```
The application will be available at `http://localhost:8000`.

## Project Structure

```
pabt/
├── main.py                 # Main FastAPI application entry point
├── config.py              # Configuration management and constants
├── database.py            # Database operations and schema management
├── data_access.py         # Database queries and media file operations
├── media_processing.py    # FFmpeg operations and media processing
├── ml_processing.py       # Machine learning frame analysis and embeddings
├── jobs_manager.py        # Background job system
├── utils.py               # Utility functions
├── routes/                # Modular route definitions
│   ├── media_routes.py    # Media browsing and player routes
│   ├── processing_routes.py # Media processing and transcoding routes
│   ├── settings_routes.py # Application settings routes
│   ├── job_routes.py      # Background job status routes
│   └── ml_routes.py       # Machine learning processing routes
├── templates/             # HTML templates for the frontend
├── static/                # Static assets
│   ├── icons/            # SVG icons
│   ├── previews/         # Generated hover previews (auto-created)
│   └── thumbnails/       # Generated thumbnails (auto-created)
├── media/                 # Default media directory (configurable)
├── media_transcoded/      # Transcoded video files (auto-created)
├── ml_analysis/           # ML analysis results and frame embeddings (auto-created)
├── media_library.db       # SQLite database (auto-created)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Key Features & Changes

### Modular Architecture
*   Routes are organized into separate modules for better maintainability.
*   Configuration, database operations, and utilities are separated into dedicated modules.
*   Background job system handles long-running operations without blocking the UI.

### Database-Driven Configuration
*   Application settings are stored in SQLite database instead of JSON files.
*   Automatic schema migration and settings initialization.
*   Persistent user preferences for media grid size, pagination, and player settings.

### Enhanced Media Management
*   User-defined titles and tagging system for better organization.
*   Advanced search and filtering capabilities.
*   Queue-based video navigation with customizable player behavior.

### Background Processing
*   All media processing operations run as background jobs.
*   Real-time progress tracking and status updates.
*   Job history and automatic cleanup of old completed jobs.

## Keyboard Shortcuts (Video Player)

The video player supports comprehensive keyboard navigation and control:

### Video Playback
*   **Space** - Play/pause video
*   **← Arrow** - Seek backward 10 seconds
*   **→ Arrow** - Seek forward 10 seconds  
*   **↑ Arrow** - Increase volume
*   **↓ Arrow** - Decrease volume
*   **M** - Toggle mute
*   **F** - Toggle fullscreen

### Navigation
*   **Ctrl + ← Arrow** - Previous video in queue
*   **Ctrl + → Arrow** - Next video in queue
*   **Q** - Toggle queue sidebar
*   **C** - Center current video in queue

### Advanced Features
*   **Ctrl + S** - Open frame search modal (for semantic video search)
*   **?** (Shift + /) - Show keyboard shortcuts modal

### General
*   **Esc** - Close any open modal or dialog

*Note: Keyboard shortcuts are disabled when typing in input fields or text areas.*

## Notes

*   **FFmpeg Dependency:** The application heavily relies on FFmpeg. If FFmpeg is not found in your system's PATH, transcoding and preview generation features will fail.
*   **Database Migration:** Legacy `config.json` files are automatically migrated to the database on first run.
*   **Background Jobs:** Long-running operations like bulk transcoding run in the background with progress tracking.
*   **File Paths & Slugification:** Filenames are slugified for IDs and generated file paths to prevent issues with special characters.
*   **Settings Persistence:** All configuration changes are immediately saved to the database and persist across application restarts. 