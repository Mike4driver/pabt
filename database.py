import sqlite3
import json
from pathlib import Path

DATABASE_NAME = "media_library.db"
CONFIG_FILE = Path("config.json")

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    """Creates the necessary tables if they don't already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Media files table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS media_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL UNIQUE,
        original_path TEXT NOT NULL,
        media_type TEXT CHECK(media_type IN ('video', 'audio', 'image')),
        user_title TEXT, -- User-defined title
        duration REAL,
        width INTEGER,
        height INTEGER,
        fps REAL,
        size_bytes INTEGER,
        date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_scanned TIMESTAMP,
        thumbnail_path TEXT, -- Path to specific thumbnail, if generated
        has_specific_thumbnail BOOLEAN DEFAULT FALSE,
        transcoded_path TEXT,
        has_transcoded_version BOOLEAN DEFAULT FALSE,
        preview_path TEXT, -- Path to hover preview, if generated
        has_preview BOOLEAN DEFAULT FALSE,
        tags TEXT, -- JSON string array of tags, e.g., '["tag1", "tag2"]'
        metadata_json TEXT -- For any other metadata (e.g., from ffprobe)
    )
    """)

    # Application settings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)
    
    # --- Schema Migration (Add new columns if they don't exist) ---
    # Check if columns exist (e.g., user_title and tags)
    # This is a simple check, more complex migrations might need versioning
    cursor.execute("PRAGMA table_info(media_files)")
    columns = [col[1] for col in cursor.fetchall()]

    if 'user_title' not in columns:
        print("Adding 'user_title' column to media_files table...")
        cursor.execute("ALTER TABLE media_files ADD COLUMN user_title TEXT")
        print("'user_title' column added.")

    if 'tags' not in columns:
        print("Adding 'tags' column to media_files table...")
        cursor.execute("ALTER TABLE media_files ADD COLUMN tags TEXT DEFAULT '[]'") # Default to empty JSON array
        print("'tags' column added.")

    # --- End Schema Migration ---

    # Default settings
    # Check if media_directory_name setting exists
    cursor.execute("SELECT value FROM settings WHERE key = 'media_directory_name'")
    if cursor.fetchone() is None:
        # Try to load from config.json if it exists
        media_dir = "media" # default
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config_data = json.load(f)
                    media_dir = config_data.get("media_directory_name", "media")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode {CONFIG_FILE}. Using default media directory.")
        
        cursor.execute("INSERT INTO settings (key, value) VALUES (?, ?)", 
                       ('media_directory_name', media_dir))
        print(f"Initialized 'media_directory_name' setting to: {media_dir}")


    conn.commit()
    conn.close()
    print("Database tables created/verified.")

def get_setting(key: str):
    """Retrieves a setting value from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row['value'] if row else None

def update_setting(key: str, value: str):
    """Updates or inserts a setting value in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()
    print(f"Setting '{key}' updated to '{value}'.")

if __name__ == "__main__":
    # This allows running `python database.py` to initialize the DB
    create_tables()
    
    # Example: Update media directory if config.json exists and is different or not set
    media_dir_db = get_setting("media_directory_name")
    
    current_config_media_dir = "media" # default
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config_data = json.load(f)
                current_config_media_dir = config_data.get("media_directory_name", "media")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {CONFIG_FILE} during standalone run.")

    if media_dir_db != current_config_media_dir:
        print(f"Media directory in DB ('{media_dir_db}') differs from {CONFIG_FILE} ('{current_config_media_dir}').")
        print(f"The application will use the value from {CONFIG_FILE} on next startup if it's different and `main.py` is updated to sync this.")
        # In a real scenario, main.py would handle syncing this at startup.
        # For now, we're just reporting. If you want to force update from config.json via this script:
        # update_setting("media_directory_name", current_config_media_dir)
        # print(f"Updated DB media_directory_name to '{current_config_media_dir}' based on {CONFIG_FILE}.")

    print(f"Current media directory from DB: {get_setting('media_directory_name')}")
    print("Database setup script finished.") 