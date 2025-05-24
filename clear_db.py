import os
from pathlib import Path

DATABASE_FILE = Path(__file__).resolve().parent / "media_library.db"

if DATABASE_FILE.exists():
    try:
        os.remove(DATABASE_FILE)
        print(f"Successfully deleted database file: {DATABASE_FILE}")
    except OSError as e:
        print(f"Error deleting database file {DATABASE_FILE}: {e}")
else:
    print(f"Database file not found: {DATABASE_FILE}")

print("Database clearing script finished.") 