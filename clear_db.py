import os
import shutil
import socket
from pathlib import Path

# Database and data directories
DATABASE_FILE = Path(__file__).resolve().parent / "media_library.db"
CHROMA_DB_DIR = Path(__file__).resolve().parent / "chroma_db"
ML_ANALYSIS_DIR = Path(__file__).resolve().parent / "ml_analysis"
TRANSCODED_DIR = Path(__file__).resolve().parent / "media_transcoded"
PREVIEWS_DIR = Path(__file__).resolve().parent / "static" / "previews"
THUMBNAILS_DIR = Path(__file__).resolve().parent / "static" / "thumbnails"

def check_server_running():
    """Check if PABT server is running on port 8000"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        return result == 0
    except:
        return False

def clear_sqlite_database():
    """Clear the SQLite database file"""
    if DATABASE_FILE.exists():
        try:
            os.remove(DATABASE_FILE)
            print(f"✅ Successfully deleted SQLite database: {DATABASE_FILE}")
            return True
        except OSError as e:
            print(f"❌ Error deleting SQLite database {DATABASE_FILE}: {e}")
            return False
    else:
        print(f"⚠️ SQLite database file not found: {DATABASE_FILE}")
        return True

def clear_chroma_database():
    """Clear the ChromaDB directory"""
    if CHROMA_DB_DIR.exists():
        try:
            # Count files before deletion for reporting
            total_files = sum(1 for _ in CHROMA_DB_DIR.rglob('*') if _.is_file())
            
            shutil.rmtree(CHROMA_DB_DIR)
            print(f"✅ Successfully deleted ChromaDB directory: {CHROMA_DB_DIR}")
            print(f"   📁 Removed {total_files} ChromaDB files")
            return True
        except OSError as e:
            if "being used by another process" in str(e) or "WinError 32" in str(e):
                print(f"⚠️ ChromaDB directory is locked (server running): {CHROMA_DB_DIR}")
                print(f"   💡 Stop the PABT server first, then run this script again")
                return False
            else:
                print(f"❌ Error deleting ChromaDB directory {CHROMA_DB_DIR}: {e}")
                return False
    else:
        print(f"⚠️ ChromaDB directory not found: {CHROMA_DB_DIR}")
        return True

def clear_ml_analysis():
    """Clear the ML analysis directory"""
    if ML_ANALYSIS_DIR.exists():
        try:
            # Count subdirectories before deletion for reporting
            video_dirs = [d for d in ML_ANALYSIS_DIR.iterdir() if d.is_dir() and d.name.startswith('video_')]
            frame_count = 0
            
            # Count total frames for reporting
            for video_dir in video_dirs:
                frames_dir = video_dir / "frames"
                if frames_dir.exists():
                    frame_count += len(list(frames_dir.glob("*.jpg")))
            
            shutil.rmtree(ML_ANALYSIS_DIR)
            print(f"✅ Successfully deleted ML analysis directory: {ML_ANALYSIS_DIR}")
            print(f"   📊 Removed analysis for {len(video_dirs)} videos")
            print(f"   🖼️ Removed {frame_count} extracted frames")
            return True
        except OSError as e:
            print(f"❌ Error deleting ML analysis directory {ML_ANALYSIS_DIR}: {e}")
            return False
    else:
        print(f"⚠️ ML analysis directory not found: {ML_ANALYSIS_DIR}")
        return True

def clear_transcoded_files():
    """Clear the transcoded media directory"""
    if TRANSCODED_DIR.exists():
        try:
            total_files = sum(1 for _ in TRANSCODED_DIR.rglob('*') if _.is_file())
            shutil.rmtree(TRANSCODED_DIR)
            # Recreate the directory as the application expects it to exist
            TRANSCODED_DIR.mkdir(exist_ok=True)
            print(f"✅ Successfully deleted transcoded media directory: {TRANSCODED_DIR}")
            print(f"   🎞️ Removed {total_files} transcoded files")
            return True
        except OSError as e:
            print(f"❌ Error deleting transcoded media directory {TRANSCODED_DIR}: {e}")
            return False
    else:
        print(f"⚠️ Transcoded media directory not found: {TRANSCODED_DIR}")
        return True

def clear_preview_files():
    """Clear the video previews directory"""
    if PREVIEWS_DIR.exists():
        try:
            total_files = sum(1 for _ in PREVIEWS_DIR.rglob('*') if _.is_file())
            shutil.rmtree(PREVIEWS_DIR)
            # Recreate the directory as the application expects it to exist
            PREVIEWS_DIR.mkdir(exist_ok=True)
            print(f"✅ Successfully deleted previews directory: {PREVIEWS_DIR}")
            print(f"   🖼️ Removed {total_files} preview files")
            return True
        except OSError as e:
            print(f"❌ Error deleting previews directory {PREVIEWS_DIR}: {e}")
            return False
    else:
        print(f"⚠️ Previews directory not found: {PREVIEWS_DIR}")
        return True

def clear_thumbnail_files():
    """Clear the video thumbnails directory"""
    if THUMBNAILS_DIR.exists():
        try:
            total_files = sum(1 for _ in THUMBNAILS_DIR.rglob('*') if _.is_file())
            shutil.rmtree(THUMBNAILS_DIR)
            # Recreate the directory as the application expects it to exist
            THUMBNAILS_DIR.mkdir(exist_ok=True)
            print(f"✅ Successfully deleted thumbnails directory: {THUMBNAILS_DIR}")
            print(f"   🏞️ Removed {total_files} thumbnail files")
            return True
        except OSError as e:
            print(f"❌ Error deleting thumbnails directory {THUMBNAILS_DIR}: {e}")
            return False
    else:
        print(f"⚠️ Thumbnails directory not found: {THUMBNAILS_DIR}")
        return True

def main():
    """Main function to clear all databases and ML data"""
    print("🗑️ PABT Database Clearing Script")
    print("=" * 50)
    print("This will delete:")
    print("• SQLite media database")
    print("• ChromaDB vector database")
    print("• ML analysis results and extracted frames")
    print("• Transcoded media directory")
    print("• Video previews directory")
    print("• Video thumbnails directory")
    print("=" * 50)
    
    # Check if server is running
    if check_server_running():
        print("⚠️ WARNING: PABT server appears to be running on port 8000")
        print("💡 For best results, stop the server first (Ctrl+C) before clearing databases")
        print("   ChromaDB files may be locked and unable to delete while server is running")
        print()
    
    # Ask for confirmation
    try:
        confirm = input("Are you sure you want to proceed? (yes/no): ").lower().strip()
        if confirm not in ['yes', 'y']:
            print("❌ Operation cancelled by user")
            return
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return
    
    print("\n🚀 Starting database clearing process...")
    
    # Clear each component
    sqlite_success = clear_sqlite_database()
    chroma_success = clear_chroma_database()
    ml_success = clear_ml_analysis()
    transcoded_success = clear_transcoded_files()
    previews_success = clear_preview_files()
    thumbnails_success = clear_thumbnail_files()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 CLEARING SUMMARY")
    print("=" * 50)
    print(f"SQLite Database: {'✅ CLEARED' if sqlite_success else '❌ FAILED'}")
    print(f"ChromaDB:        {'✅ CLEARED' if chroma_success else '❌ FAILED'}")
    print(f"ML Analysis:     {'✅ CLEARED' if ml_success else '❌ FAILED'}")
    print(f"Transcoded Media: {'✅ CLEARED' if transcoded_success else '❌ FAILED'}")
    print(f"Previews:         {'✅ CLEARED' if previews_success else '❌ FAILED'}")
    print(f"Thumbnails:       {'✅ CLEARED' if thumbnails_success else '❌ FAILED'}")
    
    if all([sqlite_success, chroma_success, ml_success, transcoded_success, previews_success, thumbnails_success]):
        print("\n🎉 All databases and ML data cleared successfully!")
        print("\n📋 Next steps:")
        print("1. Restart the PABT server")
        print("2. Use 'Scan Media Directory' in tools to rebuild the database")
        print("3. Use 'Process All Videos for ML Analysis' to rebuild semantic search")
    else:
        print("\n⚠️ Some operations failed. Check the errors above.")
        
        if not chroma_success:
            print("\n💡 ChromaDB Clearing Failed:")
            print("1. Stop the PABT server (Ctrl+C in the terminal running it)")
            print("2. Run this script again to clear ChromaDB")
            print("3. Restart the server after clearing")
        else:
            print("You may need to manually delete remaining files.")
        
        # Show what was successfully cleared
        cleared_items = []
        if sqlite_success:
            cleared_items.append("SQLite database")
        if chroma_success:
            cleared_items.append("ChromaDB")
        if ml_success:
            cleared_items.append("ML analysis data")
        if transcoded_success:
            cleared_items.append("Transcoded media directory")
        if previews_success:
            cleared_items.append("Video previews directory")
        if thumbnails_success:
            cleared_items.append("Video thumbnails directory")
        
        if cleared_items:
            print(f"\n✅ Successfully cleared: {', '.join(cleared_items)}")

if __name__ == "__main__":
    main() 