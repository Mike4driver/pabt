import pytest
import sqlite3
from unittest.mock import patch, mock_open
import json
from pathlib import Path

from database import (
    get_db_connection, create_tables, get_setting, update_setting, db_connection
)


@pytest.mark.unit
class TestDatabase:
    """Unit tests for database functionality"""
    
    def test_get_db_connection(self, test_db):
        """Test database connection creation"""
        with patch('database.DATABASE_NAME', test_db):
            conn = get_db_connection()
            assert conn is not None
            assert isinstance(conn, sqlite3.Connection)
            
            # Test row factory is set
            cursor = conn.cursor()
            cursor.execute("SELECT 1 as test_col")
            row = cursor.fetchone()
            assert row['test_col'] == 1
            conn.close()
    
    def test_db_connection_context_manager(self, test_db):
        """Test database context manager"""
        with patch('database.DATABASE_NAME', test_db):
            with db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO settings (key, value) VALUES (?, ?)", 
                              ("test_key", "test_value"))
                # Should auto-commit on success
            
            # Verify the insert was committed
            with db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM settings WHERE key = ?", ("test_key",))
                result = cursor.fetchone()
                assert result['value'] == "test_value"
    
    def test_db_connection_rollback_on_exception(self, test_db):
        """Test database rollback on exception"""
        with patch('database.DATABASE_NAME', test_db):
            try:
                with db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO settings (key, value) VALUES (?, ?)", 
                                  ("rollback_test", "should_not_exist"))
                    # Force an exception
                    raise ValueError("Test exception")
            except ValueError:
                pass
            
            # Verify the insert was rolled back
            with db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM settings WHERE key = ?", ("rollback_test",))
                result = cursor.fetchone()
                assert result is None
    
    def test_create_tables(self, temp_dirs):
        """Test table creation"""
        db_path = temp_dirs["base"] / "test_create_tables.db"
        
        with patch('database.DATABASE_NAME', str(db_path)):
            create_tables()
            
            # Verify tables exist
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Check media_files table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='media_files'")
            assert cursor.fetchone() is not None
            
            # Check settings table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='settings'")
            assert cursor.fetchone() is not None
            
            # Check required columns in media_files
            cursor.execute("PRAGMA table_info(media_files)")
            columns = [col[1] for col in cursor.fetchall()]
            expected_columns = [
                'id', 'filename', 'original_path', 'media_type', 'user_title',
                'duration', 'width', 'height', 'fps', 'size_bytes', 'date_added',
                'last_scanned', 'thumbnail_path', 'has_specific_thumbnail',
                'transcoded_path', 'has_transcoded_version', 'preview_path',
                'has_preview', 'tags', 'metadata_json'
            ]
            for col in expected_columns:
                assert col in columns
            
            conn.close()
    
    def test_create_tables_with_config_file(self, temp_dirs):
        """Test table creation with existing config.json"""
        db_path = temp_dirs["base"] / "test_config.db"
        config_path = Path("config.json")
        
        # Create a mock config.json
        config_data = {"media_directory_name": "custom_media"}
        mock_file_content = json.dumps(config_data)
        
        with patch('database.DATABASE_NAME', str(db_path)), \
             patch('database.CONFIG_FILE.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=mock_file_content)):
            
            create_tables()
            
            # Verify the setting was inserted from config
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = 'media_directory_name'")
            result = cursor.fetchone()
            assert result['value'] == "custom_media"
            conn.close()
    
    def test_get_setting(self, test_db):
        """Test getting settings from database"""
        with patch('database.DATABASE_NAME', test_db):
            # Test existing setting
            result = get_setting("media_directory_name")
            assert result == "test_media"
            
            # Test non-existent setting
            result = get_setting("non_existent_key")
            assert result is None
    
    def test_update_setting(self, test_db):
        """Test updating settings in database"""
        with patch('database.DATABASE_NAME', test_db):
            # Update existing setting
            update_setting("media_directory_name", "updated_media")
            result = get_setting("media_directory_name")
            assert result == "updated_media"
            
            # Insert new setting
            update_setting("new_setting", "new_value")
            result = get_setting("new_setting")
            assert result == "new_value"
    
    def test_schema_migration_user_title(self, temp_dirs):
        """Test schema migration for user_title column"""
        db_path = temp_dirs["base"] / "test_migration.db"
        
        with patch('database.DATABASE_NAME', str(db_path)):
            # Create initial table without user_title
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE media_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL UNIQUE,
                    original_path TEXT NOT NULL,
                    media_type TEXT
                )
            """)
            conn.commit()
            conn.close()
            
            # Run create_tables which should add missing columns
            create_tables()
            
            # Verify user_title column was added
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(media_files)")
            columns = [col[1] for col in cursor.fetchall()]
            assert 'user_title' in columns
            assert 'tags' in columns
            conn.close()
    
    def test_schema_migration_tags(self, temp_dirs):
        """Test schema migration for tags column"""
        db_path = temp_dirs["base"] / "test_tags_migration.db"
        
        with patch('database.DATABASE_NAME', str(db_path)):
            # Create initial table without tags
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE media_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL UNIQUE,
                    original_path TEXT NOT NULL,
                    media_type TEXT,
                    user_title TEXT
                )
            """)
            conn.commit()
            conn.close()
            
            # Run create_tables which should add missing columns
            create_tables()
            
            # Verify tags column was added with default value
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(media_files)")
            columns = {col[1]: col for col in cursor.fetchall()}
            assert 'tags' in columns
            
            # Insert a record and verify default tags value
            cursor.execute("""
                INSERT INTO media_files (filename, original_path, media_type)
                VALUES (?, ?, ?)
            """, ("test.mp4", "/test/test.mp4", "video"))
            conn.commit()
            
            cursor.execute("SELECT tags FROM media_files WHERE filename = ?", ("test.mp4",))
            result = cursor.fetchone()
            # Default should be empty JSON array
            assert result['tags'] == '[]'
            
            conn.close()


@pytest.mark.unit 
class TestDatabaseIntegration:
    """Integration tests for database operations"""
    
    def test_full_media_file_lifecycle(self, test_db):
        """Test complete media file CRUD operations"""
        with patch('database.DATABASE_NAME', test_db):
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Create
            cursor.execute("""
                INSERT INTO media_files 
                (filename, original_path, media_type, user_title, duration, tags)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("test.mp4", "/test/test.mp4", "video", "Test Video", 120.0, '["test"]'))
            conn.commit()
            
            # Read
            cursor.execute("SELECT * FROM media_files WHERE filename = ?", ("test.mp4",))
            result = cursor.fetchone()
            assert result is not None
            assert result['filename'] == "test.mp4"
            assert result['user_title'] == "Test Video"
            assert result['duration'] == 120.0
            assert result['tags'] == '["test"]'
            
            media_id = result['id']
            
            # Update
            cursor.execute("""
                UPDATE media_files 
                SET user_title = ?, tags = ? 
                WHERE id = ?
            """, ("Updated Title", '["test", "updated"]', media_id))
            conn.commit()
            
            # Verify update
            cursor.execute("SELECT user_title, tags FROM media_files WHERE id = ?", (media_id,))
            result = cursor.fetchone()
            assert result['user_title'] == "Updated Title"
            assert result['tags'] == '["test", "updated"]'
            
            # Delete
            cursor.execute("DELETE FROM media_files WHERE id = ?", (media_id,))
            conn.commit()
            
            # Verify delete
            cursor.execute("SELECT * FROM media_files WHERE id = ?", (media_id,))
            result = cursor.fetchone()
            assert result is None
            
            conn.close()
    
    def test_media_file_constraints(self, test_db):
        """Test database constraints and validations"""
        with patch('database.DATABASE_NAME', test_db):
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Test unique filename constraint
            cursor.execute("""
                INSERT INTO media_files (filename, original_path, media_type)
                VALUES (?, ?, ?)
            """, ("unique_test.mp4", "/test/unique_test.mp4", "video"))
            conn.commit()
            
            # Try to insert duplicate filename
            with pytest.raises(sqlite3.IntegrityError):
                cursor.execute("""
                    INSERT INTO media_files (filename, original_path, media_type)
                    VALUES (?, ?, ?)
                """, ("unique_test.mp4", "/test/another_path.mp4", "video"))
                conn.commit()
            
            # Test media_type constraint
            with pytest.raises(sqlite3.IntegrityError):
                cursor.execute("""
                    INSERT INTO media_files (filename, original_path, media_type)
                    VALUES (?, ?, ?)
                """, ("invalid_type.mp4", "/test/invalid_type.mp4", "invalid_type"))
                conn.commit()
            
            conn.close()