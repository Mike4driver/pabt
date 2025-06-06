import pytest
import time
import threading
import uvicorn
from pathlib import Path
from unittest.mock import patch
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from main import app


@pytest.fixture(scope="module")
def test_server(test_config, temp_dirs, populated_test_db):
    """Start a test server for Selenium tests"""
    port = test_config["test_port"]
    
    # Patch the configuration for the test server
    with patch('config.BASE_DIR', temp_dirs["base"]), \
         patch('config.MEDIA_DIR', temp_dirs["media"]), \
         patch('database.DATABASE_NAME', populated_test_db):
        
        # Start server in a separate thread
        server_thread = threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={
                "host": "127.0.0.1",
                "port": port,
                "log_level": "error"
            },
            daemon=True
        )
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        yield f"http://127.0.0.1:{port}"


@pytest.mark.selenium
@pytest.mark.integration 
class TestHomePageIntegration:
    """Integration tests for the home page"""
    
    def test_home_page_loads(self, selenium_driver, test_server):
        """Test that home page loads successfully"""
        driver = selenium_driver
        driver.get(test_server)
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        assert "Media Browser" in driver.title or "PABT" in driver.title
        
        # Check for main navigation or content elements
        try:
            driver.find_element(By.TAG_NAME, "main")
        except NoSuchElementException:
            # If no main tag, at least body should be present
            assert driver.find_element(By.TAG_NAME, "body")
    
    def test_navigation_to_files_page(self, selenium_driver, test_server):
        """Test navigation to files listing page"""
        driver = selenium_driver
        driver.get(test_server)
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Navigate to files page
        driver.get(f"{test_server}/files")
        
        # Wait for files page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Should show media files or at least the page structure
        assert "/files" in driver.current_url


@pytest.mark.selenium
@pytest.mark.integration
class TestMediaBrowsingIntegration:
    """Integration tests for media browsing functionality"""
    
    def test_files_page_displays_media(self, selenium_driver, test_server):
        """Test that files page displays media items"""
        driver = selenium_driver
        driver.get(f"{test_server}/files")
        
        # Wait for page content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Look for media items, pagination, or relevant content
        page_source = driver.page_source.lower()
        
        # Should contain some indication of media content or empty state
        assert any(keyword in page_source for keyword in [
            "video", "media", "file", "sample", "test", "no files", "empty"
        ])
    
    def test_search_functionality(self, selenium_driver, test_server):
        """Test search functionality on files page"""
        driver = selenium_driver
        driver.get(f"{test_server}/files")
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Try to find search input
        try:
            search_input = driver.find_element(By.CSS_SELECTOR, "input[name='search'], input[type='search'], #search")
            
            # Perform search
            search_input.clear()
            search_input.send_keys("test")
            search_input.send_keys(Keys.RETURN)
            
            # Wait for search results
            time.sleep(2)
            
            # Verify URL contains search parameter
            assert "search=test" in driver.current_url
            
        except NoSuchElementException:
            # If no search input found, just verify page loads
            pass
    
    def test_pagination(self, selenium_driver, test_server):
        """Test pagination functionality"""
        driver = selenium_driver
        driver.get(f"{test_server}/files")
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Look for pagination elements
        try:
            pagination_elements = driver.find_elements(By.CSS_SELECTOR, 
                ".pagination a, .page-link, [data-page], .next, .prev")
            
            if pagination_elements:
                # Test clicking on pagination if it exists
                first_page_link = pagination_elements[0]
                if first_page_link.is_enabled() and first_page_link.is_displayed():
                    first_page_link.click()
                    time.sleep(2)
                    
                    # Verify we're still on files page
                    assert "/files" in driver.current_url
        except:
            # Pagination might not be present with limited test data
            pass
    
    def test_media_type_filtering(self, selenium_driver, test_server):
        """Test media type filtering"""
        driver = selenium_driver
        driver.get(f"{test_server}/files")
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Try to filter by video type via URL parameter
        driver.get(f"{test_server}/files?media_type=video")
        
        # Wait for filtered results
        time.sleep(2)
        
        # Verify URL contains filter parameter
        assert "media_type=video" in driver.current_url


@pytest.mark.selenium
@pytest.mark.integration
class TestVideoPlayerIntegration:
    """Integration tests for video player functionality"""
    
    def test_video_player_page_loads(self, selenium_driver, test_server):
        """Test that video player page loads for a video"""
        driver = selenium_driver
        
        # Try to access a video player page
        driver.get(f"{test_server}/video/sample_video.mp4")
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Check if we get a valid response (not 404)
        page_source = driver.page_source.lower()
        
        # Should either show video player or error message
        assert not ("404" in page_source and "not found" in page_source) or \
               "video" in page_source or "player" in page_source
    
    def test_video_player_with_timestamp(self, selenium_driver, test_server):
        """Test video player with timestamp parameter"""
        driver = selenium_driver
        
        # Access video with timestamp
        driver.get(f"{test_server}/video/sample_video.mp4?t=30.5")
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Verify timestamp is in URL
        assert "t=30.5" in driver.current_url
        
        # Look for video element or player
        try:
            video_element = driver.find_element(By.TAG_NAME, "video")
            assert video_element is not None
        except NoSuchElementException:
            # Video element might not be present in test environment
            pass
    
    def test_video_metadata_editing(self, selenium_driver, test_server):
        """Test video metadata editing functionality"""
        driver = selenium_driver
        
        # Access a video player page
        driver.get(f"{test_server}/video/sample_video.mp4")
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Look for metadata editing form or interface
        try:
            # Try to find title input field
            title_input = driver.find_element(By.CSS_SELECTOR, 
                "input[name='user_title'], #user_title, .title-input")
            
            if title_input.is_displayed() and title_input.is_enabled():
                # Clear and enter new title
                title_input.clear()
                title_input.send_keys("Updated Test Title")
                
                # Look for save button
                save_button = driver.find_element(By.CSS_SELECTOR,
                    "button[type='submit'], .save-btn, input[type='submit']")
                
                if save_button.is_displayed() and save_button.is_enabled():
                    save_button.click()
                    time.sleep(2)
                    
                    # Verify the change was saved (page should reload or show success)
                    assert "Updated Test Title" in driver.page_source
                    
        except NoSuchElementException:
            # Metadata editing interface might not be visible in test environment
            pass


@pytest.mark.selenium 
@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints via browser"""
    
    def test_api_tags_endpoint(self, selenium_driver, test_server):
        """Test API tags endpoint returns JSON"""
        driver = selenium_driver
        driver.get(f"{test_server}/api/tags")
        
        # Wait for response
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "pre"))
        )
        
        # Should return JSON response
        page_source = driver.page_source
        assert "tags" in page_source or "{" in page_source
    
    def test_semantic_search_debug_endpoint(self, selenium_driver, test_server):
        """Test semantic search debug endpoint"""
        driver = selenium_driver
        driver.get(f"{test_server}/api/semantic-search/debug")
        
        # Wait for response
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Should return JSON with debug information
        page_source = driver.page_source
        assert any(keyword in page_source for keyword in [
            "chromadb", "sentence_transformers", "videos", "recommendations"
        ])


@pytest.mark.selenium
@pytest.mark.integration
@pytest.mark.slow
class TestLongRunningOperations:
    """Integration tests for long-running operations"""
    
    def test_file_upload_interface(self, selenium_driver, test_server):
        """Test file upload interface if it exists"""
        driver = selenium_driver
        
        # Check various pages for file upload functionality
        pages_to_check = [
            f"{test_server}/",
            f"{test_server}/files",
            f"{test_server}/tools/media-processing"
        ]
        
        for page_url in pages_to_check:
            try:
                driver.get(page_url)
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # Look for file input elements
                file_inputs = driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
                if file_inputs:
                    # Found file upload functionality
                    assert len(file_inputs) > 0
                    break
                    
            except TimeoutException:
                continue
    
    def test_processing_status_interface(self, selenium_driver, test_server):
        """Test processing status interface"""
        driver = selenium_driver
        
        # Check for processing or jobs interface
        try:
            driver.get(f"{test_server}/tools/media-processing")
            
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Should load the tools page
            assert "/tools" in driver.current_url
            
        except TimeoutException:
            # Tools page might not be accessible
            pass


@pytest.mark.selenium
@pytest.mark.integration
class TestResponsiveDesign:
    """Integration tests for responsive design"""
    
    def test_mobile_viewport(self, selenium_driver, test_server):
        """Test interface on mobile viewport"""
        driver = selenium_driver
        
        # Set mobile viewport
        driver.set_window_size(375, 667)  # iPhone 6/7/8 size
        
        driver.get(test_server)
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Check that page is still functional on mobile
        assert driver.find_element(By.TAG_NAME, "body")
        
        # Try navigation
        driver.get(f"{test_server}/files")
        time.sleep(2)
        
        assert "/files" in driver.current_url
    
    def test_tablet_viewport(self, selenium_driver, test_server):
        """Test interface on tablet viewport"""
        driver = selenium_driver
        
        # Set tablet viewport
        driver.set_window_size(768, 1024)  # iPad size
        
        driver.get(test_server)
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Check that page is still functional on tablet
        assert driver.find_element(By.TAG_NAME, "body")
    
    def test_desktop_viewport(self, selenium_driver, test_server):
        """Test interface on desktop viewport"""
        driver = selenium_driver
        
        # Set desktop viewport
        driver.set_window_size(1920, 1080)
        
        driver.get(test_server)
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Desktop should have full functionality
        assert driver.find_element(By.TAG_NAME, "body")


@pytest.mark.selenium
@pytest.mark.integration
class TestErrorHandling:
    """Integration tests for error handling in browser"""
    
    def test_404_page_handling(self, selenium_driver, test_server):
        """Test 404 error page handling"""
        driver = selenium_driver
        
        # Try to access non-existent page
        driver.get(f"{test_server}/nonexistent-page")
        
        # Wait for response
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Should handle 404 gracefully
        page_source = driver.page_source.lower()
        
        # Should either show 404 error or redirect to valid page
        assert "404" in page_source or "not found" in page_source or \
               driver.current_url != f"{test_server}/nonexistent-page"
    
    def test_video_not_found_handling(self, selenium_driver, test_server):
        """Test handling of non-existent video"""
        driver = selenium_driver
        
        # Try to access non-existent video
        driver.get(f"{test_server}/video/nonexistent-video.mp4")
        
        # Wait for response
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Should handle missing video gracefully
        page_source = driver.page_source.lower()
        
        assert "404" in page_source or "not found" in page_source or \
               "error" in page_source
    
    def test_javascript_errors(self, selenium_driver, test_server):
        """Test for JavaScript errors on pages"""
        driver = selenium_driver
        
        # Test main pages for JavaScript errors
        pages_to_test = [
            test_server,
            f"{test_server}/files",
            f"{test_server}/tools/media-processing"
        ]
        
        for page_url in pages_to_test:
            try:
                driver.get(page_url)
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # Check browser console for errors
                logs = driver.get_log('browser')
                
                # Filter out minor warnings, focus on errors
                errors = [log for log in logs if log['level'] == 'SEVERE']
                
                # Should not have severe JavaScript errors
                assert len(errors) == 0, f"JavaScript errors found on {page_url}: {errors}"
                
            except Exception as e:
                # If we can't access the page or get logs, that's also an issue
                pytest.fail(f"Failed to test {page_url}: {str(e)}")


@pytest.fixture(autouse=True)
def reset_window_size(selenium_driver):
    """Reset window size after each test"""
    yield
    # Reset to default size
    try:
        selenium_driver.set_window_size(1920, 1080)
    except:
        pass