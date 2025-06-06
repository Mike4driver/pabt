# Testing Documentation for PABT Media Library

This document describes the comprehensive testing suite for the PABT (Personal Archive Browser Tool) media library application.

## 📋 Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Setup](#setup)
- [Running Tests](#running-tests)
- [Test Types](#test-types)
- [Coverage](#coverage)
- [CI/CD Integration](#cicd-integration)
- [Writing Tests](#writing-tests)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

The testing suite provides comprehensive coverage for the PABT application including:

- **Unit Tests**: Test individual functions and classes in isolation
- **Integration Tests**: Test component interactions and API endpoints
- **End-to-End Tests**: Test complete user workflows using Selenium WebDriver
- **Performance Tests**: Test application performance under load
- **ML Tests**: Test machine learning functionality with mocked models

## 📁 Test Structure

```
tests/
├── conftest.py                 # Pytest configuration and fixtures
├── unit/                       # Unit tests
│   ├── test_database.py       # Database operations
│   ├── test_utils.py          # Utility functions
│   ├── test_routes.py         # FastAPI routes
│   └── test_ml_processing.py  # ML functionality
├── integration/                # Integration tests
│   └── test_selenium_integration.py  # Selenium E2E tests
└── fixtures/                   # Test data and fixtures
    └── sample_media/          # Sample media files for testing
```

## 🚀 Setup

### 1. Install Dependencies

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Or use the test runner to install dependencies
python run_tests.py --install-deps
```

### 2. Prerequisites

#### For Selenium Tests:
- Chrome or Chromium browser
- ChromeDriver (automatically managed by webdriver-manager)

#### For ML Tests:
- Optional: sentence-transformers library
- Optional: chromadb library

#### For FFmpeg Tests:
- FFmpeg installed and available in PATH

### 3. Environment Setup

The test suite automatically creates isolated test environments including:
- Temporary databases
- Temporary media directories
- Mock configurations

## 🏃 Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run with coverage
python run_tests.py --coverage

# Run specific test types
python run_tests.py --unit
python run_tests.py --integration
python run_tests.py --selenium
```

### Detailed Options

```bash
# Run tests with various options
python run_tests.py --unit --coverage --verbose
python run_tests.py --selenium --html-report
python run_tests.py --fast --parallel
python run_tests.py --pattern "test_database"
python run_tests.py --file tests/unit/test_utils.py
python run_tests.py --markers "requires_ml"
```

### Using pytest directly

```bash
# Basic pytest usage
pytest tests/

# Unit tests only
pytest -m unit

# Integration tests only  
pytest -m integration

# Selenium tests only
pytest -m selenium

# With coverage
pytest --cov=. --cov-report=html

# Parallel execution
pytest -n auto

# Verbose output
pytest -v

# Specific patterns
pytest -k "test_database"
```

## 🧪 Test Types

### Unit Tests (tests/unit/)

Test individual components in isolation:

- **Database Tests** (`test_database.py`)
  - Table creation and migration
  - CRUD operations
  - Settings management
  - Transaction handling

- **Utility Tests** (`test_utils.py`)
  - Slugify functions
  - Path security
  - Media type detection
  - File operations

- **Route Tests** (`test_routes.py`)
  - FastAPI endpoint functionality
  - Request/response handling
  - Error handling
  - Authentication (if implemented)

- **ML Processing Tests** (`test_ml_processing.py`)
  - Model loading and encoding
  - ChromaDB operations
  - Semantic search functionality
  - Frame extraction

### Integration Tests (tests/integration/)

Test component interactions and full workflows:

- **Selenium Tests** (`test_selenium_integration.py`)
  - Complete user workflows
  - Browser compatibility
  - JavaScript functionality
  - Responsive design
  - Error handling in browser

### Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.selenium` - Selenium WebDriver tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_ml` - Tests requiring ML models
- `@pytest.mark.requires_ffmpeg` - Tests requiring FFmpeg

## 📊 Coverage

### Generating Coverage Reports

```bash
# HTML coverage report
python run_tests.py --coverage

# View coverage report
open htmlcov/index.html
```

### Coverage Goals

- **Overall Coverage**: >85%
- **Critical Modules**: >95%
  - Database operations
  - Security functions
  - API endpoints
- **ML Modules**: >80% (with mocked dependencies)

### Coverage Exclusions

- Third-party integrations
- Development utilities
- Platform-specific code

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run unit tests
      run: python run_tests.py --unit --coverage
    
    - name: Run integration tests
      run: python run_tests.py --integration
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
```

### Docker Testing

```dockerfile
FROM python:3.11

# Install Chrome for Selenium tests
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN echo "deb http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google.list
RUN apt-get update && apt-get install -y google-chrome-stable

# Install FFmpeg
RUN apt-get install -y ffmpeg

WORKDIR /app
COPY requirements*.txt ./
RUN pip install -r requirements.txt -r requirements-test.txt

COPY . .
RUN python run_tests.py --unit --integration
```

## ✍️ Writing Tests

### Test Structure Template

```python
import pytest
from unittest.mock import patch, Mock
from pathlib import Path

# Import your modules
from module_to_test import function_to_test

@pytest.mark.unit
class TestYourModule:
    """Unit tests for your module"""
    
    def test_basic_functionality(self):
        """Test basic functionality with valid input"""
        result = function_to_test("valid_input")
        assert result == "expected_output"
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        assert function_to_test("") == "default_output"
        assert function_to_test(None) is None
    
    def test_error_handling(self):
        """Test error handling"""
        with pytest.raises(ValueError):
            function_to_test("invalid_input")
    
    @patch('module_to_test.external_dependency')
    def test_with_mocking(self, mock_dependency):
        """Test with mocked external dependencies"""
        mock_dependency.return_value = "mocked_result"
        result = function_to_test("input")
        assert result == "expected_with_mock"
        mock_dependency.assert_called_once()
```

### Integration Test Template

```python
@pytest.mark.integration
class TestIntegration:
    """Integration tests"""
    
    def test_api_endpoint(self, client, populated_test_db):
        """Test API endpoint with real database"""
        response = client.get("/api/endpoint")
        assert response.status_code == 200
        assert "expected_data" in response.json()
    
    @pytest.mark.selenium
    def test_user_workflow(self, selenium_driver, test_server):
        """Test complete user workflow"""
        driver = selenium_driver
        driver.get(f"{test_server}/")
        
        # Perform user actions
        element = driver.find_element(By.ID, "search-input")
        element.send_keys("test query")
        element.submit()
        
        # Verify results
        assert "test" in driver.page_source
```

### Using Fixtures

```python
def test_with_custom_fixture(test_media_files, populated_test_db):
    """Test using custom fixtures"""
    # test_media_files provides sample media files
    # populated_test_db provides database with test data
    pass

@pytest.fixture
def custom_fixture():
    """Create custom test data"""
    data = create_test_data()
    yield data
    cleanup_test_data(data)
```

## 🐛 Troubleshooting

### Common Issues

#### 1. ChromeDriver Issues
```bash
# Update webdriver-manager
pip install --upgrade webdriver-manager

# Or install ChromeDriver manually
# Download from https://chromedriver.chromium.org/
```

#### 2. Database Lock Issues
```bash
# Clean up test databases
rm -f test_*.db

# Or use different test isolation
pytest --forked
```

#### 3. Port Conflicts (Selenium tests)
```bash
# Change test server port
export TEST_PORT=8002
python run_tests.py --selenium
```

#### 4. FFmpeg Not Found
```bash
# Install FFmpeg
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Or skip FFmpeg tests:
pytest -m "not requires_ffmpeg"
```

#### 5. ML Model Loading Issues
```bash
# Skip ML tests if models not available
pytest -m "not requires_ml"

# Or install ML dependencies
pip install sentence-transformers chromadb
```

### Debug Mode

```bash
# Run with debug output
python run_tests.py --verbose -s

# Run specific failing test
pytest tests/unit/test_database.py::TestDatabase::test_specific_method -v -s

# Drop into debugger on failure
pytest --pdb

# Run with coverage debug
pytest --cov-report=term-missing --cov-config=.coveragerc
```

### Performance Issues

```bash
# Run tests in parallel
python run_tests.py --parallel

# Skip slow tests
python run_tests.py --fast

# Profile test execution
pytest --durations=10
```

## 📝 Best Practices

1. **Test Organization**: Group related tests in classes
2. **Test Isolation**: Each test should be independent
3. **Descriptive Names**: Use clear, descriptive test names
4. **Mock External Dependencies**: Use mocks for external services
5. **Test Data**: Use fixtures for reusable test data
6. **Assertions**: Use specific assertions with clear messages
7. **Documentation**: Document complex test scenarios

## 🔗 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Selenium Documentation](https://selenium-python.readthedocs.io/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)

---

For questions about testing, please check the project issues or create a new issue with the "testing" label.