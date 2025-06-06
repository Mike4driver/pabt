#!/usr/bin/env python3
"""
Test runner script for PABT media library application.

This script provides convenient ways to run different types of tests:
- Unit tests only
- Integration tests only  
- All tests
- Tests with coverage reporting
- Selenium tests (requires Chrome/Chromium)
"""

import sys
import subprocess
import argparse
from pathlib import Path
import os


def install_test_dependencies():
    """Install test dependencies from requirements-test.txt"""
    print("📦 Installing test dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"
        ], check=True)
        print("✅ Test dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install test dependencies: {e}")
        return False


def check_chrome_available():
    """Check if Chrome/Chromium is available for Selenium tests"""
    chrome_commands = ['google-chrome', 'chromium', 'chromium-browser', 'chrome']
    
    for cmd in chrome_commands:
        try:
            subprocess.run([cmd, '--version'], 
                         capture_output=True, check=True, timeout=5)
            print(f"✅ Found Chrome/Chromium: {cmd}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    print("⚠️  Chrome/Chromium not found. Selenium tests will be skipped.")
    return False


def run_pytest(args, extra_args=None):
    """Run pytest with the given arguments"""
    cmd = [sys.executable, "-m", "pytest"] + args
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"🚀 Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run PABT tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--selenium", action="store_true", help="Run Selenium tests only")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--install-deps", action="store_true", help="Install test dependencies")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--html-report", action="store_true", help="Generate HTML test report")
    parser.add_argument("--pattern", "-k", help="Run tests matching pattern")
    parser.add_argument("--file", help="Run specific test file")
    parser.add_argument("--markers", help="Run tests with specific markers")
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_test_dependencies():
            return 1
    
    # Check if test directory exists
    if not Path("tests").exists():
        print("❌ Tests directory not found. Please run from project root.")
        return 1
    
    # Build pytest arguments
    pytest_args = []
    
    # Test selection
    if args.unit:
        pytest_args.extend(["-m", "unit"])
    elif args.integration:
        pytest_args.extend(["-m", "integration"])
    elif args.selenium:
        if not check_chrome_available():
            print("❌ Chrome/Chromium required for Selenium tests")
            return 1
        pytest_args.extend(["-m", "selenium"])
    
    # Skip slow tests if requested
    if args.fast:
        pytest_args.extend(["-m", "not slow"])
    
    # Coverage
    if args.coverage:
        pytest_args.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    
    # Parallel execution
    if args.parallel:
        try:
            import pytest_xdist
            pytest_args.extend(["-n", "auto"])
        except ImportError:
            print("⚠️  pytest-xdist not available. Running tests sequentially.")
    
    # Verbose output
    if args.verbose:
        pytest_args.append("-v")
    
    # HTML report
    if args.html_report:
        pytest_args.extend(["--html=reports/test_report.html", "--self-contained-html"])
        # Ensure reports directory exists
        Path("reports").mkdir(exist_ok=True)
    
    # Pattern matching
    if args.pattern:
        pytest_args.extend(["-k", args.pattern])
    
    # Specific file
    if args.file:
        pytest_args.append(args.file)
    
    # Markers
    if args.markers:
        pytest_args.extend(["-m", args.markers])
    
    # Default to all tests if no specific selection
    if not any([args.unit, args.integration, args.selenium, args.file, args.markers]):
        pytest_args.append("tests/")
    
    # Create necessary directories
    for directory in ["reports", "htmlcov"]:
        Path(directory).mkdir(exist_ok=True)
    
    print("🧪 Starting PABT test suite...")
    print(f"📂 Test directory: {Path('tests').absolute()}")
    
    # Run the tests
    success = run_pytest(pytest_args)
    
    if success:
        print("✅ All tests passed!")
        
        # Show coverage report location if generated
        if args.coverage and Path("htmlcov/index.html").exists():
            print(f"📊 Coverage report: {Path('htmlcov/index.html').absolute()}")
        
        # Show HTML report location if generated
        if args.html_report and Path("reports/test_report.html").exists():
            print(f"📋 Test report: {Path('reports/test_report.html').absolute()}")
        
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())