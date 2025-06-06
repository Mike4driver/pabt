# Makefile for PABT Media Library

.PHONY: help install install-dev test test-unit test-integration test-selenium test-coverage clean format lint docker-test setup

# Default target
help:
	@echo "🏠 PABT Media Library - Available Commands"
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  make install         Install production dependencies"
	@echo "  make install-dev     Install development dependencies"
	@echo "  make setup           Complete project setup"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-integration  Run integration tests only"
	@echo "  make test-selenium  Run Selenium tests only"
	@echo "  make test-coverage  Run tests with coverage report"
	@echo "  make test-fast      Run tests (skip slow tests)"
	@echo "  make test-parallel  Run tests in parallel"
	@echo ""
	@echo "🔍 Code Quality:"
	@echo "  make lint           Run code linting"
	@echo "  make format         Format code with black"
	@echo "  make type-check     Run type checking with mypy"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  make clean          Clean build artifacts and caches"
	@echo "  make clean-test     Clean test artifacts"
	@echo "  make clean-all      Clean everything"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-test    Run tests in Docker"
	@echo "  make docker-run     Run application in Docker"
	@echo ""
	@echo "🚀 Development:"
	@echo "  make dev            Start development server"
	@echo "  make docs           Generate documentation"
	@echo "  make security       Run security checks"

# Installation targets
install:
	@echo "📦 Installing production dependencies..."
	pip install -r requirements.txt

install-dev:
	@echo "🔧 Installing development dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-test.txt
	pip install black flake8 mypy isort bandit

setup: install-dev
	@echo "🚀 Setting up project..."
	python database.py
	@echo "✅ Project setup complete!"

# Testing targets
test:
	@echo "🧪 Running all tests..."
	python run_tests.py

test-unit:
	@echo "🔬 Running unit tests..."
	python run_tests.py --unit

test-integration:
	@echo "🔗 Running integration tests..."
	python run_tests.py --integration

test-selenium:
	@echo "🌐 Running Selenium tests..."
	python run_tests.py --selenium

test-coverage:
	@echo "📊 Running tests with coverage..."
	python run_tests.py --coverage
	@echo "📋 Coverage report: htmlcov/index.html"

test-fast:
	@echo "⚡ Running fast tests..."
	python run_tests.py --fast

test-parallel:
	@echo "🏃‍♂️ Running tests in parallel..."
	python run_tests.py --parallel

test-ml:
	@echo "🤖 Running ML tests..."
	python run_tests.py --markers "requires_ml"

test-no-ml:
	@echo "🚫 Running tests without ML..."
	pytest -m "not requires_ml"

# Code quality targets
lint:
	@echo "🔍 Running code linting..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	@echo "🎨 Formatting code..."
	black .
	isort .

type-check:
	@echo "🔍 Running type checking..."
	mypy . --ignore-missing-imports

security:
	@echo "🔒 Running security checks..."
	bandit -r . -x tests/

# Cleanup targets
clean:
	@echo "🧹 Cleaning build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.db" -delete

clean-test:
	@echo "🧪 Cleaning test artifacts..."
	rm -rf htmlcov/
	rm -rf reports/
	rm -rf .pytest_cache/
	rm -rf .coverage
	find . -name "test_*.db" -delete

clean-all: clean clean-test
	@echo "🗑️  Deep cleaning..."
	rm -rf dist/
	rm -rf build/
	rm -rf chroma_db/
	rm -rf media_transcoded/
	find . -name "*.log" -delete

# Docker targets
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t pabt-media-library .

docker-test:
	@echo "🐳 Running tests in Docker..."
	docker build -f Dockerfile.test -t pabt-test .
	docker run --rm pabt-test

docker-run:
	@echo "🐳 Running application in Docker..."
	docker-compose up

# Development targets
dev:
	@echo "🚀 Starting development server..."
	python main.py

dev-reload:
	@echo "🔄 Starting development server with auto-reload..."
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

docs:
	@echo "📚 Generating documentation..."
	# Add documentation generation commands here

# Database management
db-init:
	@echo "🗄️  Initializing database..."
	python database.py

db-reset:
	@echo "🔄 Resetting database..."
	rm -f media_library.db
	python database.py

db-backup:
	@echo "💾 Backing up database..."
	cp media_library.db media_library_backup_$(shell date +%Y%m%d_%H%M%S).db

# Media processing
scan-media:
	@echo "📁 Scanning media directory..."
	python -c "from data_access import scan_media_directory_and_update_db; scan_media_directory_and_update_db()"

# Performance testing
test-load:
	@echo "⚡ Running load tests..."
	# Add load testing commands here

# Dependency management
update-deps:
	@echo "📦 Updating dependencies..."
	pip list --outdated
	@echo "Run 'pip install --upgrade <package>' to update specific packages"

freeze-deps:
	@echo "❄️  Freezing current dependencies..."
	pip freeze > requirements-frozen.txt

# Git hooks
install-hooks:
	@echo "🪝 Installing git hooks..."
	echo "#!/bin/sh\nmake lint\nmake test-unit" > .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit

# Monitoring and health checks
health-check:
	@echo "🏥 Running health checks..."
	python -c "import requests; print('✅ Server healthy' if requests.get('http://localhost:8000/').status_code == 200 else '❌ Server not responding')"

logs:
	@echo "📋 Showing application logs..."
	tail -f *.log 2>/dev/null || echo "No log files found"

# CI/CD helpers
ci-test:
	@echo "🤖 Running CI test suite..."
	python run_tests.py --unit --coverage --html-report

ci-build:
	@echo "🏗️  Building for CI/CD..."
	make clean
	make install
	make test-unit

# Release management
version:
	@echo "📋 Current version info:"
	@python -c "import sys; print(f'Python: {sys.version}')"
	@python -c "try: import fastapi; print(f'FastAPI: {fastapi.__version__}'); except: print('FastAPI: Not installed')"

release-check:
	@echo "🔍 Pre-release checks..."
	make clean
	make lint
	make type-check
	make security
	make test-coverage
	@echo "✅ Ready for release!"

# Performance profiling
profile:
	@echo "📊 Profiling application..."
	python -m cProfile -o profile.stats main.py
	@echo "Profile saved to profile.stats"

# SSL and security
ssl-cert:
	@echo "🔐 Generating SSL certificate..."
	openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Backup and restore
backup:
	@echo "💾 Creating backup..."
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		--exclude='__pycache__' \
		--exclude='*.pyc' \
		--exclude='.git' \
		--exclude='htmlcov' \
		--exclude='reports' \
		.

restore:
	@echo "📥 Restore from backup:"
	@ls -la backup_*.tar.gz 2>/dev/null || echo "No backup files found"

# Quick shortcuts
quick-test: test-unit
install-all: install-dev setup
full-test: clean test-coverage
dev-setup: install-dev db-init
check: lint type-check security test-unit