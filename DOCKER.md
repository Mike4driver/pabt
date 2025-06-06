# Docker Setup for PABT

This guide explains how to run PABT (Personal Audio/Video Browser and Transcoding Toolkit) using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (usually included with Docker Desktop)

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository and navigate to the project directory:**
   ```bash
   git clone <repository-url>
   cd pabt
   ```

2. **Create necessary directories:**
   ```bash
   mkdir -p media media_transcoded ml_analysis data static/previews static/thumbnails chroma_db
   ```

3. **Build and start the application:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
   Open your browser and go to `http://localhost:8000`

### Using Docker directly

1. **Build the Docker image:**
   ```bash
   docker build -t pabt .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name pabt-app \
     -p 8000:8000 \
     -v $(pwd)/media:/app/media \
     -v $(pwd)/media_transcoded:/app/media_transcoded \
     -v $(pwd)/ml_analysis:/app/ml_analysis \
     -v $(pwd)/static/previews:/app/static/previews \
     -v $(pwd)/static/thumbnails:/app/static/thumbnails \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/chroma_db:/app/chroma_db \
     pabt
   ```

## Volume Mounts

The Docker setup uses several volume mounts for persistent data:

- `./media` - Your source media files
- `./media_transcoded` - Transcoded video files
- `./ml_analysis` - ML analysis results and frame data
- `./static/previews` - Generated video previews
- `./static/thumbnails` - Generated thumbnails
- `./data` - Database and application data
- `./chroma_db` - ChromaDB vector database storage

## Configuration

### Environment Variables

You can customize the application behavior using environment variables:

```yaml
environment:
  - PYTHONUNBUFFERED=1
  - LOG_LEVEL=INFO
```

### Media Directory

Place your media files in the `./media` directory on the host system. The application will automatically scan and index these files.

## Development

### Development with Docker

For development, you can mount the source code as a volume:

```bash
docker run -d \
  --name pabt-dev \
  -p 8000:8000 \
  -v $(pwd):/app \
  -v $(pwd)/media:/app/media \
  pabt
```

### Rebuilding

When you make changes to the code or dependencies:

```bash
docker-compose down
docker-compose up --build
```

## Production Deployment

### Using Docker Compose with Nginx (Optional)

Uncomment the nginx service in `docker-compose.yml` and create an `nginx.conf` file for production deployment with SSL termination.

### Resource Limits

For production, consider adding resource limits:

```yaml
services:
  pabt:
    # ... other configuration
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

## Troubleshooting

### Common Issues

1. **Permission Issues:**
   ```bash
   sudo chown -R $USER:$USER media media_transcoded ml_analysis data static
   ```

2. **Port Already in Use:**
   Change the port mapping in docker-compose.yml:
   ```yaml
   ports:
     - "8080:8000"  # Use port 8080 instead
   ```

3. **FFmpeg Issues:**
   The Docker image includes FFmpeg. If you encounter issues, check the container logs:
   ```bash
   docker-compose logs pabt
   ```

### Logs

View application logs:
```bash
docker-compose logs -f pabt
```

### Health Check

Check if the application is healthy:
```bash
docker-compose ps
```

## Backup

To backup your data:
```bash
tar -czf pabt-backup-$(date +%Y%m%d).tar.gz media data static/previews static/thumbnails ml_analysis chroma_db
```

## Updates

To update to a new version:
```bash
git pull
docker-compose down
docker-compose up --build
```

## Security Notes

- The application runs as a non-root user inside the container
- Media files are mounted as volumes, not copied into the image
- Consider using a reverse proxy (nginx) for production deployments
- Regularly update the base Docker image for security patches 