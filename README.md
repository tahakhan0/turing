# Turing AI - Home Security System

This project provides a comprehensive home security system with face recognition and area segmentation capabilities. It uses FastAPI for the web framework and includes web interfaces for easy interaction. The system can perform three main functions:

1.  **Face Recognition**: Detect and label people in security camera footage to build a resident database
2.  **Area Segmentation**: Automatically detect and segment different areas (backyard, pool, garage, etc.) in your property
3.  **Access Control**: Manage permissions for residents to access different areas with conditional rules

---

## Architecture

-   **FastAPI**: Provides REST API endpoints and serves web interfaces
-   **Face Recognition**: YOLOv8 + face_recognition library for detecting and recognizing people
-   **Area Segmentation**: Grounded-SAM (Grounding DINO + Segment Anything Model) for property area detection
-   **Persistent Storage**: File-based storage system that survives Docker container restarts
-   **Docker**: Containerizes the entire service with all AI models and dependencies
-   **Web Interfaces**: Interactive UIs for face labeling, area verification, and permission management

---

## Prerequisites

Before you begin, ensure you have the following installed:
-   [**Docker**](https://www.docker.com/get-started)
-   **Python 3.10.11**
-   [**uv**](https://github.com/astral-sh/uv) (A fast Python package installer)

---

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd turing
    ```

2.  **Install dependencies:**
    This project is built to run on Python 3.10.11. You can either use the existing virtual environment or let `uv` manage dependencies.

    **Option A: Using existing virtual environment (.venv):**
    ```bash
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```

    **Option B: Using uv's automatic environment management:**
    ```bash
    uv sync
    ```

---

## Docker Usage

This service is designed to be run as a Docker container.

### 1. Build the Docker Image

From the root of the project directory, run the build command:

```bash
docker build -t turing .
```

### 2. Run the Container

**Run the container with access to ring camera files:**
```bash
docker run --rm --name turing-service \
  -p 8000:8000 \
  -v /Users/tahakhan/Desktop/ring_camera:/app/videos \
  turing
```

This command:
- Maps port 8000 from the container to your local machine
- Mounts your ring camera folder at `/app/ring_camera` inside the container
- Starts the Turing FastAPI service

### 3. Legacy Celery Worker Setup (if needed)

To start the Celery worker, you need a running Redis instance.

**Start Redis (for local testing):**
```bash
docker run -d --name redis-server -p 6379:6379 redis
```

**Run the worker container:**
The worker needs to connect to Redis. The REDIS_URL environment variable is used for this.

```bash
docker run --rm --name turing-face-worker \
  -e REDIS_URL="redis://host.docker.internal:6379/0" \
  -e DJANGO_WEBHOOK_URL="http://host.docker.internal:8000/api/receive-alert/" \
  -e ENV="local" \
  turing
```

**Notes:**
- `--rm`: Automatically removes the container when it exits
- `host.docker.internal`: A special DNS name that allows the container to connect to services running on your host machine (like the local Redis or your Django dev server)
- `DJANGO_WEBHOOK_URL`: (Optional) Set this to the endpoint in your main application that should receive alerts
- `ENV=local`: Sets the service to local development mode for file-based face encoding storage

### 3. How to Trigger Tasks

From your main application (e.g., Django), you will use Celery to send tasks to the Redis queue, where this worker will pick them up.

```python
# In your main application's code

from celery import Celery

# Configure Celery to point to the same Redis broker
# This should be configured in your Django settings
celery_client = Celery('your-main-app', broker='redis://localhost:6379/0')

def trigger_enrollment_task(user_id, video_path):
    """Sends a task to scan a video for new faces."""
    payload = {
        "user_id": user_id,
        "video_path": video_path,  # Can be URL or local file path
    }
    celery_client.send_task('tasks.process_for_enrollment', args=[payload])
    print(f"Sent enrollment task for user {user_id}")

def trigger_monitoring_task(user_id, video_path, zones, rules):
    """Sends a task to monitor a video for rule violations."""
    payload = {
        "user_id": user_id,
        "video_path": video_path,  # Can be URL or local file path
        "zones": zones, # e.g., [{"name": "Pool", "points": [...]}]
        "rules": rules, # e.g., [{"zone_name": "Pool", ...}]
    }
    celery_client.send_task('tasks.process_for_monitoring', args=[payload])
    print(f"Sent monitoring task for user {user_id}")

# Example Usage:
# trigger_enrollment_task("user-abc-123", "https://example.com/video1.mp4")
# trigger_enrollment_task("user-abc-123", "/path/to/local/video.mp4")
```

---

## Local Development

### Quick Start with Docker Compose (Recommended)

The easiest way to run the service locally is using Docker Compose with persistent storage:

```bash
# Build and run in foreground (with logs visible in terminal)
docker-compose up --build

# Or run without rebuilding (if no code changes)
docker-compose up

# Run in background (detached from terminal)
docker-compose up -d --build
```

**What this does:**
- Builds the Docker image with all AI models
- Mounts your Ring camera videos from `/Users/tahakhan/Desktop/ring_camera/`
- Mounts persistent storage to `/Users/tahakhan/Desktop/home-security-db/`
- Starts the service on `http://localhost:8000`
- Keeps your face recognition labels and area segmentation data safe across restarts

**Stop the service:**
```bash
# Stop running containers
docker-compose down

# View logs of running service
docker-compose logs -f
```

### Docker Commands Reference

| Command | Build | Run | Foreground/Background |
|---------|-------|-----|---------------------|
| `docker-compose up --build` | ✅ Yes | ✅ Yes | 🖥️ Foreground |
| `docker-compose up` | ⚠️ Only if needed | ✅ Yes | 🖥️ Foreground |
| `docker-compose up -d --build` | ✅ Yes | ✅ Yes | 🔄 Background |
| `docker-compose up -d` | ⚠️ Only if needed | ✅ Yes | 🔄 Background |

### Alternative: Run Service Directly (No Docker)

```bash
# Activate virtual environment and install dependencies
source .venv/bin/activate
uv pip install -r requirements.txt

# Start the FastAPI server
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

### Web Interfaces

Once the service is running, access the web interfaces at:

- **Face Recognition Studio**: `http://localhost:8000/interfaces/face-recognition/`
  - Upload and analyze videos for face detection
  - Label detected faces to build your resident database
  - View analysis results and manage face encodings

- **Area Segmentation Studio**: `http://localhost:8000/interfaces/segmentation/`
  - Automatically detect areas in your property (backyard, pool, garage, etc.)
  - Verify and approve detected areas
  - Set up access permissions for labeled residents

### API Documentation

- **Interactive API Docs**: `http://localhost:8000/docs`
- **Face Recognition API**: `http://localhost:8000/face-recognition/*`
- **Segmentation API**: `http://localhost:8000/segmentation/*`

### Legacy Client Usage

The project includes a Python client (`src/app/client.py`) for easy interaction:

**Enroll faces from a video:**
```bash
python src/app/client.py --user-id user123 --video-path /path/to/video.mp4 --task enrollment
```

**Visualize YOLO detections (development mode):**
```bash
python src/app/client.py --user-id user123 --video-path /path/to/video.mp4 --visualize --output-dir ./visualizations
```

**List existing face encodings:**
```bash
python src/app/client.py --user-id user123 --video-path dummy --list-encodings
```

**Monitor a video for security violations:**
```bash
python src/app/client.py --user-id user123 --video-path /path/to/video.mp4 --task monitoring
```

### 3. Face Encodings Storage

In local development mode, face encodings are stored in the `face_encodings/` directory:
```
face_encodings/
├── user123/
│   ├── face_123456_0.json    # Face encoding data
│   ├── face_123456_0.png     # Face image
│   ├── face_789012_1.json
│   └── face_789012_1.png
└── user456/
    └── ...
```

### 4. Development vs Production Endpoints

- **Development endpoints** (`/development/*`): Run synchronously, return immediate results
- **Production endpoints** (`/tasks/*`): Use Celery for asynchronous processing

The client uses development endpoints by default. Use `--production-mode` flag to use production endpoints.

---