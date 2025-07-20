# AI Security Worker

This project provides a background worker service designed for processing security camera footage. It uses FastAPI for the web framework and Celery for asynchronous task processing. The service can perform two main functions:

1.  **Resident Enrollment**: Scans videos to detect and save unique faces for building a dataset of known residents.
2.  **Security Monitoring**: Analyzes videos to detect strangers in predefined zones and sends alerts if rules are violated.

---

## Architecture

-   **FastAPI**: Provides a basic web server and health check endpoints.
-   **Celery**: Manages the asynchronous processing of long-running video analysis tasks.
-   **Redis**: Acts as the message broker and backend for Celery, queuing tasks sent from your main application (e.g., a Django backend).
-   **Docker**: Containerizes the entire service, including all dependencies and models, for consistent and easy deployment.
-   **YOLOv8**: A fast and accurate object detection model used to find people in video frames.
-   **face_recognition**: A library for finding and encoding human faces.

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
  -v /Users/tahakhan/Desktop/ring_camera:/app/ring_camera \
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

For local development, you have two options: run the service directly or use Docker.

### Option A: Run Service Directly (No Docker)

```bash
# Activate virtual environment and install dependencies
source .venv/bin/activate
uv pip install -r requirements.txt

# Set environment for local development
export ENV=local

# Start the FastAPI server
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Option B: Run Service with Docker

**For development endpoints (no Redis/Celery needed):**
```bash
# Build the image
docker build -t turing .

# Run just the FastAPI server
docker run --rm --name turing-face-worker \
  -e ENV="local" \
  -p 8000:8000 \
  turing uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

**For full setup with Redis/Celery (production endpoints):**
```bash
# Build the image
docker build -t turing .

# Start Redis
docker run -d --name redis-server -p 6379:6379 redis

# Run the worker container
docker run --rm --name turing-face-worker \
  -e REDIS_URL="redis://host.docker.internal:6379/0" \
  -e ENV="local" \
  -p 8000:8000 \
  turing
```

### 2. Use the Client

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