# Home Area Segmentation API

A comprehensive segmentation API that can detect and label different areas around homes (backyard, pool, garage, roads, etc.) using AI models. The system allows users to verify segmentations, manage resident permissions, and control access to specific areas based on conditions.

## Features

### 🏠 Area Detection & Segmentation
- **Automated Detection**: Detects various home areas including:
  - Backyard, Pool, Garage, Road, Driveway
  - Front yard, Lawn, Patio, Deck, Fence
- **Precise Segmentation**: Uses Grounded-SAM for accurate area boundaries
- **Dimensional Analysis**: Calculates real-world dimensions (width, height, area in meters)
- **Polygon Generation**: Converts masks to precise polygon coordinates

### 👤 User Verification System
- **Manual Verification**: Users can approve/reject detected segments
- **Visual Confirmation**: Generate visualization images with segmented polygons
- **User Association**: Each segment is tied to a specific user account
- **Privacy Protection**: Users can only access their own segments

### 🔐 Access Control & Permissions
- **Resident Management**: Assign specific residents to areas
- **Conditional Access**: Support for complex permission rules:
  - Adult supervision required (e.g., pool area for children)
  - Daylight only access
  - Weekend-only permissions
- **Real-time Access Checks**: Validate resident access in real-time
- **Context-aware Decisions**: Consider environmental factors (time, weather, supervision)

## Architecture

### Core Components

1. **SegmentationService** (`service.py`)
   - Handles AI model inference (Grounding DINO + SAM)
   - Image processing and area detection
   - Mask-to-polygon conversion
   - Dimension calculations

2. **SegmentManager** (`segment_manager.py`)
   - Manages segment storage and retrieval
   - User verification workflows
   - Permission management
   - Access control logic

3. **API Router** (`router.py`)
   - RESTful API endpoints
   - File upload handling
   - Response formatting
   - Error handling

4. **Data Models** (`schemas.py`)
   - Pydantic models for type safety
   - Request/response validation
   - Data serialization

5. **Enums** (`enums.py`)
   - Area type definitions
   - Permission condition types

## API Endpoints

### Segmentation
- `POST /segmentation/segment` - Segment areas in an image
- `POST /segmentation/segment/upload` - Upload and segment image
- `POST /segmentation/visualization` - Generate visualization image

### Verification
- `POST /segmentation/segment/verify` - Verify/reject segmented areas
- `GET /segmentation/segments/user/{user_id}` - Get user's segments

### Access Control
- `POST /segmentation/permission/add` - Add resident permissions
- `POST /segmentation/access/check` - Check resident access
- `GET /segmentation/permissions/resident/{resident_id}` - Get resident permissions

### Health & Status
- `GET /segmentation/health` - Service health check

## Installation & Setup

### Prerequisites
- Python 3.10.11
- Docker
- GPU support (recommended for performance)

### Using Docker (Recommended)

1. **Build the image:**
   ```bash
   docker build -t turing-service .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 turing-service
   ```

The service includes both face recognition and segmentation capabilities in a single container. The segmentation API will be available at `/segmentation/*` endpoints.

### Manual Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Clone Grounded-SAM:**
   ```bash
   git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
   cd Grounded-Segment-Anything
   pip install -e .
   pip install -e segment_anything
   pip install -e GroundingDINO
   ```

3. **Download models:**
   ```bash
   mkdir -p segmentation_models
   
   # GroundingDINO model
   wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
        -O segmentation_models/groundingdino_swint_ogc.pth
   
   # SAM model
   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
        -O segmentation_models/sam_vit_h_4b8939.pth
   ```

4. **Set environment variables:**
   ```bash
   export PYTHONPATH="Grounded-Segment-Anything:Grounded-Segment-Anything/GroundingDINO:$PYTHONPATH"
   ```

5. **Run the application:**
   ```bash
   uvicorn src.app:app --host 0.0.0.0 --port 8000
   ```

## Usage Examples

### 1. Segment an Image

```python
import requests

# Upload and segment an image
with open("home_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/segmentation/segment/upload",
        params={"user_id": "user123"},
        files={"file": f}
    )

result = response.json()
print(f"Found {len(result['segments'])} areas")
```

### 2. Verify Segments

```python
# Approve a detected segment
response = requests.post(
    "http://localhost:8000/segmentation/segment/verify",
    json={
        "area_id": "segment_id_here",
        "user_id": "user123",
        "approved": True
    }
)
```

### 3. Add Resident Permissions

```python
# Allow child in backyard only with adult supervision
response = requests.post(
    "http://localhost:8000/segmentation/permission/add",
    json={
        "resident_id": "child_001",
        "area_id": "backyard_segment_id",
        "allowed": True,
        "conditions": ["adult_supervision_required"]
    }
)
```

### 4. Check Access

```python
# Check if resident can access area
response = requests.post(
    "http://localhost:8000/segmentation/access/check",
    json={
        "resident_id": "child_001",
        "area_id": "pool_segment_id",
        "context": {
            "adult_present": True,
            "daylight": True
        }
    }
)

access_result = response.json()
print(f"Access: {access_result['allowed']}, Reason: {access_result['reason']}")
```

## Security Considerations

### Data Privacy
- Each user can only access their own segments
- Segment verification required before use
- No cross-user data sharing

### Access Control
- Multi-layered permission system
- Context-aware access decisions
- Audit trail for access attempts

### Model Security
- SHA256 verification for model weights
- Pinned dependency versions
- Containerized deployment

## Performance Optimization

### GPU Acceleration
- CUDA support for faster inference
- Optimized model loading
- Batch processing capabilities

### Caching
- Model weights cached in container
- Processed segments stored in memory
- Efficient polygon operations

### Error Handling
- Graceful degradation on model failures
- Comprehensive logging
- User-friendly error messages

## Limitations & Considerations

### Detection Accuracy
- Performance depends on image quality
- May require fine-tuning for specific environments
- Lighting conditions can affect results

### Real-world Calibration
- Dimension calculations require calibration
- Default: 50 pixels per meter (adjustable)
- Consider camera angle and distance

### Resource Requirements
- GPU recommended for optimal performance
- Large model files (~3GB total)
- Memory intensive for high-resolution images

## Contributing

1. Follow existing code structure
2. Add comprehensive tests
3. Update documentation
4. Ensure Docker compatibility
5. Maintain security best practices

## License

This project is part of the Turing AI Security System.

## Support

For issues and questions:
- Check the health endpoint: `GET /segmentation/health`
- Review logs for detailed error information
- Ensure all model files are properly downloaded
- Verify CUDA setup for GPU acceleration