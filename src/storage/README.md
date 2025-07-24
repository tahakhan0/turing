# Persistent Storage Setup

This document explains how the Turing AI system uses persistent storage to keep your face recognition labels, encodings, and segmented areas data safe across Docker container restarts.

## Storage Location

All data is stored in: `/Users/tahakhan/Desktop/home-security-db/`

This directory contains:
- `face_recognition/` - Face labels and encodings for each user
- `segmentation/` - Segmented areas and permissions for each user  
- `uploads/` - Uploaded files organized by user

**Video Source**: `/Users/tahakhan/Desktop/ring_camera/` - Ring camera videos for analysis

## Directory Structure

```
/Users/tahakhan/Desktop/home-security-db/
├── face_recognition/
│   └── {user_id}/
│       ├── labels.json          # Face recognition labels
│       ├── encodings.pkl        # Face encodings for recognition
│       └── extracted_frames/    # Extracted video frames
│           └── frame_*.jpg
├── segmentation/
│   └── {user_id}/
│       └── segments.json        # Segmented areas and permissions
└── uploads/
    └── {user_id}/
        └── uploaded_files/      # User uploaded files
```

## Running with Docker Compose (Recommended)

Use the provided docker-compose.yml file to automatically mount the persistent storage:

```bash
# Build and start the service
docker-compose up --build

# Run in background
docker-compose up -d --build
```

## Running with Docker directly

If you prefer to use Docker directly, mount the storage and video directories:

```bash
# Build the image
docker build -t turing-service .

# Run with persistent storage and ring camera videos mounted
docker run -p 8000:8000 \
  -v /Users/tahakhan/Desktop/home-security-db:/Users/tahakhan/Desktop/home-security-db \
  -v /Users/tahakhan/Desktop/ring_camera:/Users/tahakhan/Desktop/ring_camera \
  -v $(pwd)/static:/app/static \
  turing-service
```

## Data Persistence

### Face Recognition Data
- **Labels**: Person names and bounding boxes are saved to `labels.json`
- **Encodings**: Face encodings for recognition are saved to `encodings.pkl`
- **Frames**: Extracted video frames are saved as individual image files

### Segmentation Data
- **Areas**: Detected and verified areas with polygons and metadata
- **Permissions**: Access permissions for labeled people to specific areas
- **Verification Status**: Whether areas have been user-verified

### Benefits

1. **Data Survives Container Restarts**: Your labeled faces and segmented areas persist across Docker rebuilds
2. **No Data Loss**: Face recognition training and area segmentation work is preserved
3. **User Isolation**: Each user's data is kept separate and organized
4. **Backup Friendly**: Easy to backup the entire `/Users/tahakhan/Desktop/home-security-db/` directory

### Environment Variables

You can customize the storage path using:

```bash
# Set custom storage path
export PERSISTENT_STORAGE_PATH="/path/to/your/storage"
docker-compose up
```

### Storage Management

The system includes built-in storage management:

- **Automatic cleanup** of old files (configurable)
- **Storage statistics** via API endpoints
- **File organization** by user and data type
- **Atomic writes** to prevent data corruption

### Backup Recommendations

1. **Regular backups** of the entire storage directory
2. **Version control** for critical configurations
3. **Separate backup** of user data vs. temporary files

## Troubleshooting

### Permission Issues
```bash
# Fix permissions if needed
sudo chown -R $USER:$USER /Users/tahakhan/Desktop/home-security-db/
chmod -R 755 /Users/tahakhan/Desktop/home-security-db/
```

### Storage Full
```bash
# Check storage usage
df -h /Users/tahakhan/Desktop/home-security-db/

# Clean up old files (via API)
curl -X POST http://localhost:8000/storage/cleanup?days_old=30
```

### Missing Data
- Check if the storage directory is properly mounted
- Verify file permissions
- Check container logs for storage-related errors