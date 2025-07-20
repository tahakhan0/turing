# Turing Face Recognition Web UI

A modern, responsive web interface for the Turing Face Recognition service. This UI provides a visual interface for analyzing videos, detecting faces, and labeling them for training face recognition models.

## Features

- **Video Analysis**: Upload videos or provide URLs for face detection analysis
- **Real-time Results**: View detected faces with bounding boxes and confidence scores
- **Interactive Labeling**: Click on faces to assign names and labels
- **Batch Operations**: Save all unlabeled faces as "Unknown" or filter by status
- **Responsive Design**: Works on desktop and mobile devices
- **Live Service Status**: Shows connection status to the FastAPI backend

## Quick Start

1. **Start the Turing service** (from the main project):
   ```bash
   # In the turing directory
   docker-compose up
   ```

2. **Open the Web UI**:
   Simply open `index.html` in your web browser, or serve it with a local server:
   ```bash
   # Option 1: Direct file access
   open index.html
   
   # Option 2: Simple HTTP server
   python -m http.server 8080
   # Then visit http://localhost:8080
   
   # Option 3: Node.js server
   npx serve .
   ```

## Usage

### 1. Configure Connection
- Ensure the **Service URL** is set correctly (default: `http://localhost:8000`)
- The connection status indicator will show green when connected

### 2. Analyze Video
- Enter a **User ID** (e.g., "user123")
- Provide a **Video Path** (local file path or URL)
- Click **"Start Analysis"**

### 3. Label Faces
- Review the detected faces in the grid
- Click on any face to open the labeling modal
- Enter a person's name or select from suggestions
- Choose to **Confirm**, **Skip**, or **Reject** each face

### 4. Manage Results
- Use filters to view only labeled/unlabeled faces
- Search by person name
- Use **"Save All Labels"** to batch-label remaining faces as "Unknown"

## API Integration

This web UI communicates with the Turing service using these endpoints:

- `GET /health` - Check service status
- `POST /face-recognition/enrollment` - Analyze video for person detection (enrollment mode)
- `POST /face-recognition/analyze` - Analyze video with face recognition
- `POST /face-recognition/save-face` - Save labeled face encoding

## File Structure

```
face-recognition/
├── index.html          # Main HTML interface
├── app.js             # JavaScript application logic
└── README.md          # This documentation
```

## Customization

### Styling
The UI uses Tailwind CSS loaded from CDN. To customize styles:
- Modify Tailwind classes in `index.html`
- Add custom CSS in the `<style>` section

### API Configuration
To use with a different backend:
- Update the `apiBaseUrl` in `app.js`
- Modify API endpoints if your service uses different routes

### Features
Key areas for enhancement:
- **Batch Upload**: Support multiple video files
- **Export Options**: Download labeled data in various formats
- **Advanced Filtering**: Filter by confidence, frame range, etc.
- **User Management**: Support multiple users in the same session

## Browser Compatibility

- Modern browsers with ES6+ support
- Chrome, Firefox, Safari, Edge (latest versions)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Integration with Other Applications

This web UI is designed to be easily integrated into other applications:

1. **Standalone**: Use as-is for face recognition workflows
2. **Embedded**: Include in existing web applications
3. **API Bridge**: Use the JavaScript classes to build custom interfaces

## Development

To extend the functionality:

1. **Add new API endpoints** in `app.js`
2. **Create new UI components** in `index.html`
3. **Extend the `FaceRecognitionUI` class** for additional features

## License

This project is part of the Turing Face Recognition system and follows the same licensing terms.