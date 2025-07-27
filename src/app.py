from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.face_recognition import router as face_recognition_router
from src.segmentation import router as segmentation_router
from src.monitoring import router as monitoring_router
import os

app = FastAPI(title="Turing Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create required directories if they don't exist
static_dir = "/app/static"
os.makedirs(static_dir, exist_ok=True)
os.makedirs(f"{static_dir}/visualizations", exist_ok=True)
os.makedirs(f"{static_dir}/notification_frames", exist_ok=True)
os.makedirs(f"{static_dir}/extracted_frames", exist_ok=True)
os.makedirs("/app/face_encodings", exist_ok=True)
os.makedirs("/app/person_labels", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include routers
app.include_router(face_recognition_router.router, prefix="/face-recognition", tags=["Face Recognition"])
app.include_router(segmentation_router.router, prefix="/segmentation", tags=["Segmentation"])
app.include_router(monitoring_router.router, prefix="/monitoring", tags=["Monitoring"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Turing Service"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Turing AI"}