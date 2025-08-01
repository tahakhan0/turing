from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.face_recognition import router as face_recognition_router
from src.segmentation import router as segmentation_router
from src.monitoring import router as monitoring_router
from src.storage.persistent_storage import PersistentStorage
import os

app = FastAPI(title="Turing Service")

# Initialize persistent storage
storage = PersistentStorage()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create required directories using persistent storage base path
static_dir = storage.base_path
os.makedirs(static_dir, exist_ok=True)
os.makedirs(f"{static_dir}/visualizations", exist_ok=True)
os.makedirs(f"{static_dir}/segmentation_visualizations", exist_ok=True)
os.makedirs(f"{static_dir}/notification_frames", exist_ok=True)
os.makedirs(f"{static_dir}/extracted_frames", exist_ok=True)

# Mount static files from persistent storage
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