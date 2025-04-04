# main.py
from fastapi import FastAPI
from utils import ImageTo3d

app = FastAPI()

# Create instance of your custom class
image_to_3d_service = ImageTo3d()

# Mount the internal FastAPI app under /ImageTo3D
app.mount("/ImageTo3D", image_to_3d_service.app)
