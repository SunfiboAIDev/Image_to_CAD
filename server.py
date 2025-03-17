import uvicorn
from utils import ImageTo3d  # Import the ImageTo3d class from utils.py

if __name__ == "__main__":
    image_to_3d_app = ImageTo3d()
    uvicorn.run(image_to_3d_app.app, host="127.0.0.1", port=8000)
    