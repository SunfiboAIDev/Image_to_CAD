import os
import threading
import uvicorn
from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional
from pyngrok import ngrok
import time

GLB_OUTPUT_DIR = "/content/drive/MyDrive/Image-to-CAD"

class ImageTo3d:
    def __init__(self):
        self.app = FastAPI()
        self.is_processing = False  # Simple in-memory lock
        self.add_routes()

    def add_routes(self):
        @self.app.post("/image_to_3d")
        async def image_to_3d(
            background_tasks: BackgroundTasks,
            image: str = Query(...),
            multiimages: Optional[List[str]] = Query(None),
            is_multiimage: bool = Query(False),
            seed: int = Query(0),
            ss_guidance_strength: float = Query(0.1),
            ss_sampling_steps: int = Query(100),
            slat_guidance_strength: float = Query(0.1),
            slat_sampling_steps: int = Query(100),
            multiimage_algo: str = Query("multidiffusion"),
        ):
            if self.is_processing:
                return JSONResponse(content={"message": "Already processing..."}, status_code=202)

            self.is_processing = True
            background_tasks.add_task(self.generate_model, image)
            return JSONResponse(content={"message": "Processing started in background.."})

        @self.app.get("/get_model")
        async def get_model(name: str = Query(...)):
            model_path = os.path.join(GLB_OUTPUT_DIR, name)
            if os.path.exists(model_path):
                return FileResponse(path=model_path, filename=name, media_type="application/octet-stream")
            else:
                return JSONResponse(content={"error": "Model not found"}, status_code=404)

    def generate_model(self, image_path: str):
        try:
            print("🟡 Started processing the image")
            os.makedirs(GLB_OUTPUT_DIR, exist_ok=True)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            model_path = os.path.join(GLB_OUTPUT_DIR, f"{image_name}.glb")

            # Simulate generation delay
            time.sleep(5)

            # Simulate model creation
            with open(model_path, "w") as f:
                f.write("This is a dummy 3D GLB model.")
            
            print("✅ Finished processing the image")

            print(f"✅ Model saved to: {model_path}")
        finally:
            self.is_processing = False


# Function to start server in background thread
def run_server():
    image_to_3d_app = ImageTo3d()
    uvicorn.run(image_to_3d_app.app, host="0.0.0.0", port=7860)

# Start the FastAPI server in a background thread
server_thread = threading.Thread(target=run_server)
server_thread.start()

# Start ngrok and print public URL
public_url = ngrok.connect(7860)
print(f"🚀 Public URL: {public_url}")
