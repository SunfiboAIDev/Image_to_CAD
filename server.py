import os
import threading
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from typing import List, Optional

# For ngrok (Colab-specific)
from pyngrok import ngrok

class ImageTo3d:
    def __init__(self):
        self.app = FastAPI()
        self.add_routes()

    def add_routes(self):
        @self.app.post("/image_to_3d")
        async def image_to_3d(
            path: str = Query(...),  # This is your main input
            multiimages: Optional[List[str]] = Query(None),
            is_multiimage: bool = Query(False),
            seed: int = Query(0),
            ss_guidance_strength: float = Query(0.1),
            ss_sampling_steps: int = Query(100),
            slat_guidance_strength: float = Query(0.1),
            slat_sampling_steps: int = Query(100),
            multiimage_algo: str = Query("multidiffusion"),
        ):
            if not os.path.exists(path):
                return JSONResponse(content={"error": "Path not found"}, status_code=404)
            
            # This is where your real 3D generation code would run
            return {
                "message": "Path verified and 3D generation would start here",
                "path": path,
                "multiimage_algo": multiimage_algo,
                "is_multiimage": is_multiimage
            }

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
