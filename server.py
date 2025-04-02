import os
import sys

sys.path.append('/content/drive/MyDrive/packages')
os.environ['ATTN_BACKEND'] = 'xformers'

import uvicorn
import nest_asyncio
from pyngrok import ngrok

from utils import ImageTo3d

if __name__ == "__main__":
    image_to_3d_app = ImageTo3d()
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(image_to_3d_app.app, host="127.0.0.1", port=8000)
