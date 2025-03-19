import sys
sys.path.append('/content/drive/MyDrive/packages')

import os
import shutil
from typing import *
import torch
import numpy as np
import imageio
from easydict import EasyDict as edict
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import base64
import io
import json

class ImageTo3d:
    def __init__(self):
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=['*'],
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
        )
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        self.pipeline.cuda()
        self.MAX_SEED = np.iinfo(np.int32).max
        self.TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
        self.state = None
        os.makedirs(self.TMP_DIR, exist_ok=True)

        # Add FastAPI routes
        self.add_routes()

    def add_routes(self):
        @self.app.post("/image_to_3d")
        async def image_to_3d(
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
            image = Image.open(image)
            multiimages = [Image.open(i) for i in multiimages] if multiimages else None
            seed = self.get_seed(False, seed)
            state, video_path = self.image_to_3d(
                image, multiimages, is_multiimage, seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps, multiimage_algo
            )
            self.state = state
            return JSONResponse(content={"video_path": video_path})
        
        @self.app.post("/extract_glb")
        async def extract_glb(
            mesh_simplify: float = Query(0.1),
            texture_size: int = Query(1024),
        ):
            glb_path = ''
            if self.state is not None:
              glb_path = self.extract_glb(self.state, mesh_simplify, texture_size)
            return JSONResponse(content={"glb_path": glb_path})
        
        @self.app.post("/extract_gaussian")
        async def extract_gaussian():
            gaussian_path = ''
            if self.state is not None:
              gaussian_path = self.extract_gaussian(self.state)
            return JSONResponse(content={"gaussian_path": gaussian_path})

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess the input image.

        Args:
            image (Image.Image): The input image.

        Returns:
            Image.Image: The preprocessed image.
        """
        processed_image = self.pipeline.preprocess_image(image)
        return processed_image


    def preprocess_images(self, images: List[Tuple[Image.Image, str]]) -> List[Image.Image]:
        """
        Preprocess a list of input images.
        
        Args:
            images (List[Tuple[Image.Image, str]]): The input images.
            
        Returns:
            List[Image.Image]: The preprocessed images.
        """
        images = [image[0] for image in images]
        processed_images = [self.pipeline.preprocess_image(image) for image in images]
        return processed_images


    def pack_state(self, gs: Gaussian, mesh: MeshExtractResult) -> dict:
        return {
            'gaussian': {
                **gs.init_params,
                '_xyz': gs._xyz.cpu().numpy(),
                '_features_dc': gs._features_dc.cpu().numpy(),
                '_scaling': gs._scaling.cpu().numpy(),
                '_rotation': gs._rotation.cpu().numpy(),
                '_opacity': gs._opacity.cpu().numpy(),
            },
            'mesh': {
                'vertices': mesh.vertices.cpu().numpy(),
                'faces': mesh.faces.cpu().numpy(),
            },
        }
    
    
    def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
        gs = Gaussian(
            aabb=state['gaussian']['aabb'],
            sh_degree=state['gaussian']['sh_degree'],
            mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
            scaling_bias=state['gaussian']['scaling_bias'],
            opacity_bias=state['gaussian']['opacity_bias'],
            scaling_activation=state['gaussian']['scaling_activation'],
        )
        gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
        gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
        gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
        gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
        gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
        
        mesh = edict(
            vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
            faces=torch.tensor(state['mesh']['faces'], device='cuda'),
        )
        
        return gs, mesh


    def get_seed(self, randomize_seed: bool, seed: int) -> int:
        """
        Get the random seed.
        """
        return np.random.randint(0, self.MAX_SEED) if randomize_seed else seed


    def image_to_3d(
        self,
        image: Image.Image,
        multiimages: List[Tuple[Image.Image, str]],
        is_multiimage: bool,
        seed: int,
        ss_guidance_strength: float,
        ss_sampling_steps: int,
        slat_guidance_strength: float,
        slat_sampling_steps: int,
        multiimage_algo: Literal["multidiffusion", "stochastic"]
    ) -> Tuple[dict, str]:
        """
        Convert an image to a 3D model.

        Args:
            image (Image.Image): The input image.
            multiimages (List[Tuple[Image.Image, str]]): The input images in multi-image mode.
            is_multiimage (bool): Whether is in multi-image mode.
            seed (int): The random seed.
            ss_guidance_strength (float): The guidance strength for sparse structure generation.
            ss_sampling_steps (int): The number of sampling steps for sparse structure generation.
            slat_guidance_strength (float): The guidance strength for structured latent generation.
            slat_sampling_steps (int): The number of sampling steps for structured latent generation.
            multiimage_algo (Literal["multidiffusion", "stochastic"]): The algorithm for multi-image generation.

        Returns:
            dict: The information of the generated 3D model.
            str: The path to the video of the 3D model.
        """
        user_dir = self.TMP_DIR
        if not is_multiimage:
            outputs = self.pipeline.run(
                image,
                seed=seed,
                formats=["gaussian", "mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": ss_sampling_steps,
                    "cfg_strength": ss_guidance_strength,
                },
                slat_sampler_params={
                    "steps": slat_sampling_steps,
                    "cfg_strength": slat_guidance_strength,
                },
            )
        else:
            outputs = self.pipeline.run_multi_image(
                [image[0] for image in multiimages],
                seed=seed,
                formats=["gaussian", "mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": ss_sampling_steps,
                    "cfg_strength": ss_guidance_strength,
                },
                slat_sampler_params={
                    "steps": slat_sampling_steps,
                    "cfg_strength": slat_guidance_strength,
                },
                mode=multiimage_algo,
            )
        video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
        video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
        video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
        video_path = os.path.join(user_dir, 'sample.mp4')
        imageio.mimsave(video_path, video, fps=15)
        state = self.pack_state(outputs['gaussian'][0], outputs['mesh'][0])
        torch.cuda.empty_cache()
        return state, video_path


    def extract_glb(
        self,
        state: dict,
        mesh_simplify: float,
        texture_size: int
    ) -> str:
        """
        Extract a GLB file from the 3D model.

        Args:
            state (dict): The state of the generated 3D model.
            mesh_simplify (float): The mesh simplification factor.
            texture_size (int): The texture resolution.

        Returns:
            str: The path to the extracted GLB file.
        """
        user_dir = self.TMP_DIR
        gs, mesh = self.unpack_state(state)
        glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
        glb_path = os.path.join(user_dir, 'sample.glb')
        glb.export(glb_path)
        torch.cuda.empty_cache()
        return glb_path
    

    def extract_gaussian(self, state: dict) -> str:
        """
        Extract a Gaussian file from the 3D model.

        Args:
            state (dict): The state of the generated 3D model.

        Returns:
            str: The path to the extracted Gaussian file.
        """
        user_dir = self.TMP_DIR
        gs, _ = self.unpack_state(state)
        gaussian_path = os.path.join(user_dir, 'sample.ply')
        gs.save_ply(gaussian_path)
        torch.cuda.empty_cache()
        return gaussian_path


    def prepare_multi_example(self) -> List[Image.Image]:
        multi_case = list(set([i.split('_')[0] for i in os.listdir("assets/example_multi_image")]))
        images = []
        for case in multi_case:
            _images = []
            for i in range(1, 4):
                img = Image.open(f'assets/example_multi_image/{case}_{i}.png')
                W, H = img.size
                img = img.resize((int(W / H * 512), 512))
                _images.append(np.array(img))
            images.append(Image.fromarray(np.concatenate(_images, axis=1)))
        return images


    def split_image(self, image: Image.Image) -> List[Image.Image]:
        """
        Split an image into multiple views.
        """
        image = np.array(image)
        alpha = image[..., 3]
        alpha = np.any(alpha>0, axis=0)
        start_pos = np.where(~alpha[:-1] & alpha[1:])[0].tolist()
        end_pos = np.where(alpha[:-1] & ~alpha[1:])[0].tolist()
        images = []
        for s, e in zip(start_pos, end_pos):
            images.append(Image.fromarray(image[:, s:e+1]))
        return [self.preprocess_image(image) for image in images]
