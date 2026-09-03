# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license
#
# Modified from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2)
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.

from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel, Voxel
import utils3d
from .. import models as trellis_models
from ..renderers import VoxelRenderer
import os

class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        self.image_cond_model.image_size = resolution
        if self.low_vram:
            self.image_cond_model.to(self.device) # self.image_cond_model = DinoV3FeatureExtractor
        cond = self.image_cond_model(image)
        if self.low_vram:
            self.image_cond_model.cpu()
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int, # 32
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model'] # SparseStructureFlowModel
        # breakpoint()
        reso = flow_model.resolution # 16
        in_channels = flow_model.in_channels # 8
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device) # [1,8,16,16,16]
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        
        # breakpoint()
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model, 
            noise,
            **cond, # cond['cond'].shape = [1, 1029, 1024]
            **sampler_params, # {'steps': 12, 'guidance_strength': 7.5, 'guidance_rescale': 0.7, 'guidance_interval': [0.6, 1.0], 'rescale_t': 5}
            verbose=True,
            tqdm_desc="Sampling sparse structure",
        ).samples # z_s.shape = [1,8,16,16,16]
        if self.low_vram:
            flow_model.cpu()
        # breakpoint()
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder'] # SparseStructureDecoder
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s)>0 # decoded.shape = [1, 1, 64, 64, 64], 
        # breakpoint()
        # if self.low_vram:
        #     decoder.cpu()
        if resolution != decoded.shape[2]: # must visit here. resolution = 32
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5  # decoded.shape = [1, 1, 32, 32, 32]
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        # Visualization
        if cond['cond'].ndim == 3:
            batch_size = cond['cond'].shape[0]
        elif cond['cond'].ndim == 4:
            batch_size = cond['cond'].shape[1]

        get_voxel_vis = True
        if get_voxel_vis:
            # Use VoxelRenderer instead of OctreeRenderer
            renderer = VoxelRenderer({
                'resolution': 512,
                'near': 0.8,
                'far': 1.6,
                'ssaa': 4,
            })

            # yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
            yaws = [0 * np.pi / 180 for _ in range(4)]
            # yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
            # yaws = [y + yaws_offset for y in yaws]
            # pitch = [np.random.uniform(-75 * np.pi / 180, 90 * np.pi / 180) for _ in range(4)]
            # set pitch to 0 degrees
            # pitch = [0 * np.pi / 180 for _ in range(4)]
            pitch = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

            exts = []
            ints = []
            for yaw, p in zip(yaws, pitch):
                orig = torch.tensor([
                    np.sin(yaw) * np.cos(p),
                    np.cos(yaw) * np.cos(p),
                    np.sin(p),
                ]).float().cuda() * 2
                fov = torch.deg2rad(torch.tensor(40)).cuda()
                extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
                intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
                exts.append(extrinsics)
                ints.append(intrinsics)

            images = []
            x_0 = decoder(z_s)
            vis_resolution = x_0.shape[-1]
            voxel_size = 1.0 / vis_resolution

            # if self.low_vram:
            #     decoder.cpu()

            for i in range(batch_size):
                coords_vis = torch.nonzero(x_0[i, 0] > 0, as_tuple=False).int()  # [N, 3]

                # Create Voxel representation
                # Use position-based colors (normalized to [0, 1])
                positions_normalized = coords_vis.float() / vis_resolution  # [N, 3] in [0, 1]

                voxel = Voxel(
                    origin=[-0.5, -0.5, -0.5],
                    voxel_size=voxel_size,
                    coords=coords_vis.to(self.device),
                    attrs=positions_normalized.to(self.device),  # Use position as color
                    layout={'color': slice(0, 3)},
                    device=self.device,
                )

                image = torch.zeros(3, 1024, 1024).cuda()
                tile = [2, 2]
                for j, (ext, intr) in enumerate(zip(exts, ints)):
                    res = renderer.render(voxel, ext, intr, colors_overwrite=positions_normalized.to(self.device))
                    image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
                images.append(image)

            os.makedirs('voxel_vis_random_noise', exist_ok=True)
            for i, image in enumerate(images):
                image = image.cpu().permute(1, 2, 0).numpy()
                image = np.clip(image, 0, 1)
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
                image.save(f'voxel_vis_random_noise/voxel_vis_{i}.png')
            print(f"Saved scene voxel visualization to voxel_vis for each sample")

            # Save voxel in PLY format
            for i in range(batch_size):
                coords_vis = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)  # [N, 3]
                # Convert to normalized coordinates [-0.5, 0.5]
                positions = (coords_vis.float() / vis_resolution) - 0.5  # [N, 3] in [-0.5, 0.5]
                positions_np = positions.cpu().numpy()

                # Use position-based colors (RGB from XYZ)
                colors_np = ((coords_vis.float() / vis_resolution).cpu().numpy() * 255).astype(np.uint8)

                # Save PLY using utils3d
                ply_path = f'voxel_vis_random_noise/voxel_{i}.ply'
                utils3d.io.write_ply(
                    ply_path,
                    positions_np,
                    vertex_colors=colors_np,
                )
                print(f"Saved voxel PLY to {ply_path}")

        if self.low_vram:
            decoder.cpu()

        return coords # [4692, 4]


    def get_voxels(self, voxelized_ply_path, resolution: int = 64):
        position = utils3d.io.read_ply(voxelized_ply_path)[0]
        coords = ((torch.tensor(position) + 0.5) * resolution).int().contiguous() # back to the 0-64 scale, coords.shape = [21274, 3]
        ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
        ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1 # ss.sum() = 21274
        return ss # ss.shape = [1, 64, 64, 64]

    def _load_sparse_structure_encoder(self, encoder_path: str = "/path/to/TRELLIS/microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"):
        """
        Load the sparse structure encoder if not already loaded.

        Args:
            encoder_path (str): The path to the pretrained encoder model.
        """
        if 'sparse_structure_encoder' not in self.models:
            encoder = trellis_models.from_pretrained(encoder_path).eval().to(self.device)
            self.models['sparse_structure_encoder'] = encoder
        return self.models['sparse_structure_encoder']

    # added by kookie 25.12.26
    def sample_sparse_structure_from_voxelized_ply(
        self,
        cond: dict,
        voxelized_ply_path: str,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model'] # SparseStructureFlowModel
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        # noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device) # [1,8,16,16,16]

        # instead of using random gaussian noise, use voxelized ply
        # Step 1: Load voxelized ply and convert to [1, 64, 64, 64] voxel grid
        sparse_resolution = 64
        voxels = self.get_voxels(voxelized_ply_path, sparse_resolution)  # [1, 64, 64, 64]
        voxels = voxels[None].float().to(self.device)  # [1, 1, 64, 64, 64]

        # Step 2: Encode the voxel grid using sparse structure encoder
        # Load encoder if not already loaded
        encoder = self._load_sparse_structure_encoder()

        # Encode voxels to latent: [1, 1, 64, 64, 64] -> [1, 8, 16, 16, 16]
        z_init = encoder(voxels, sample_posterior=False)  # [1, 8, 16, 16, 16]

        # Expand to num_samples if needed
        if num_samples > 1:
            z_init = z_init.repeat(num_samples, 1, 1, 1, 1)

        print(f"Encoded voxel latent shape: {z_init.shape}")  # Should be [num_samples, 8, 16, 16, 16]

        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        z_s = self.sparse_structure_sampler.sample(
            flow_model, 
            z_init, # noise,
            **cond, # cond['cond'].shape = [1, 1029, 1024]
            **sampler_params, # {'steps': 12, 'guidance_strength': 7.5, 'guidance_rescale': 0.7, 'guidance_interval': [0.6, 1.0], 'rescale_t': 5}
            verbose=True,
            tqdm_desc="Sampling sparse structure",
        ).samples # z_s.shape = [1,8,16,16,16]
        if self.low_vram:
            flow_model.cpu()
        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder'] # SparseStructureDecoder
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s)>0 # decoded.shape = [1, 1, 32, 32, 32], 
        # if self.low_vram:
        #     decoder.cpu()
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5 
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        # Visualization
        if cond['cond'].ndim == 3:
            batch_size = cond['cond'].shape[0]
        elif cond['cond'].ndim == 4:
            batch_size = cond['cond'].shape[1]

        get_voxel_vis = True
        if get_voxel_vis:
            # Use VoxelRenderer instead of OctreeRenderer
            renderer = VoxelRenderer({
                'resolution': 512,
                'near': 0.8,
                'far': 1.6,
                'ssaa': 4,
            })

            # yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
            yaws = [0 * np.pi / 180 for _ in range(4)]
            # yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
            # yaws = [y + yaws_offset for y in yaws]
            # pitch = [np.random.uniform(-75 * np.pi / 180, 90 * np.pi / 180) for _ in range(4)]
            # pitch = [0 * np.pi / 180 for _ in range(4)]
            pitch = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

            exts = []
            ints = []
            for yaw, p in zip(yaws, pitch):
                orig = torch.tensor([
                    np.sin(yaw) * np.cos(p),
                    np.cos(yaw) * np.cos(p),
                    np.sin(p),
                ]).float().cuda() * 2
                fov = torch.deg2rad(torch.tensor(40)).cuda()
                extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
                intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
                exts.append(extrinsics)
                ints.append(intrinsics)

            images = []
            x_0 = decoder(z_s)
            vis_resolution = x_0.shape[-1]
            voxel_size = 1.0 / vis_resolution

            # if self.low_vram:
            #     decoder.cpu()

            for i in range(batch_size):
                coords_vis = torch.nonzero(x_0[i, 0] > 0, as_tuple=False).int()  # [N, 3]

                # Create Voxel representation
                # Use position-based colors (normalized to [0, 1])
                positions_normalized = coords_vis.float() / vis_resolution  # [N, 3] in [0, 1]

                voxel = Voxel(
                    origin=[-0.5, -0.5, -0.5],
                    voxel_size=voxel_size,
                    coords=coords_vis.to(self.device),
                    attrs=positions_normalized.to(self.device),  # Use position as color
                    layout={'color': slice(0, 3)},
                    device=self.device,
                )

                image = torch.zeros(3, 1024, 1024).cuda()
                tile = [2, 2]
                for j, (ext, intr) in enumerate(zip(exts, ints)):
                    res = renderer.render(voxel, ext, intr, colors_overwrite=positions_normalized.to(self.device))
                    image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
                images.append(image)

            # voxelized_ply_path = /path/to/OmniPart/voxel_3d_future/18461765-da45-4d0a-8133-a9df3fa8aee1/LivingRoom-150057_0003_DAP_wo_ceiling/voxels_direct.ply
            room_name = os.path.basename(os.path.dirname(voxelized_ply_path))

            os.makedirs('voxel_vis/' + room_name, exist_ok=True)
            for i, image in enumerate(images):
                image = image.cpu().permute(1, 2, 0).numpy()
                image = np.clip(image, 0, 1)
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
                image.save(f'voxel_vis/{room_name}/voxel_vis_{room_name}_{i}.png')
            print(f"Saved scene voxel visualization to voxel_vis for each sample")

            # Save voxel in PLY format
            for i in range(batch_size):
                coords_vis = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)  # [N, 3]
                # Convert to normalized coordinates [-0.5, 0.5]
                positions = (coords_vis.float() / vis_resolution) - 0.5  # [N, 3] in [-0.5, 0.5]
                positions_np = positions.cpu().numpy()

                # Use position-based colors (RGB from XYZ)
                colors_np = ((coords_vis.float() / vis_resolution).cpu().numpy() * 255).astype(np.uint8)

                # Save PLY using utils3d
                ply_path = f'voxel_vis/{room_name}/voxel_vis_{room_name}_{i}.ply'
                utils3d.io.write_ply(
                    ply_path,
                    positions_np,
                    vertex_colors=colors_np,
                )
                print(f"Saved voxel PLY to {ply_path}")
        
        if self.low_vram:
            decoder.cpu()

        return coords  # [4692, 4]
    

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device), # [4962, 32]
            coords=coords, # [4962, 4]
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params} # {'steps': 12, 'guidance_strength': 7.5, 'guidance_rescale': 0.5, 'guidance_interval': [0.6, 1.0], 'rescale_t': 3}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model, # SLatFlowModel
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples # slat.coords.shape = [4962, 4], slat.feats.shape = [4962, 32]
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model_lr, # SLatFlowModel
            noise,
            **lr_cond, # lr_cond['cond'].shape = [1, 1029, 1024]
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples # slat.coords.shape = [2724, 4], slat.feats.shape = [2724, 32]
        if self.low_vram:
            flow_model_lr.cpu()
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        # Upsample
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4) # FlexiDualGridVaeDecoder hr_coords.shape = torch.Size([3177026, 4])
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution # hr_resolution = 1024
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        ) # noise.coords.shape = [28767, 4], noise.feats.shape = [28767, 32]
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples # slat.coords.shape = [28767, 4], slat.feats.shape = [28767, 32]
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        self.models['shape_slat_decoder'].set_resolution(resolution) # FlexiDualGridVaeDecoder
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        ret = self.models['shape_slat_decoder'](slat, return_subs=True)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std
        # shape_slat.replace(feats=...)  # keep the shape_slat coordinates, replace only the features with noise
        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device)) # shape_slat.coords.shape[0]=4962, in_channels - shape_slat.feats.shape[1] = 32, noise.shape = [4962, 32]
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
        ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
        return ret
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        meshes, subs = self.decode_shape_slat(shape_slat, resolution) # FlexiDualGridVaeDecoder, len(subs) = 4, subs= [SparseTensor(shape=torch.Size([1, 8]), dtype=torch.float16, device=cuda:0), ...] (4 entries)
        tex_voxels = self.decode_tex_slat(tex_slat, subs) # tex_voxels.coords.shape = [1843665, 4], tex_voxels.feats.shape = [1843665, 6]
        out_mesh = [] # meshes[0].__dict__['vertices'].shape = torch.Size([1843665, 3])
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh
    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        voxelized_ply_path: str = None, # added by kookie 25.12.30
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            voxelized_ply_path (str): The path to the voxelized PLY file.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")
        
        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512) # cond_512['cond'].shape = [1, 1029, 1024]
        cond_1024 = self.get_cond([image], 1024) if pipeline_type != '512' else None
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type] # 32

        # Use voxelized PLY input if provided, otherwise use random noise
        if voxelized_ply_path is not None:
            coords = self.sample_sparse_structure_from_voxelized_ply(
                cond_512, voxelized_ply_path, ss_res,
                num_samples, sparse_structure_sampler_params
            )
        else:
            coords = self.sample_sparse_structure(
                cond_512, ss_res, # ss_res = 32
                num_samples, sparse_structure_sampler_params
            ) # coords.shape = [4692, 4]

        if pipeline_type == '512':
            shape_slat = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'], # SLatFlowModel
                coords, shape_slat_sampler_params # {'steps': 12, 'guidance_strength': 7.5, 'guidance_rescale': 0.5, 'rescale_t': 3}
            )
            tex_slat = self.sample_tex_slat(
                cond_512, self.models['tex_slat_flow_model_512'],
                shape_slat, tex_slat_sampler_params
            )
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            res = 1024
        elif pipeline_type == '1024_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens # max_num_tokens = 49152
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        elif pipeline_type == '1536_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res) # res=512
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh

    @torch.no_grad()
    def run_with_voxelized_ply(
        self,
        image: Image.Image,
        voxelized_ply_path: str,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline with a voxelized PLY file as initial sparse structure.
        Instead of sampling sparse structure from random noise, this uses the
        encoded latent from the voxelized PLY file.

        Args:
            image (Image.Image): The image prompt.
            voxelized_ply_path (str): Path to the voxelized PLY file.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512)
        cond_1024 = self.get_cond([image], 1024) if pipeline_type != '512' else None
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]

        # Use voxelized PLY for sparse structure sampling
        coords = self.sample_sparse_structure_from_voxelized_ply(
            cond_512, voxelized_ply_path, ss_res,
            num_samples, sparse_structure_sampler_params
        )

        if pipeline_type == '512':
            shape_slat = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_512, self.models['tex_slat_flow_model_512'],
                shape_slat, tex_slat_sampler_params
            )
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
            res = 1024
        elif pipeline_type == '1024_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        elif pipeline_type == '1536_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens
            )
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params
            )
        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh
