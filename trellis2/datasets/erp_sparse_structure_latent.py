# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

## ERP-to-3D Scene Generation Dataset for TRELLIS 2
## This file implements dataset classes for ERP (panorama) conditioned sparse structure generation
## Adapted from TRELLIS 1 for TRELLIS 2 architecture
##
## Data structure expected (ERP_3D_FRONT_test):
## {root}/{uuid}/{room_name}/
##   - ss_latents/{encoder}_{resolution}/full_room_wo_ceiling.npz  (encoded GT SS latents)
##   - cubic_fov_120/{view_idx}/  (6 cubemap images: front.png, back.png, left.png, right.png, top.png, bottom.png)
##   - dap_depth_voxels_ss_latent/{encoder}/{view_idx}.npz  (encoded initial voxels from DAP depth, optional)
##   - camera_poses.json  (camera positions and rotations)
##   - room_info.json  (room bounding box for normalization)
##   - mesh_dumps/normalization_info.json  (normalization parameters from O-Voxel)
##
## Camera Center Normalization:
##   The O-Voxel process transforms world coordinates to a normalized space using:
##     normalized_pos = (world_pos - center) * scale
##   where center and scale are stored in mesh_dumps/normalization_info.json

import os
import json
from typing import *
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

import utils3d
from ..representations import Voxel
from ..renderers import VoxelRenderer
from .. import models
from ..utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics


class ERPDatasetBase(Dataset):
    """
    Base class for ERP-conditioned datasets.

    Unlike StandardDatasetBase which uses metadata.csv with sha256,
    this class scans room folders directly.

    Supports folder structure: {root}/{id}/{room_name}/

    Args:
        roots (str): paths to the dataset (comma-separated if multiple)
    """
    def __init__(self, roots: str):
        super().__init__()
        self.roots = roots.split(',')
        self.instances = []  # List of (root, house_id, room_name, view_idx)
        self._stats = {}

        for root in self.roots:
            key = os.path.basename(root)
            self._stats[key] = {}

            # Scan for house_id folders (first level)
            house_ids = [d for d in os.listdir(root)
                        if os.path.isdir(os.path.join(root, d)) and not d.startswith('.')]

            total_rooms = 0
            valid_rooms = 0
            total_views = 0

            for house_id in house_ids:
                house_path = os.path.join(root, house_id)

                # Scan for room folders (second level)
                room_folders = [d for d in os.listdir(house_path)
                               if os.path.isdir(os.path.join(house_path, d)) and not d.startswith('.')]

                for room_name in room_folders:
                    total_rooms += 1
                    room_path = os.path.join(house_path, room_name)

                    # Check if required files exist
                    if not self._validate_room(room_path):
                        continue

                    valid_rooms += 1

                    # Get number of views from camera_poses.json or cubic folder
                    n_views = self._get_num_views(room_path)
                    total_views += n_views

                    for view_idx in range(n_views):
                        self.instances.append((root, house_id, room_name, view_idx))

            self._stats[key]['Total rooms'] = total_rooms
            self._stats[key]['Valid rooms'] = valid_rooms
            self._stats[key]['Total views'] = total_views


    def _validate_room(self, room_path: str) -> bool:
        """Check if room has required files"""
        # Must have cubic_fov_120 folder
        cubic_path = os.path.join(room_path, 'cubic_fov_120')
        if not os.path.isdir(cubic_path):
            return False

        return True

    def _get_num_views(self, room_path: str) -> int:
        """Get number of views for a room"""
        # Try camera_poses.json first
        camera_poses_path = os.path.join(room_path, 'camera_poses.json')
        if os.path.exists(camera_poses_path):
            with open(camera_poses_path, 'r') as f:
                camera_poses = json.load(f)
            return camera_poses.get('n_views', len(camera_poses.get('views', [])))

        # Fallback: count cubic_fov_120 folders
        cubic_path = os.path.join(room_path, 'cubic_fov_120')
        if os.path.isdir(cubic_path):
            cubic_folders = [d for d in os.listdir(cubic_path)
                            if os.path.isdir(os.path.join(cubic_path, d)) and not d.startswith('.')]
            return len(cubic_folders)
        return 0

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, house_id, room_name, view_idx = self.instances[index]
            return self.get_instance(root, house_id, room_name, view_idx)
        except Exception as e:
            print(f"Error loading instance {index}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))

    def get_instance(self, root: str, house_id: str, room_name: str, view_idx: int) -> Dict[str, Any]:
        raise NotImplementedError

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class ERPSparseStructureLatentVisMixin:
    """
    Mixin for visualizing sparse structure latents from ERP dataset
    """
    def __init__(
        self,
        *args,
        pretrained_ss_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ss_dec = None
        self.pretrained_ss_dec = pretrained_ss_dec
        self.ss_dec_path = ss_dec_path
        self.ss_dec_ckpt = ss_dec_ckpt

    def _loading_ss_dec(self):
        if self.ss_dec is not None:
            return
        if self.ss_dec_path is not None:
            cfg = json.load(open(os.path.join(self.ss_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.ss_dec_path, 'ckpts', f'decoder_{self.ss_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_ss_dec)
        self.ss_dec = decoder.cuda().eval()

    def _delete_ss_dec(self):
        del self.ss_dec
        self.ss_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        self._loading_ss_dec()
        ss = []
        if self.normalization is not None:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            ss.append(self.ss_dec(z[i:i+batch_size]))
        ss = torch.cat(ss, dim=0)
        self._delete_ss_dec()
        return ss

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[torch.Tensor, dict]):
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0['x_0'] # [16, 8, 16, 16, 16]
        x_0 = self.decode_latent(x_0.cuda()) # [16, 1, 64, 64, 64]

        renderer = VoxelRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.ssaa = 4

        # from TRELLIS v1
        # renderer = OctreeRenderer()
        # renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        # renderer.pipe.primitive = 'voxel'

        # Build camera
        yaw = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaw_offset = -16 / 180 * np.pi
        yaw = [y + yaw_offset for y in yaw]
        pitch = [20 / 180 * np.pi for _ in range(4)]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)

        images = []

        x_0 = x_0.cuda()
        for i in range(x_0.shape[0]):
            coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            color = coords / resolution
            rep = Voxel(
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1/resolution,
                coords=coords,
                attrs=color,
                layout={
                    'color': slice(0, 3),
                }
            )
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(rep, ext, intr, colors_overwrite=color)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)

        return torch.stack(images)

    @torch.no_grad()
    def visualize_sample_topdown(self, x_0: Union[torch.Tensor, dict]):
        """Visualize sparse structure latents from a top-down view (pitch=90, looking straight down)."""
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0['x_0']
        x_0 = self.decode_latent(x_0.cuda())

        renderer = VoxelRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.ssaa = 4
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)

        # Top-down camera: yaw=0, pitch=90 degrees (straight down)
        yaw = [0]
        pitch = [90 / 180 * np.pi]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)

        images = []
        x_0 = x_0.cuda()
        for i in range(x_0.shape[0]):
            coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            color = coords / resolution
            rep = Voxel(
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1/resolution,
                coords=coords,
                attrs=color,
                layout={
                    'color': slice(0, 3),
                }
            )
            res = renderer.render(rep, exts[0], ints[0], colors_overwrite=color)
            images.append(res['color'])

        return torch.stack(images)

    @torch.no_grad()
    def visualize_sample_topdown_camera_center(self, data: dict):
        """
        Visualize top-down view with camera center marked as a cyan circle.

        Same rendering as visualize_sample_topdown but overlays the camera center
        position projected onto the top-down image. Requires 'camera_center' in data.

        Returns [B, 3, 512, 512], or None if camera_center is missing.
        """
        from PIL import ImageDraw

        if not isinstance(data, dict) or 'camera_center' not in data:
            return None

        x_0 = data['x_0']
        camera_centers = data['camera_center']  # [B, 3]
        x_0 = self.decode_latent(x_0.cuda())  # [B, 1, 64, 64, 64]

        render_res = 512
        renderer = VoxelRenderer()
        renderer.rendering_options.resolution = render_res
        renderer.rendering_options.ssaa = 4
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)

        yaw = [0]
        pitch = [90 / 180 * np.pi]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)

        images = []
        x_0 = x_0.cuda()
        for i in range(x_0.shape[0]):
            coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            color = coords.float() / resolution
            rep = Voxel(
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1 / resolution,
                coords=coords,
                attrs=color,
                layout={'color': slice(0, 3)},
            )
            res = renderer.render(rep, exts[0], ints[0], colors_overwrite=color)
            face_img = res['color']  # [3, 512, 512]

            # Project camera center onto the top-down image
            cam_3d = camera_centers[i].float().cuda()  # [3]
            point_h = torch.cat([cam_3d, torch.ones(1, device='cuda')])  # [4]
            point_cam = exts[0] @ point_h  # [4]
            point_proj = ints[0] @ point_cam[:3]  # [3]

            if point_proj[2].abs() > 1e-6:
                u = (point_proj[0] / point_proj[2]).item()
                v = (point_proj[1] / point_proj[2]).item()
                px = u * render_res
                py = v * render_res

                # Draw if within image bounds (with margin)
                if -20 < px < render_res + 20 and -20 < py < render_res + 20:
                    img_np = (face_img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)
                    draw = ImageDraw.Draw(pil_img)
                    radius = 8
                    # Cyan fill (#00FFFF) + white outline: high contrast on black bg
                    draw.ellipse(
                        [px - radius, py - radius, px + radius, py + radius],
                        fill=(0, 255, 255),
                        outline=(255, 255, 255),
                        width=2,
                    )
                    face_img = torch.tensor(np.array(pil_img)).permute(2, 0, 1).float().cuda() / 255.0

            images.append(face_img)

        return torch.stack(images)

    def _make_label_strip(self, labels: list, tile_size: int, label_height: int = 24) -> torch.Tensor:
        """Create a white strip with text labels as a tensor [3, label_height, len(labels)*tile_size]."""
        from PIL import ImageDraw, ImageFont
        total_w = len(labels) * tile_size
        strip = Image.new('RGB', (total_w, label_height), (255, 255, 255))
        draw = ImageDraw.Draw(strip)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", label_height - 8)
        except Exception:
            font = ImageFont.load_default()
        for i, label in enumerate(labels):
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            x = i * tile_size + (tile_size - tw) // 2
            draw.text((x, 2), label, fill=(0, 0, 0), font=font)
        strip_tensor = torch.tensor(np.array(strip)).permute(2, 0, 1).float() / 255.0
        return strip_tensor

    @torch.no_grad()
    def visualize_sample_interior(self, data: dict, tile_size: int = 256):
        """
        Visualize interior views from camera center position.

        Returns a composite per sample:
            Row 0: Face labels (white strip with face names)
            Row 1: 6 actual cubemap images (front, right, back, left, top, bottom)
            Row 2: 6 rendered voxel views from the same camera center and directions

        Note on top face: py360convert pitches the camera up by rotating around X-axis,
        which makes image-right=+X and image-up=-Y (backward). To match, utils3d needs
        up_param=[0,-1,0] for the top face, giving right=+X and image-up=-Y naturally.
        Bottom face uses up_param=[0,1,0], giving right=+X and image-up=+Y (forward).

        Returns [B, 3, label_h + 2*tile_size, 6*tile_size], or None if camera_center/cond missing.
        """
        if not isinstance(data, dict) or 'camera_center' not in data or 'cond' not in data:
            return None

        x_0 = data['x_0']
        x_0 = self.decode_latent(x_0.cuda())  # [B, 1, 64, 64, 64]

        camera_centers = data['camera_center']  # [B, 3]
        cubemap_images = data['cond']  # [B, 6, 3, H, W]

        # Scale up the world to work around CUDA rasterizer's hardcoded
        # near culling (p_view.z <= 0.2 in auxiliary.h). near/far scaled too.
        world_scale = 10.0
        renderer = VoxelRenderer()
        renderer.rendering_options.resolution = tile_size
        renderer.rendering_options.ssaa = 4
        renderer.rendering_options.near = 0.01 * world_scale
        renderer.rendering_options.far = 2.0 * world_scale
        renderer.rendering_options.bg_color = (0, 0, 0)

        # Face directions matching cubemap convention
        # front=+Y, right=+X, back=-Y, left=-X, top=+Z, bottom=-Z
        face_labels = ['front (+Y)', 'right (+X)', 'back (-Y)', 'left (-X)', 'top (+Z)', 'bottom (-Z)']
        face_directions = [
            [0.0, 1.0, 0.0],   # front
            [1.0, 0.0, 0.0],   # right
            [0.0, -1.0, 0.0],  # back
            [-1.0, 0.0, 0.0],  # left
            [0.0, 0.0, 1.0],   # top
            [0.0, 0.0, -1.0],  # bottom
        ]
        fov = torch.deg2rad(torch.tensor(120.0)).cuda()

        # Create label strip once
        label_h = max(20, tile_size // 10)
        label_strip = self._make_label_strip(face_labels, tile_size, label_h).cuda()  # [3, label_h, 6*tile_size]

        composites = []
        x_0 = x_0.cuda()
        for i in range(x_0.shape[0]):
            coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            color = coords.float() / resolution

            # Render 6 interior faces
            rendered_faces = []
            if coords.shape[0] > 0:
                rep = Voxel(
                    origin=[-0.5 * world_scale, -0.5 * world_scale, -0.5 * world_scale],
                    voxel_size=world_scale / resolution,
                    coords=coords,
                    attrs=color,
                    layout={'color': slice(0, 3)},
                )
                cam = camera_centers[i].float().cuda() * world_scale

                for face_idx, fd in enumerate(face_directions):
                    look_at = cam + torch.tensor(fd, dtype=torch.float32).cuda()
                    if face_idx == 4:  # top (+Z): up=-Y to match py360convert
                        up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32).cuda()
                    elif face_idx == 5:  # bottom (-Z): up=+Y
                        up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).cuda()
                    else:  # horizontal faces: up=+Z
                        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).cuda()
                    ext = utils3d.torch.extrinsics_look_at(cam, look_at, up)
                    intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
                    res = renderer.render(rep, ext, intr, colors_overwrite=color)
                    rendered_faces.append(res['color'])
            else:
                for _ in range(6):
                    rendered_faces.append(torch.zeros(3, tile_size, tile_size).cuda())

            # Resize cubemap images to tile_size
            cubemap_resized = F.interpolate(
                cubemap_images[i].cuda(),  # [6, 3, H, W]
                size=(tile_size, tile_size), mode='bilinear', align_corners=False,
            )  # [6, 3, tile_size, tile_size]

            # Row 0: labels, Row 1: cubemap images, Row 2: rendered interior
            row1 = torch.cat([cubemap_resized[j] for j in range(6)], dim=2)  # [3, tile_size, 6*tile_size]
            row2 = torch.cat(rendered_faces, dim=2)  # [3, tile_size, 6*tile_size]
            composite = torch.cat([label_strip, row1, row2], dim=1)  # [3, label_h+2*tile_size, 6*tile_size]
            composites.append(composite)

        return torch.stack(composites)


class ERPCubemapConditionedMixin:
    """
    Mixin for loading 6 cubemap images as conditioning.

    Loads cubemap images from: {room_path}/cubic_fov_120/{view_idx}/
    Images: front.png, back.png, left.png, right.png, top.png, bottom.png
    """
    # Cubemap face order (consistent ordering for view position embedding)
    CUBEMAP_FACES = ['front', 'right', 'back', 'left', 'top', 'bottom']

    def __init__(self, roots, *, image_size=512, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)

    def _load_cubemap(self, room_path: str, view_idx: int) -> torch.Tensor:
        """
        Load 6 cubemap images for a view.

        Returns:
            torch.Tensor: [6, 3, image_size, image_size] tensor of cubemap images
        """
        cubic_path = os.path.join(room_path, 'cubic_fov_120', f'{view_idx:04d}')

        images = []
        for face in self.CUBEMAP_FACES:
            image_path = os.path.join(cubic_path, f'{face}.png')
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            images.append(image)

        return torch.stack(images, dim=0)  # [6, 3, H, W]

    def get_instance(self, root: str, house_id: str, room_name: str, view_idx: int) -> Dict[str, Any]:
        pack = super().get_instance(root, house_id, room_name, view_idx)

        room_path = os.path.join(root, house_id, room_name)
        cubemap = self._load_cubemap(room_path, view_idx)
        pack['cond'] = cubemap  # [6, 3, 512, 512]

        return pack


class ERPInitialVoxelLatentMixin:
    """
    Mixin for loading initial voxel latents from DAP depth.

    This enables starting from initial voxel latents instead of random Gaussian noise.
    Loads pre-encoded latents directly (no need to encode at runtime).

    Loads from: {room_path}/dap_depth_voxels_ss_latent/{encoder}/{view_idx}.npz
    """
    def __init__(
        self,
        roots,
        *,
        use_initial_voxel: bool = False,
        initial_voxel_latent_folder: str = 'dap_depth_voxels_ss_latent',
        **kwargs
    ):
        self.use_initial_voxel = use_initial_voxel
        self.initial_voxel_latent_folder = initial_voxel_latent_folder
        super().__init__(roots, **kwargs)

    def _load_initial_voxel_latent(self, room_path: str, view_idx: int) -> Optional[torch.Tensor]:
        """
        Load initial voxel latent from NPZ file.

        Returns:
            torch.Tensor: [8, 16, 16, 16] latent tensor, or None if not available
        """
        npz_path = os.path.join(
            room_path,
            self.initial_voxel_latent_folder,
            self.latent_model,
            f'{view_idx:04d}.npz'
        )

        if not os.path.exists(npz_path):
            return None

        latent = np.load(npz_path)
        z = torch.tensor(latent['z']).float()  # [8, 16, 16, 16]

        # Apply normalization if set
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        return z

    def get_instance(self, root: str, house_id: str, room_name: str, view_idx: int) -> Dict[str, Any]:
        pack = super().get_instance(root, house_id, room_name, view_idx)

        if self.use_initial_voxel:
            room_path = os.path.join(root, house_id, room_name)
            initial_voxel_latent = self._load_initial_voxel_latent(room_path, view_idx)
            if initial_voxel_latent is not None:
                pack['initial_voxel_latent'] = initial_voxel_latent  # [8, 16, 16, 16]

        return pack


class ERPCameraCenterMixin:
    """
    Mixin for loading and normalizing camera center position.

    This enables spatially-aware cross-attention where each cubemap face
    attends to relevant voxel regions based on camera position.

    Loads from:
    - camera_poses.json: {"views": [{"view_idx": 0, "location": [x, y, z]}, ...]}
    - mesh_dumps/normalization_info.json: {"center": [x, y, z], "scale": float}

    Normalization (matches O-Voxel):
    - normalized_pos = (world_pos - center) * scale
    - This matches the voxel grid normalization used in O-Voxel processing
    """
    def __init__(
        self,
        roots,
        *,
        load_camera_center: bool = True,
        **kwargs
    ):
        self.load_camera_center = load_camera_center
        self._camera_poses_cache = {}  # Cache for camera_poses.json
        self._normalization_info_cache = {}  # Cache for normalization_info.json
        super().__init__(roots, **kwargs)

    def _get_room_key(self, root: str, house_id: str, room_name: str) -> str:
        """Get a unique key for caching room data"""
        return f"{root}/{house_id}/{room_name}"

    def _load_camera_poses(self, room_path: str) -> Optional[dict]:
        """Load camera_poses.json with caching"""
        camera_poses_path = os.path.join(room_path, 'camera_poses.json')

        if camera_poses_path in self._camera_poses_cache:
            return self._camera_poses_cache[camera_poses_path]

        if not os.path.exists(camera_poses_path):
            self._camera_poses_cache[camera_poses_path] = None
            return None

        with open(camera_poses_path, 'r') as f:
            camera_poses = json.load(f)

        self._camera_poses_cache[camera_poses_path] = camera_poses
        return camera_poses

    def _load_normalization_info(self, room_path: str) -> Optional[dict]:
        """Load mesh_dumps/normalization_info.json with caching"""
        norm_info_path = os.path.join(room_path, 'mesh_dumps', 'normalization_info.json')

        if norm_info_path in self._normalization_info_cache:
            return self._normalization_info_cache[norm_info_path]

        if not os.path.exists(norm_info_path):
            self._normalization_info_cache[norm_info_path] = None
            return None

        with open(norm_info_path, 'r') as f:
            norm_info = json.load(f)

        self._normalization_info_cache[norm_info_path] = norm_info
        return norm_info

    def _get_normalized_camera_center(
        self,
        room_path: str,
        view_idx: int
    ) -> Optional[torch.Tensor]:
        """
        Get normalized camera center for a view.

        The camera center is normalized using O-Voxel normalization:
        normalized_pos = (world_pos - center) * scale

        Returns:
            torch.Tensor: [3] normalized camera center in voxel space, or None
        """
        camera_poses = self._load_camera_poses(room_path)
        norm_info = self._load_normalization_info(room_path)

        if camera_poses is None or norm_info is None:
            return None

        # Get camera location for this view
        views = camera_poses.get('views', [])
        camera_location = None
        for view in views:
            if view.get('view_idx') == view_idx:
                camera_location = view.get('location')
                break

        if camera_location is None:
            return None

        # Get normalization parameters from O-Voxel
        center = norm_info.get('center')
        scale = norm_info.get('scale')

        if center is None or scale is None:
            return None

        camera_location = np.array(camera_location, dtype=np.float32)
        center = np.array(center, dtype=np.float32)

        # Apply O-Voxel normalization: (pos - center) * scale
        normalized = (camera_location - center) * scale
        camera_center = torch.tensor(normalized, dtype=torch.float32)

        return camera_center

    def get_instance(self, root: str, house_id: str, room_name: str, view_idx: int) -> Dict[str, Any]:
        pack = super().get_instance(root, house_id, room_name, view_idx)

        if self.load_camera_center:
            room_path = os.path.join(root, house_id, room_name)
            camera_center = self._get_normalized_camera_center(room_path, view_idx)
            if camera_center is not None:
                pack['camera_center'] = camera_center  # [3] normalized position
            else:
                # Provide default camera_center at origin if loading fails
                # This prevents KeyError during training for samples with missing data
                pack['camera_center'] = torch.zeros(3, dtype=torch.float32)
                if view_idx == 0:  # Only warn once per room
                    print(f"Warning: camera_center not found for {house_id}/{room_name}, using default [0,0,0]")

        return pack


class ERPSparseStructureLatent(ERPSparseStructureLatentVisMixin, ERPDatasetBase):
    """
    ERP sparse structure latent dataset.

    Loads pre-encoded sparse structure latents (GT voxels encoded by SS-VAE).

    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model with resolution (e.g., 'ss_enc_conv3d_16l8_fp16_64')
        gt_latent_folder (str): folder name for GT latents (default: 'ss_latents')
        normalization (dict): normalization stats (mean, std)
        pretrained_ss_dec (str): pretrained decoder for visualization
    """
    def __init__(
        self,
        roots: str,
        *,
        latent_model: str = 'ss_enc_conv3d_16l8_fp16_64',
        gt_latent_folder: str = 'ss_latents',
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
    ):
        self.latent_model = latent_model
        self.gt_latent_folder = gt_latent_folder
        self.normalization = normalization
        self.value_range = (0, 1)

        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )

        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)

    def _validate_room(self, room_path: str) -> bool:
        """Override to also check for ss_latent"""
        if not super()._validate_room(room_path):
            return False

        # Check for ss_latent in ss_latents folder (new structure: ss_latents/{encoder}_{resolution}/)
        latent_path = os.path.join(room_path, self.gt_latent_folder, self.latent_model, 'full_room_wo_ceiling.npz')
        if not os.path.exists(latent_path):
            return False

        return True

    def get_instance(self, root: str, house_id: str, room_name: str, view_idx: int) -> Dict[str, Any]:
        """
        Load sparse structure latent for a room.

        Note: All views in a room share the same GT latent (one room = one GT structure)
        """
        room_path = os.path.join(root, house_id, room_name)
        latent_path = os.path.join(room_path, self.gt_latent_folder, self.latent_model, 'full_room_wo_ceiling.npz')

        latent = np.load(latent_path)
        z = torch.tensor(latent['z']).float()  # [8, 16, 16, 16]

        if self.normalization is not None:
            z = (z - self.mean) / self.std

        pack = {
            'x_0': z,
            '_data_path': f'{house_id}/{room_name}/view_{view_idx:04d}',
        }
        return pack


class ERPCubemapConditionedSparseStructureLatent(
    ERPCubemapConditionedMixin,
    ERPCameraCenterMixin,
    ERPInitialVoxelLatentMixin,
    ERPSparseStructureLatent
):
    """
    ERP cubemap-conditioned sparse structure latent dataset.

    Main dataset class for ERP-to-3D scene generation training in TRELLIS 2.

    Features:
    - Loads 6 cubemap images as conditioning [6, 3, 512, 512]
    - Loads pre-encoded GT sparse structure latent [8, 16, 16, 16]
    - Optionally loads initial voxel latents from DAP depth [8, 16, 16, 16]
    - Optionally loads normalized camera center for spatial attention [3]

    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        gt_latent_folder (str): folder for GT latents (default: 'voxels_ss_latent')
        image_size (int): size to resize cubemap images (default: 512 for DINOv3)
        use_initial_voxel (bool): whether to load initial voxel latents
        initial_voxel_latent_folder (str): folder for initial voxel latents (default: 'dap_depth_voxels_ss_latent')
        load_camera_center (bool): whether to load normalized camera center (default: True)
        normalization (dict): normalization stats
        pretrained_ss_dec (str): pretrained decoder for visualization
    """
    pass


# Alias for backward compatibility and shorter name
ERPImageConditionedSparseStructureLatent = ERPCubemapConditionedSparseStructureLatent
