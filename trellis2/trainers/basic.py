# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license
#
# Modified from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2)
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.

from abc import abstractmethod
from typing import List, Dict, Optional, Tuple
import os
import time
import json
import copy
import threading
import warnings
from functools import partial
from contextlib import nullcontext

warnings.filterwarnings("ignore", message="Tight layout not applied")

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from torchvision import utils
from torch.utils.tensorboard import SummaryWriter

from .utils import *
from ..utils.general_utils import *
from ..utils.data_utils import recursive_to_device, cycle, ResumableSampler, WeightedResumableSampler
from ..utils.dist_utils import *
from ..utils import grad_clip_utils, elastic_utils
from tqdm import tqdm

class BasicTrainer:
    """
    Trainer for basic training loop.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        mix_precision_mode (str):
            - None: No mixed precision.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        mix_precision_dtype (str): Mixed precision dtype.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        parallel_mode (str): Parallel mode. Options are 'ddp'.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
    """
    def __init__(self,
        models,
        dataset,
        *,
        eval_dataset=None,
        output_dir,
        load_dir,
        step,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        optimizer={},
        lr_scheduler=None,
        elastic=None,
        grad_clip=None,
        ema_rate=0.9999,
        fp16_mode=None,
        mix_precision_mode='inflat_all',
        mix_precision_dtype='float16',
        fp16_scale_growth=1e-3,
        parallel_mode='ddp',
        finetune_ckpt=None,
        log_param_stats=False,
        prefetch_data=True,
        snapshot_batch_size=4,
        i_print=1000,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        i_ddpcheck=10000,
        **kwargs
    ):
        assert batch_size is not None or batch_size_per_gpu is not None, 'Either batch_size or batch_size_per_gpu must be specified.'

        self.models = models
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.batch_split = batch_split if batch_split is not None else 1
        self.max_steps = max_steps
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.elastic_controller_config = elastic
        self.grad_clip = grad_clip
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate
        if fp16_mode is not None:
            mix_precision_dtype = 'float16'
            mix_precision_mode = fp16_mode
        self.mix_precision_mode = mix_precision_mode
        self.mix_precision_dtype = str_to_dtype(mix_precision_dtype)
        self.fp16_scale_growth = fp16_scale_growth
        self.parallel_mode = parallel_mode
        self.log_param_stats = log_param_stats
        self.prefetch_data = prefetch_data
        self.snapshot_batch_size = snapshot_batch_size
        self.log = []
        if self.prefetch_data:
            self._data_prefetched = None

        self.output_dir = output_dir
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.i_save = i_save
        self.i_ddpcheck = i_ddpcheck        

        if dist.is_initialized():
            # Multi-GPU params
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_rank() % torch.cuda.device_count()
            self.is_master = self.rank == 0
        else:
            # Single-GPU params
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True

        self.batch_size = batch_size if batch_size_per_gpu is None else batch_size_per_gpu * self.world_size
        self.batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size
        assert self.batch_size % self.world_size == 0, 'Batch size must be divisible by the number of GPUs.'
        assert self.batch_size_per_gpu % self.batch_split == 0, 'Batch size per GPU must be divisible by batch split.'

        self.init_models_and_more(**kwargs)
        self.prepare_dataloader(**kwargs)
        
        # Load checkpoint
        self.step = 0
        if load_dir is not None and step is not None:
            self.load(load_dir, step)
        elif finetune_ckpt is not None:
            self.finetune_from(finetune_ckpt)
        
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'ckpts'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.output_dir, 'tb_logs'))

        if self.parallel_mode == 'ddp' and self.world_size > 1:
            self.check_ddp()
            
        if self.is_master:
            print('\n\nTrainer initialized.')
            print(self)

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Models:')
        for name, model in self.models.items():
            lines.append(f'    - {name}: {model.__class__.__name__}')
        lines.append(f'  - Dataset: {indent(str(self.dataset), 2)}')
        lines.append(f'  - Dataloader:')
        lines.append(f'    - Sampler: {self.dataloader.sampler.__class__.__name__}')
        lines.append(f'    - Num workers: {self.dataloader.num_workers}')
        lines.append(f'  - Number of steps: {self.max_steps}')
        lines.append(f'  - Number of GPUs: {self.world_size}')
        lines.append(f'  - Batch size: {self.batch_size}')
        lines.append(f'  - Batch size per GPU: {self.batch_size_per_gpu}')
        lines.append(f'  - Batch split: {self.batch_split}')
        lines.append(f'  - Optimizer: {self.optimizer.__class__.__name__}')
        lines.append(f'  - Learning rate: {self.optimizer.param_groups[0]["lr"]}')
        if self.lr_scheduler_config is not None:
            lines.append(f'  - LR scheduler: {self.lr_scheduler.__class__.__name__}')
        if self.elastic_controller_config is not None:
            lines.append(f'  - Elastic memory: {indent(str(self.elastic_controller), 2)}')
        if self.grad_clip is not None:
            lines.append(f'  - Gradient clip: {indent(str(self.grad_clip), 2)}')
        lines.append(f'  - EMA rate: {self.ema_rate}')
        lines.append(f'  - Mixed precision dtype: {self.mix_precision_dtype}')
        lines.append(f'  - Mixed precision mode: {self.mix_precision_mode}')
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            lines.append(f'  - FP16 scale growth: {self.fp16_scale_growth}')
        lines.append(f'  - Parallel mode: {self.parallel_mode}')
        return '\n'.join(lines)

    @property
    def device(self):
        for _, model in self.models.items():
            if hasattr(model, 'device'):
                return model.device
        return next(list(self.models.values())[0].parameters()).device
            
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        """
        if self.world_size > 1:
            # Prepare distributed data parallel
            self.training_models = {
                name: DDP(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    bucket_cap_mb=128,
                    find_unused_parameters=False
                )
                for name, model in self.models.items()
            }
        else:
            self.training_models = self.models

        # Build master params
        self.model_params = sum(
            [[p for p in model.parameters() if p.requires_grad] for model in self.models.values()]
        , [])
        if self.mix_precision_mode == 'amp':
            self.master_params = self.model_params
            if self.mix_precision_dtype == torch.float16:
                self.scaler = torch.GradScaler()
        elif self.mix_precision_mode == 'inflat_all':
            self.master_params = make_master_params(self.model_params)
            if self.mix_precision_dtype == torch.float16:
                self.log_scale = 20.0
        elif self.mix_precision_mode is None:
            self.master_params = self.model_params
        else:
            raise NotImplementedError(f'Mix precision mode {self.mix_precision_mode} is not implemented.')

        # Build EMA params
        if self.is_master:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]

        # Initialize optimizer
        if hasattr(torch.optim, self.optimizer_config['name']):
            self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(self.master_params, **self.optimizer_config['args'])
        else:
            self.optimizer = globals()[self.optimizer_config['name']](self.master_params, **self.optimizer_config['args'])
        
        # Initalize learning rate scheduler
        if self.lr_scheduler_config is not None:
            if hasattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name']):
                self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name'])(self.optimizer, **self.lr_scheduler_config['args'])
            else:
                self.lr_scheduler = globals()[self.lr_scheduler_config['name']](self.optimizer, **self.lr_scheduler_config['args'])

        # Initialize elastic memory controller
        if self.elastic_controller_config is not None:
            assert any([isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)) for model in self.models.values()]), \
                'No elastic module found in models, please inherit from ElasticModule or ElasticModuleMixin'
            self.elastic_controller = getattr(elastic_utils, self.elastic_controller_config['name'])(**self.elastic_controller_config['args'])
            for model in self.models.values():
                if isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)):
                    model.register_memory_controller(self.elastic_controller)

        # Initialize gradient clipper
        if self.grad_clip is not None:
            if isinstance(self.grad_clip, (float, int)):
                self.grad_clip = float(self.grad_clip)
            else:
                self.grad_clip = getattr(grad_clip_utils, self.grad_clip['name'])(**self.grad_clip['args'])

    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        Supports optional weighted sampling via kwargs:
            sampler: {"name": "weighted", ...} to use WeightedResumableSampler
        """
        sampler_config = kwargs.get('sampler', None)
        if sampler_config is not None and sampler_config.get('name') == 'weighted':
            assert hasattr(self.dataset, 'sample_weights'), \
                'Dataset must have "sample_weights" attribute for weighted sampling'
            self.data_sampler = WeightedResumableSampler(
                self.dataset,
                shuffle=True,
            )
            if self.is_master:
                print(f'[Trainer] Using WeightedResumableSampler for area-based oversampling')
        else:
            self.data_sampler = ResumableSampler(
                self.dataset,
                shuffle=True,
            )
        num_workers = min(int(np.ceil(os.cpu_count() / torch.cuda.device_count())), 8)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=f'cuda:{self.local_rank}',
            drop_last=True,
            persistent_workers=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def _master_params_to_state_dicts(self, master_params):
        """
        Convert master params to dict of state_dicts.
        """
        if self.mix_precision_mode == 'inflat_all':
            master_params = unflatten_master_params(self.model_params, master_params)
        state_dicts = {name: model.state_dict() for name, model in self.models.items()}
        master_params_names = sum(
            [[(name, n) for n, p in model.named_parameters() if p.requires_grad] for name, model in self.models.items()]
        , [])
        for i, (model_name, param_name) in enumerate(master_params_names):
            state_dicts[model_name][param_name] = master_params[i]
        return state_dicts

    def _state_dicts_to_master_params(self, master_params, state_dicts):
        """
        Convert a state_dict to master params.
        """
        master_params_names = sum(
            [[(name, n) for n, p in model.named_parameters() if p.requires_grad] for name, model in self.models.items()]
        , [])
        params = [state_dicts[name][param_name] for name, param_name in master_params_names]
        if self.mix_precision_mode == 'inflat_all':
            model_params_to_master_params(params, master_params)
        else:
            for i, param in enumerate(params):
                master_params[i].data.copy_(param.data)

    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print(f'\nLoading checkpoint from step {step}...', end='')
            
        model_ckpts = {}
        for name, model in self.models.items():
            model_ckpt = torch.load(read_file_dist(os.path.join(load_dir, 'ckpts', f'{name}_step{step:07d}.pt')), map_location=self.device, weights_only=True)
            model_ckpts[name] = model_ckpt
            model.load_state_dict(model_ckpt)
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpts = {}
                for name, model in self.models.items():
                    ema_ckpt = torch.load(os.path.join(load_dir, 'ckpts', f'{name}_ema{ema_rate}_step{step:07d}.pt'), map_location=self.device, weights_only=True)
                    ema_ckpts[name] = ema_ckpt
                self._state_dicts_to_master_params(self.ema_params[i], ema_ckpts)
                del ema_ckpts
        
        misc_ckpt = torch.load(read_file_dist(os.path.join(load_dir, 'ckpts', f'misc_step{step:07d}.pt')), map_location=torch.device('cpu'), weights_only=False)
        self.optimizer.load_state_dict(misc_ckpt['optimizer'])
        self.step = misc_ckpt['step']
        self.data_sampler.load_state_dict(misc_ckpt['data_sampler'])
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            self.scaler.load_state_dict(misc_ckpt['scaler'])
        elif self.mix_precision_mode == 'inflat_all' and self.mix_precision_dtype == torch.float16:
            self.log_scale = misc_ckpt['log_scale']
        if self.lr_scheduler_config is not None:
            self.lr_scheduler.load_state_dict(misc_ckpt['lr_scheduler'])
        if self.elastic_controller_config is not None:
            self.elastic_controller.load_state_dict(misc_ckpt['elastic_controller'])
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            self.grad_clip.load_state_dict(misc_ckpt['grad_clip'])
        del misc_ckpt

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(' Done.')

        if self.world_size > 1:
            self.check_ddp()

    def save(self, non_blocking=True):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """
        assert self.is_master, 'save() should be called only by the rank 0 process.'
        print(f'\nSaving checkpoint at step {self.step}...', end='')
        
        model_ckpts = self._master_params_to_state_dicts(self.master_params)
        for name, model_ckpt in model_ckpts.items():
            model_ckpt = {k: v.cpu() for k, v in model_ckpt.items()}  # Move to CPU for saving
            if non_blocking:
                threading.Thread(
                    target=torch.save,
                    args=(model_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_step{self.step:07d}.pt')),
                ).start()
            else:
                torch.save(model_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_step{self.step:07d}.pt'))
        
        for i, ema_rate in enumerate(self.ema_rate):
            ema_ckpts = self._master_params_to_state_dicts(self.ema_params[i])
            for name, ema_ckpt in ema_ckpts.items():
                ema_ckpt = {k: v.cpu() for k, v in ema_ckpt.items()}  # Move to CPU for saving
                if non_blocking:
                    threading.Thread(
                        target=torch.save,
                        args=(ema_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_ema{ema_rate}_step{self.step:07d}.pt')),
                    ).start()
                else:
                    torch.save(ema_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_ema{ema_rate}_step{self.step:07d}.pt'))

        misc_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'data_sampler': self.data_sampler.state_dict(),
        }
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            misc_ckpt['scaler'] = self.scaler.state_dict()
        elif self.mix_precision_mode == 'inflat_all' and self.mix_precision_dtype == torch.float16:
            misc_ckpt['log_scale'] = self.log_scale
        if self.lr_scheduler_config is not None:
            misc_ckpt['lr_scheduler'] = self.lr_scheduler.state_dict()
        if self.elastic_controller_config is not None:
            misc_ckpt['elastic_controller'] = self.elastic_controller.state_dict()
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            misc_ckpt['grad_clip'] = self.grad_clip.state_dict()
        if non_blocking:
            threading.Thread(
                target=torch.save,
                args=(misc_ckpt, os.path.join(self.output_dir, 'ckpts', f'misc_step{self.step:07d}.pt')),
            ).start()
        else:
            torch.save(misc_ckpt, os.path.join(self.output_dir, 'ckpts', f'misc_step{self.step:07d}.pt'))
        print(' Done.')

    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print('\nFinetuning from:')
            for name, path in finetune_ckpt.items():
                print(f'  - {name}: {path}')
        
        model_ckpts = {}
        for name, model in self.models.items():
            model_state_dict = model.state_dict()
            if name in finetune_ckpt:
                ckpt_path = finetune_ckpt[name]
                if ckpt_path.endswith('.safetensors'):
                    from safetensors.torch import load as safetensors_load
                    data = read_file_dist(ckpt_path)
                    model_ckpt = safetensors_load(data.read())
                    model_ckpt = {k: v.to(self.device) for k, v in model_ckpt.items()}
                else:
                    model_ckpt = torch.load(read_file_dist(ckpt_path), map_location=self.device, weights_only=True)
                for k, v in model_ckpt.items():
                    if k not in model_state_dict:
                        if self.is_master:
                            print(f'Warning: {k} not found in model_state_dict, skipped.')
                        model_ckpt[k] = None
                    elif model_ckpt[k].shape != model_state_dict[k].shape:
                        if self.is_master:
                            print(f'Warning: {k} shape mismatch, {model_ckpt[k].shape} vs {model_state_dict[k].shape}, skipped.')
                        model_ckpt[k] = model_state_dict[k]
                model_ckpt = {k: v for k, v in model_ckpt.items() if v is not None}
                missing_keys = [k for k in model_state_dict if k not in model_ckpt]
                if missing_keys and self.is_master:
                    print(f'Warning: {len(missing_keys)} key(s) missing from checkpoint (will use model init):')
                    for k in missing_keys:
                        print(f'  - {k}')
                # Add missing keys from model's current state (initialized values)
                for k in missing_keys:
                    model_ckpt[k] = model_state_dict[k]
                model_ckpts[name] = model_ckpt
                model.load_state_dict(model_ckpt, strict=False)
            else:
                if self.is_master:
                    print(f'Warning: {name} not found in finetune_ckpt, skipped.')
                model_ckpts[name] = model_state_dict
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                self._state_dicts_to_master_params(self.ema_params[i], model_ckpts)
        del model_ckpts

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print('Done.')

        if self.world_size > 1:
            self.check_ddp()

    @abstractmethod
    def run_snapshot(self, num_samples, batch_size=4, verbose=False, **kwargs):
        """
        Run a snapshot of the model.
        """
        pass

    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to an image.
        """
        if hasattr(self.dataset, 'visualize_sample'):
            return self.dataset.visualize_sample(sample)
        else:
            return sample

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=16, batch_size=4): # num_samples=16
        """
        Sample images from the dataset (training + eval if available).
        """
        # Visualize training dataset
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=1,
            shuffle=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        save_cfg = {}
        has_topdown = hasattr(self.dataset, 'visualize_sample_topdown')
        has_topdown_cam = hasattr(self.dataset, 'visualize_sample_topdown_camera_center')
        has_interior = hasattr(self.dataset, 'visualize_sample_interior')
        has_parts_topdown = hasattr(self.dataset, 'visualize_sample_parts_topdown')
        has_bbox_proj = hasattr(self.dataset, 'visualize_bbox_projection')
        has_cross_attn_mask = hasattr(self.dataset, 'visualize_cross_attn_mask')
        topdown_images = []
        topdown_cam_images = []
        interior_images = []
        parts_topdown_composites = []  # List of (sample_path, composite_tensor, camera_center)
        bbox_proj_images = []  # List of (sample_name, PIL.Image)
        cross_attn_mask_images = []  # List of (sample_name, PIL.Image)
        # Metadata for trainer's _visualize_cross_attn_mask (spatial/asset-aware heatmaps)
        snapshot_camera_centers = []
        snapshot_obbs = []
        snapshot_asset_names = []
        snapshot_sample_paths = []
        snapshot_gt_part_coords = []  # Per-part GT sparse voxel coords: list of [list of tensors per part]
        for i in tqdm(range(0, num_samples, batch_size), desc='Snapshot dataset', total=num_samples // batch_size):
            data = next(iter(dataloader)) # data.keys() = dict_keys(['part_layouts', 'obbs', 'asset_names', 'camera_center', 'cond', 'x_0', 'sample_paths', 'n_visible_assets'])
            data = {k: v[:min(num_samples - i, batch_size)] for k, v in data.items()} # data['x_0'].shape = SparseTensor([4,33])
            data = recursive_to_device(data, self.device)
            # Collect metadata for cross-attn mask visualization
            if 'camera_center' in data and isinstance(data['camera_center'], torch.Tensor):
                snapshot_camera_centers.append(data['camera_center'].cpu())
            if 'obbs' in data and isinstance(data['obbs'], list):
                snapshot_obbs.extend(data['obbs'])
            if 'asset_names' in data and isinstance(data['asset_names'], list):
                snapshot_asset_names.extend(data['asset_names'])
            if 'sample_paths' in data and isinstance(data['sample_paths'], list):
                snapshot_sample_paths.extend(data['sample_paths'])
            # Collect per-part GT sparse voxel coords (overall + each asset)
            if 'x_0' in data and 'part_layouts' in data:
                try:
                    x_0 = data['x_0']
                    batch_layouts = x_0.spatial_cache.get('layout', None) if hasattr(x_0, 'spatial_cache') else None
                    for j in range(len(data['part_layouts'])):
                        sample_parts = []
                        sample_start = batch_layouts[j].start if batch_layouts and j < len(batch_layouts) else 0
                        for part_slice in data['part_layouts'][j]:
                            if batch_layouts and j < len(batch_layouts):
                                global_start = sample_start + part_slice.start
                                global_stop = sample_start + part_slice.stop
                                coords = x_0.coords[global_start:global_stop, 1:].cpu()
                            else:
                                coords = x_0.coords[part_slice, 1:].cpu()
                            sample_parts.append(coords)
                        snapshot_gt_part_coords.append(sample_parts)
                except Exception as e:
                    print(f'\nWarning: Failed to collect GT sparse coords: {e}')
            vis = self.visualize_sample(data) # vis.shape = [4,3,1024,1024]
            if has_topdown:
                topdown_images.append(self.dataset.visualize_sample_topdown(data))
            if has_topdown_cam:
                vis_td_cam = self.dataset.visualize_sample_topdown_camera_center(data)
                if vis_td_cam is not None:
                    topdown_cam_images.append(vis_td_cam)
            if has_interior:
                vis_int = self.dataset.visualize_sample_interior(data)
                if vis_int is not None:
                    interior_images.append(vis_int)
            if has_parts_topdown:
                try:
                    parts_vis = self.dataset.visualize_sample_parts_topdown(data)
                    if parts_vis is not None:
                        sample_paths = data.get('sample_paths', [f'sample_{i+j}' for j in range(len(parts_vis))])
                        cam_centers = data.get('camera_center', None)
                        for j, comp in enumerate(parts_vis):
                            path_name = sample_paths[j] if j < len(sample_paths) else f'sample_{i+j}'
                            if isinstance(path_name, str):
                                path_name = path_name.replace('/', '_')
                            cam_j = cam_centers[j] if cam_centers is not None and j < len(cam_centers) else None
                            parts_topdown_composites.append((path_name, comp, cam_j))
                except Exception as e:
                    print(f'\nWarning: Failed to generate parts topdown vis: {e}')
            if has_bbox_proj:
                try:
                    bp_results = self.dataset.visualize_bbox_projection(data)
                    if bp_results:
                        bbox_proj_images.extend(bp_results)
                except Exception as e:
                    print(f'\nWarning: Failed to generate bbox projection vis: {e}')
            if has_cross_attn_mask:
                try:
                    cam_results = self.dataset.visualize_cross_attn_mask(data)
                    if cam_results:
                        cross_attn_mask_images.extend(cam_results)
                except Exception as e:
                    print(f'\nWarning: Failed to generate cross-attn mask vis: {e}')
            if isinstance(vis, dict):
                for k, v in vis.items():
                    if f'dataset_{k}' not in save_cfg:
                        save_cfg[f'dataset_{k}'] = []
                    save_cfg[f'dataset_{k}'].append(v)
            else:
                if 'dataset' not in save_cfg:
                    save_cfg['dataset'] = []
                save_cfg['dataset'].append(vis)
        for name, image in tqdm(save_cfg.items(), desc='Save snapshot dataset', total=len(save_cfg)):
            utils.save_image(
                    torch.cat(image, dim=0),
                    os.path.join(self.output_dir, 'samples', f'{name}.png'),
                    nrow=int(np.sqrt(num_samples)),
                    normalize=False,
                )
        if has_topdown and len(topdown_images) > 0:
            utils.save_image(
                torch.cat(topdown_images, dim=0),
                os.path.join(self.output_dir, 'samples', 'dataset_topdown.png'),
                nrow=int(np.sqrt(num_samples)),
                normalize=False,
            )
        if has_topdown_cam and len(topdown_cam_images) > 0:
            utils.save_image(
                torch.cat(topdown_cam_images, dim=0),
                os.path.join(self.output_dir, 'samples', 'dataset_topdown_camera_center.png'),
                nrow=int(np.sqrt(num_samples)),
                normalize=False,
            )
        if has_interior and len(interior_images) > 0:
            utils.save_image(
                torch.cat(interior_images, dim=0),
                os.path.join(self.output_dir, 'samples', 'dataset_interior.png'),
                nrow=int(np.sqrt(num_samples)),
                normalize=False,
            )
        # Save per-sample parts topdown composites
        if has_parts_topdown and len(parts_topdown_composites) > 0:
            parts_dir = os.path.join(self.output_dir, 'samples', 'init', 'parts_topdown')
            os.makedirs(parts_dir, exist_ok=True)
            for path_name, comp, _ in parts_topdown_composites:
                utils.save_image(
                    comp,
                    os.path.join(parts_dir, f'{path_name}.png'),
                    normalize=False,
                )
            # Save parts_topdown_camera_center (overlay camera center on existing composites)
            has_cam_center_method = hasattr(self.dataset, 'visualize_sample_parts_topdown_camera_center')
            has_any_cam = any(cam is not None for _, _, cam in parts_topdown_composites)
            if has_cam_center_method and has_any_cam:
                cam_dir = os.path.join(self.output_dir, 'samples', 'init', 'parts_topdown_camera_center')
                os.makedirs(cam_dir, exist_ok=True)
                for path_name, comp, cam_center in parts_topdown_composites:
                    if cam_center is not None:
                        cam_centers_batch = cam_center.unsqueeze(0)  # [1, 3]
                        cam_comps = self.dataset.visualize_sample_parts_topdown_camera_center(
                            [comp], cam_centers_batch,
                        )
                        if cam_comps is not None and len(cam_comps) > 0:
                            utils.save_image(
                                cam_comps[0],
                                os.path.join(cam_dir, f'{path_name}.png'),
                                normalize=False,
                            )

        # Save bbox projection and token selection visualizations
        if bbox_proj_images:
            bp_dir = os.path.join(self.output_dir, 'samples', 'init', '3d_bbox_projection')
            os.makedirs(bp_dir, exist_ok=True)
            for sample_name, pil_img in bbox_proj_images:
                pil_img.save(os.path.join(bp_dir, f'{sample_name}_bbox_projection.png'))
        if cross_attn_mask_images:
            cam_dir = os.path.join(self.output_dir, 'samples', 'init', '3d_bbox_projection')
            os.makedirs(cam_dir, exist_ok=True)
            for sample_name, pil_img in cross_attn_mask_images:
                pil_img.save(os.path.join(cam_dir, f'{sample_name}_token_selection.png'))

        # Visualize cross-attention mask (trainer method - spatial/asset-aware heatmaps)
        if snapshot_camera_centers:
            try:
                metadata = {
                    '_train_camera_center': {'value': torch.cat(snapshot_camera_centers, dim=0), 'type': 'metadata'},
                }
                if snapshot_sample_paths:
                    metadata['_train_paths'] = {'value': snapshot_sample_paths, 'type': 'paths'}
                if snapshot_obbs:
                    metadata['_train_obbs'] = {'value': snapshot_obbs, 'type': 'metadata'}
                if snapshot_asset_names:
                    metadata['_train_asset_names'] = {'value': snapshot_asset_names, 'type': 'metadata'}
                if snapshot_gt_part_coords:
                    metadata['_train_gt_part_coords'] = {'value': snapshot_gt_part_coords, 'type': 'metadata'}
                self._visualize_cross_attn_mask(metadata, 'init')
            except Exception as e:
                print(f'\nWarning: Failed to visualize cross-attn mask (train): {e}')

        # Visualize eval dataset (if available)
        if self.eval_dataset is not None:
            eval_dataloader = torch.utils.data.DataLoader(
                self.eval_dataset,
                batch_size=batch_size,
                num_workers=1,
                shuffle=True,
                collate_fn=self.eval_dataset.collate_fn if hasattr(self.eval_dataset, 'collate_fn') else None,
            )
            eval_save_cfg = {}
            eval_has_topdown = hasattr(self.eval_dataset, 'visualize_sample_topdown')
            eval_has_topdown_cam = hasattr(self.eval_dataset, 'visualize_sample_topdown_camera_center')
            eval_has_interior = hasattr(self.eval_dataset, 'visualize_sample_interior')
            eval_has_parts_topdown = hasattr(self.eval_dataset, 'visualize_sample_parts_topdown')
            eval_has_bbox_proj = hasattr(self.eval_dataset, 'visualize_bbox_projection')
            eval_has_cross_attn_mask = hasattr(self.eval_dataset, 'visualize_cross_attn_mask')
            eval_topdown_images = []
            eval_topdown_cam_images = []
            eval_interior_images = []
            eval_parts_topdown_composites = []
            eval_bbox_proj_images = []
            eval_cross_attn_mask_images = []
            eval_snapshot_camera_centers = []
            eval_snapshot_obbs = []
            eval_snapshot_asset_names = []
            eval_snapshot_sample_paths = []
            eval_snapshot_gt_part_coords = []
            for i in tqdm(range(0, num_samples, batch_size), desc='Snapshot eval dataset', total=num_samples // batch_size):
                data = next(iter(eval_dataloader))
                data = {k: v[:min(num_samples - i, batch_size)] for k, v in data.items()}
                data = recursive_to_device(data, self.device)
                # Collect metadata for cross-attn mask visualization
                if 'camera_center' in data and isinstance(data['camera_center'], torch.Tensor):
                    eval_snapshot_camera_centers.append(data['camera_center'].cpu())
                if 'obbs' in data and isinstance(data['obbs'], list):
                    eval_snapshot_obbs.extend(data['obbs'])
                if 'asset_names' in data and isinstance(data['asset_names'], list):
                    eval_snapshot_asset_names.extend(data['asset_names'])
                if 'sample_paths' in data and isinstance(data['sample_paths'], list):
                    eval_snapshot_sample_paths.extend(data['sample_paths'])
                # Collect per-part GT sparse voxel coords
                if 'x_0' in data and 'part_layouts' in data:
                    try:
                        x_0 = data['x_0']
                        batch_layouts = x_0.spatial_cache.get('layout', None) if hasattr(x_0, 'spatial_cache') else None
                        for j in range(len(data['part_layouts'])):
                            sample_parts = []
                            sample_start = batch_layouts[j].start if batch_layouts and j < len(batch_layouts) else 0
                            for part_slice in data['part_layouts'][j]:
                                if batch_layouts and j < len(batch_layouts):
                                    global_start = sample_start + part_slice.start
                                    global_stop = sample_start + part_slice.stop
                                    coords = x_0.coords[global_start:global_stop, 1:].cpu()
                                else:
                                    coords = x_0.coords[part_slice, 1:].cpu()
                                sample_parts.append(coords)
                            eval_snapshot_gt_part_coords.append(sample_parts)
                    except Exception as e:
                        print(f'\nWarning: Failed to collect eval GT sparse coords: {e}')
                vis = self.visualize_sample(data)
                if eval_has_topdown:
                    eval_topdown_images.append(self.eval_dataset.visualize_sample_topdown(data))
                if eval_has_topdown_cam:
                    vis_td_cam = self.eval_dataset.visualize_sample_topdown_camera_center(data)
                    if vis_td_cam is not None:
                        eval_topdown_cam_images.append(vis_td_cam)
                if eval_has_interior:
                    vis_int = self.eval_dataset.visualize_sample_interior(data)
                    if vis_int is not None:
                        eval_interior_images.append(vis_int)
                if eval_has_parts_topdown:
                    try:
                        parts_vis = self.eval_dataset.visualize_sample_parts_topdown(data)
                        if parts_vis is not None:
                            sample_paths = data.get('sample_paths', [f'sample_{i+j}' for j in range(len(parts_vis))])
                            cam_centers = data.get('camera_center', None)
                            for j, comp in enumerate(parts_vis):
                                path_name = sample_paths[j] if j < len(sample_paths) else f'sample_{i+j}'
                                if isinstance(path_name, str):
                                    path_name = path_name.replace('/', '_')
                                cam_j = cam_centers[j] if cam_centers is not None and j < len(cam_centers) else None
                                eval_parts_topdown_composites.append((path_name, comp, cam_j))
                    except Exception as e:
                        print(f'\nWarning: Failed to generate eval parts topdown vis: {e}')
                if eval_has_bbox_proj:
                    try:
                        bp_results = self.eval_dataset.visualize_bbox_projection(data)
                        if bp_results:
                            eval_bbox_proj_images.extend(bp_results)
                    except Exception as e:
                        print(f'\nWarning: Failed to generate eval bbox projection vis: {e}')
                if eval_has_cross_attn_mask:
                    try:
                        cam_results = self.eval_dataset.visualize_cross_attn_mask(data)
                        if cam_results:
                            eval_cross_attn_mask_images.extend(cam_results)
                    except Exception as e:
                        print(f'\nWarning: Failed to generate eval cross-attn mask vis: {e}')
                if isinstance(vis, dict):
                    for k, v in vis.items():
                        if f'eval_dataset_{k}' not in eval_save_cfg:
                            eval_save_cfg[f'eval_dataset_{k}'] = []
                        eval_save_cfg[f'eval_dataset_{k}'].append(v)
                else:
                    if 'eval_dataset' not in eval_save_cfg:
                        eval_save_cfg['eval_dataset'] = []
                    eval_save_cfg['eval_dataset'].append(vis)
            # value_range = self.eval_dataset.value_range if hasattr(self.eval_dataset, 'value_range') else self.dataset.value_range
            for name, image in tqdm(eval_save_cfg.items(), desc='Save snapshot eval dataset', total=len(eval_save_cfg)):
                utils.save_image(
                    torch.cat(image, dim=0),
                    os.path.join(self.output_dir, 'samples', f'{name}.png'),
                    nrow=int(np.sqrt(num_samples)),
                    normalize=False,
                )
            if eval_has_topdown:
                utils.save_image(
                    torch.cat(eval_topdown_images, dim=0),
                    os.path.join(self.output_dir, 'samples', 'eval_dataset_topdown.png'),
                    nrow=int(np.sqrt(num_samples)),
                    normalize=False,
                )
            if eval_has_topdown_cam and len(eval_topdown_cam_images) > 0:
                utils.save_image(
                    torch.cat(eval_topdown_cam_images, dim=0),
                    os.path.join(self.output_dir, 'samples', 'eval_dataset_topdown_camera_center.png'),
                    nrow=int(np.sqrt(num_samples)),
                    normalize=False,
                )
            if eval_has_interior and len(eval_interior_images) > 0:
                utils.save_image(
                    torch.cat(eval_interior_images, dim=0),
                    os.path.join(self.output_dir, 'samples', 'eval_dataset_interior.png'),
                    nrow=int(np.sqrt(num_samples)),
                    normalize=False,
                )
            # Save per-sample eval parts topdown composites
            if eval_parts_topdown_composites:
                eval_parts_dir = os.path.join(self.output_dir, 'samples', 'init', 'eval_parts_topdown')
                os.makedirs(eval_parts_dir, exist_ok=True)
                for path_name, comp, _ in eval_parts_topdown_composites:
                    utils.save_image(
                        comp,
                        os.path.join(eval_parts_dir, f'{path_name}.png'),
                        normalize=False,
                    )
                # Save eval parts_topdown_camera_center
                eval_cam_method = hasattr(self.eval_dataset, 'visualize_sample_parts_topdown_camera_center')
                eval_has_any_cam = any(cam is not None for _, _, cam in eval_parts_topdown_composites)
                if eval_cam_method and eval_has_any_cam:
                    eval_cam_dir = os.path.join(self.output_dir, 'samples', 'init', 'eval_parts_topdown_camera_center')
                    os.makedirs(eval_cam_dir, exist_ok=True)
                    for path_name, comp, cam_center in eval_parts_topdown_composites:
                        if cam_center is not None:
                            cam_centers_batch = cam_center.unsqueeze(0)
                            cam_comps = self.eval_dataset.visualize_sample_parts_topdown_camera_center(
                                [comp], cam_centers_batch,
                            )
                            if cam_comps is not None and len(cam_comps) > 0:
                                utils.save_image(
                                    cam_comps[0],
                                    os.path.join(eval_cam_dir, f'{path_name}.png'),
                                    normalize=False,
                                )
            # Save eval bbox projection and token selection visualizations
            if eval_bbox_proj_images:
                bp_dir = os.path.join(self.output_dir, 'samples', 'init', 'eval_3d_bbox_projection')
                os.makedirs(bp_dir, exist_ok=True)
                for sample_name, pil_img in eval_bbox_proj_images:
                    pil_img.save(os.path.join(bp_dir, f'{sample_name}_bbox_projection.png'))
            if eval_cross_attn_mask_images:
                cam_dir = os.path.join(self.output_dir, 'samples', 'init', 'eval_3d_bbox_projection')
                os.makedirs(cam_dir, exist_ok=True)
                for sample_name, pil_img in eval_cross_attn_mask_images:
                    pil_img.save(os.path.join(cam_dir, f'{sample_name}_token_selection.png'))

            # Visualize cross-attention mask (trainer method - spatial/asset-aware heatmaps)
            if eval_snapshot_camera_centers:
                try:
                    eval_metadata = {
                        '_eval_camera_center': {'value': torch.cat(eval_snapshot_camera_centers, dim=0), 'type': 'metadata'},
                    }
                    if eval_snapshot_sample_paths:
                        eval_metadata['_eval_paths'] = {'value': eval_snapshot_sample_paths, 'type': 'paths'}
                    if eval_snapshot_obbs:
                        eval_metadata['_eval_obbs'] = {'value': eval_snapshot_obbs, 'type': 'metadata'}
                    if eval_snapshot_asset_names:
                        eval_metadata['_eval_asset_names'] = {'value': eval_snapshot_asset_names, 'type': 'metadata'}
                    if eval_snapshot_gt_part_coords:
                        eval_metadata['_eval_gt_part_coords'] = {'value': eval_snapshot_gt_part_coords, 'type': 'metadata'}
                    self._visualize_cross_attn_mask(eval_metadata, 'init')
                except Exception as e:
                    print(f'\nWarning: Failed to visualize cross-attn mask (eval): {e}')

    @torch.no_grad()
    def _generate_additional_vis(self, samples, metadata, dataset, prefix_filter=None):
        """
        Generate topdown/topdown_camera_center/interior visualizations for GT and predicted samples.

        Args:
            samples: dict of {'key': {'value': tensor, 'type': str}} from run_snapshot
            metadata: dict of metadata entries extracted from samples (camera_center, cond, paths)
            dataset: dataset instance with visualization methods
            prefix_filter: 'train' or 'eval' to process only that split. None for all.
        """
        has_topdown = hasattr(dataset, 'visualize_sample_topdown')
        has_topdown_cam = hasattr(dataset, 'visualize_sample_topdown_camera_center')
        has_interior = hasattr(dataset, 'visualize_sample_interior')

        if not (has_topdown or has_topdown_cam or has_interior):
            return

        # Prefix mapping: 'train' -> ('train_', '_train_'), 'eval' -> ('eval_', '_eval_')
        all_prefixes = [('train', 'train_', '_train_'), ('eval', 'eval_', '_eval_')]
        for split_name, prefix, meta_prefix in all_prefixes:
            if prefix_filter is not None and split_name != prefix_filter:
                continue
            gt_key = f'{prefix}sample_gt'
            pred_key = f'{prefix}sample'

            if gt_key not in samples or samples[gt_key]['type'] != 'sample':
                continue

            cam_center = metadata.get(f'{meta_prefix}camera_center', {}).get('value', None)
            raw_cond = metadata.get(f'{meta_prefix}cond', {}).get('value', None)

            # Generate vis for both GT and predicted samples
            for src_key, vis_label in [(gt_key, f'{prefix}gt'), (pred_key, f'{prefix}pred')]:
                if src_key not in samples or samples[src_key]['type'] != 'sample':
                    continue

                value = samples[src_key]['value']
                if isinstance(value, dict):
                    # Stage 2: run_snapshot returns full data dicts as sample values
                    # (contains x_0, cond, camera_center, part_layouts, etc.)
                    data_dict = value
                else:
                    # Stage 1: run_snapshot returns plain tensors
                    data_dict = {'x_0': value}
                    if cam_center is not None:
                        data_dict['camera_center'] = cam_center.cuda()
                    if raw_cond is not None:
                        data_dict['cond'] = raw_cond  # kept on CPU, vis functions move per-sample

                # Check if camera_center/cond available (either from metadata or inside data_dict)
                has_cam = cam_center is not None or 'camera_center' in data_dict
                has_cond = raw_cond is not None or 'cond' in data_dict

                try:
                    if has_topdown:
                        td_img = dataset.visualize_sample_topdown(data_dict)
                        if td_img is not None:
                            if isinstance(td_img, dict):
                                for k, v in td_img.items():
                                    samples[f'{vis_label}_topdown_{k}'] = {'value': v, 'type': 'image'}
                            else:
                                samples[f'{vis_label}_topdown'] = {'value': td_img, 'type': 'image'}

                    if has_topdown_cam and has_cam:
                        td_cam_img = dataset.visualize_sample_topdown_camera_center(data_dict)
                        if td_cam_img is not None:
                            if isinstance(td_cam_img, dict):
                                for k, v in td_cam_img.items():
                                    samples[f'{vis_label}_topdown_cam_{k}'] = {'value': v, 'type': 'image'}
                            else:
                                samples[f'{vis_label}_topdown_cam'] = {'value': td_cam_img, 'type': 'image'}

                    if has_interior and has_cam and has_cond:
                        int_img = dataset.visualize_sample_interior(data_dict)
                        if int_img is not None:
                            if isinstance(int_img, dict):
                                for k, v in int_img.items():
                                    samples[f'{vis_label}_interior_{k}'] = {'value': v, 'type': 'image'}
                            else:
                                samples[f'{vis_label}_interior'] = {'value': int_img, 'type': 'image'}
                except Exception as e:
                    print(f'\nWarning: Failed to generate additional vis for {vis_label}: {e}')

    @torch.no_grad()
    def _generate_per_sample_vis(self, samples, metadata, dataset, suffix, prefix_filter=None):
        """
        Generate per-sample visualizations (parts_topdown, bbox_projection, cross_attn_mask)
        and save them individually under samples/{suffix}/.

        These are the same visualizations that snapshot_dataset() saves under samples/init/,
        but generated at every checkpoint step.

        Args:
            prefix_filter: 'train' or 'eval' to process only that split. None for all.
        """
        has_parts_topdown = hasattr(dataset, 'visualize_sample_parts_topdown')
        has_parts_topdown_cam = hasattr(dataset, 'visualize_sample_parts_topdown_camera_center')
        has_bbox_proj = hasattr(dataset, 'visualize_bbox_projection')
        has_cross_attn_mask = hasattr(dataset, 'visualize_cross_attn_mask')

        if not (has_parts_topdown or has_bbox_proj or has_cross_attn_mask):
            return

        # Prefix mapping: 'train' -> ('train_', '_train_'), 'eval' -> ('eval_', '_eval_')
        all_prefixes = [('train', 'train_', '_train_'), ('eval', 'eval_', '_eval_')]
        for split_name, prefix, meta_prefix in all_prefixes:
            if prefix_filter is not None and split_name != prefix_filter:
                continue
            gt_key = f'{prefix}sample_gt'
            if gt_key not in samples or samples[gt_key]['type'] != 'sample':
                continue

            value = samples[gt_key]['value']
            if not isinstance(value, dict):
                continue  # Per-sample vis only for Stage 2 (dict data)

            data_dict = value
            cam_center = metadata.get(f'{meta_prefix}camera_center', {}).get('value', None)
            paths = metadata.get(f'{meta_prefix}paths', {}).get('value', None)
            label = f'{split_name}_'  # 'train_' or 'eval_'

            # Parts topdown: GT + Pred comparison
            if has_parts_topdown:
                try:
                    sample_paths_list = data_dict.get('sample_paths', paths or [])
                    cam_centers = data_dict.get('camera_center', None)

                    # Generate GT parts
                    gt_parts_vis = dataset.visualize_sample_parts_topdown(data_dict)

                    # Generate Pred parts
                    pred_key = f'{prefix}sample'
                    pred_parts_vis = None
                    if pred_key in samples and samples[pred_key]['type'] == 'sample':
                        pred_value = samples[pred_key]['value']
                        if isinstance(pred_value, dict) and 'part_layouts' in pred_value:
                            pred_parts_vis = dataset.visualize_sample_parts_topdown(pred_value)

                    # Save pred parts topdown
                    if pred_parts_vis is not None:
                        pred_parts_dir = os.path.join(self.output_dir, 'samples', suffix, f'{label}pred_parts_topdown')
                        os.makedirs(pred_parts_dir, exist_ok=True)
                        pred_composites = []
                        for j, comp in enumerate(pred_parts_vis):
                            path_name = sample_paths_list[j] if j < len(sample_paths_list) else f'sample_{j}'
                            if isinstance(path_name, str):
                                path_name = path_name.replace('/', '_')
                            cam_j = cam_centers[j] if cam_centers is not None and j < len(cam_centers) else None
                            pred_composites.append((path_name, comp, cam_j))
                            utils.save_image(comp, os.path.join(pred_parts_dir, f'{path_name}.png'), normalize=False)

                        # Save pred parts topdown with camera center
                        if has_parts_topdown_cam and any(c is not None for _, _, c in pred_composites):
                            pred_cam_dir = os.path.join(self.output_dir, 'samples', suffix, f'{label}pred_parts_topdown_camera_center')
                            os.makedirs(pred_cam_dir, exist_ok=True)
                            for path_name, comp, cam_j in pred_composites:
                                if cam_j is not None:
                                    cam_comps = dataset.visualize_sample_parts_topdown_camera_center(
                                        [comp], cam_j.unsqueeze(0),
                                    )
                                    if cam_comps is not None and len(cam_comps) > 0:
                                        utils.save_image(cam_comps[0], os.path.join(pred_cam_dir, f'{path_name}.png'), normalize=False)

                    # Save GT vs Pred comparison
                    if gt_parts_vis is not None and pred_parts_vis is not None:
                        comp_dir = os.path.join(self.output_dir, 'samples', suffix, 'concat', f'{label}parts_comparison')
                        os.makedirs(comp_dir, exist_ok=True)
                        for j in range(min(len(gt_parts_vis), len(pred_parts_vis))):
                            gt_comp = gt_parts_vis[j]
                            pred_comp = pred_parts_vis[j]
                            # Match widths (pad shorter one with gray)
                            max_w = max(gt_comp.shape[2], pred_comp.shape[2])
                            if gt_comp.shape[2] < max_w:
                                gt_comp = F.pad(gt_comp, (0, max_w - gt_comp.shape[2]), value=0.3)
                            if pred_comp.shape[2] < max_w:
                                pred_comp = F.pad(pred_comp, (0, max_w - pred_comp.shape[2]), value=0.3)
                            # Row labels
                            gt_h, pred_h = gt_comp.shape[1], pred_comp.shape[1]
                            label_col = self._make_row_label_column(
                                ['GT', 'Pred'], [gt_h, pred_h], 40
                            ).to(gt_comp.device)
                            content = torch.cat([gt_comp, pred_comp], dim=1)
                            comparison = torch.cat([label_col, content], dim=2)
                            path_name = sample_paths_list[j] if j < len(sample_paths_list) else f'sample_{j}'
                            if isinstance(path_name, str):
                                path_name = path_name.replace('/', '_')
                            utils.save_image(comparison, os.path.join(comp_dir, f'{path_name}.png'), normalize=False)

                except Exception as e:
                    print(f'\nWarning: Failed to generate parts topdown for {label}snapshot: {e}')

            # Bbox projection + token selection
            if has_bbox_proj:
                try:
                    bp_results = dataset.visualize_bbox_projection(data_dict)
                    if bp_results:
                        bp_dir = os.path.join(self.output_dir, 'samples', suffix, f'{label}3d_bbox_projection')
                        os.makedirs(bp_dir, exist_ok=True)
                        for sample_name, pil_img in bp_results:
                            pil_img.save(os.path.join(bp_dir, f'{sample_name}_bbox_projection.png'))
                except Exception as e:
                    print(f'\nWarning: Failed to generate bbox projection for {label}snapshot: {e}')

            if has_cross_attn_mask:
                try:
                    cam_results = dataset.visualize_cross_attn_mask(data_dict)
                    if cam_results:
                        cam_dir = os.path.join(self.output_dir, 'samples', suffix, f'{label}3d_bbox_projection')
                        os.makedirs(cam_dir, exist_ok=True)
                        for sample_name, pil_img in cam_results:
                            pil_img.save(os.path.join(cam_dir, f'{sample_name}_token_selection.png'))
                except Exception as e:
                    print(f'\nWarning: Failed to generate cross-attn mask for {label}snapshot: {e}')

    @torch.no_grad()
    def _visualize_cross_attn_mask(self, metadata, suffix):
        """
        Visualize cross-attention mask per sample.

        Supports both Stage 1 (sparse structure) and Stage 2 (structured latent).

        Stage 1 (use_spatial_attention=True):
          - mask_heatmap.jpg: 2D [4096, 6*tokens_per_face] spatial mask
          - mask_3d_voxel.jpg: 2x3 grid of 3D scatter showing active voxels per face (dense 16³ grid)

        Stage 2 (use_asset_aware_attention=True):
          - cross_mask_heatmap_asset_aware.jpg: Per-part token selection (overall spatial + assets bbox)
          Dense grid (all voxel_resolution³ positions):
          - mask_3d_voxel.jpg: 2x3 per-face active voxels (dense 32³ grid)
          - mask_voxel_topdown.jpg: top-down view, all faces combined (dense grid)
          - mask_voxel_camera.jpg: camera-perspective view (dense grid)
          Active voxels (actual GT sparse structure):
          - mask_active_voxels.jpg: grid of overall + per-part active voxels, top-down, face-colored

        Saved under: samples/{suffix}/cross_attn_mask_viz/{sample_name}/
        """
        is_stage1 = getattr(self, 'use_spatial_attention', False)
        is_stage2 = getattr(self, 'use_asset_aware_attention', False)

        if not (is_stage1 or is_stage2):
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            from .flow_matching.mixins.erp_image_conditioned import create_spatial_attention_mask
        except ImportError:
            return

        tokens_per_face = getattr(self, 'tokens_per_face', 1029)
        fov_degrees = getattr(self, 'spatial_attention_fov', None) or getattr(self, 'fov_degrees', 120.0)
        voxel_resolution = getattr(self, 'voxel_resolution', 16)
        soft_mask = getattr(self, 'spatial_attention_soft', True)
        soft_margin = getattr(self, 'spatial_attention_soft_margin', 0.1)

        face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        face_colors = ['#2ecc71', '#e67e22', '#3498db', '#e91e63', '#00bcd4', '#795548']
        face_dirs = np.array([
            [0,1,0], [1,0,0], [0,-1,0], [-1,0,0], [0,0,1], [0,0,-1]
        ], dtype=float)

        # Voxel grid positions (voxel_resolution³: 16³ for Stage 1, 32³ for Stage 2)
        coords = np.linspace(-1 + 1/voxel_resolution, 1 - 1/voxel_resolution, voxel_resolution)
        xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
        voxel_pos = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

        cmap_2d = ListedColormap(['#3498db', '#e74c3c'])

        for meta_prefix in ['_train_', '_eval_']:
            cam_key = f'{meta_prefix}camera_center'
            paths_key = f'{meta_prefix}paths'
            latent_key = f'{meta_prefix}gt_latent'
            obbs_key = f'{meta_prefix}obbs'
            asset_names_key = f'{meta_prefix}asset_names'
            has_layout_key = f'{meta_prefix}has_layout'

            if cam_key not in metadata:
                continue

            camera_centers = metadata[cam_key]['value']  # [B, 3]
            paths = metadata.get(paths_key, {}).get('value', None)
            gt_latents = metadata.get(latent_key, {}).get('value', None)
            obbs_list = metadata.get(obbs_key, {}).get('value', None)
            asset_names_list = metadata.get(asset_names_key, {}).get('value', None)
            has_layout_list = metadata.get(has_layout_key, {}).get('value', None)
            B = camera_centers.shape[0]

            for i in range(B):
                if paths and i < len(paths):
                    sample_name = paths[i].replace('/', '_')
                else:
                    prefix_label = 'train' if meta_prefix == '_train_' else 'eval'
                    sample_name = f'{prefix_label}_sample_{i:03d}'

                save_dir = os.path.join(
                    self.output_dir, 'samples', suffix, 'cross_attn_mask_viz', sample_name
                )
                os.makedirs(save_dir, exist_ok=True)

                cam = camera_centers[i]  # [3]
                cam_np = cam.numpy() if hasattr(cam, 'numpy') else np.array(cam)

                # === Stage 1: Spatial attention mask (dense 16³ grid) ===
                if is_stage1:
                    mask = create_spatial_attention_mask(
                        camera_center=cam,
                        voxel_resolution=voxel_resolution,
                        tokens_per_face=tokens_per_face,
                        fov_degrees=fov_degrees,
                        soft_mask=soft_mask,
                        soft_margin=soft_margin,
                        device='cpu'
                    ).numpy()  # [4096, 6*tokens_per_face]
                    mask_binary = (mask > -1e3).astype(float)

                    # Per-face active mask at 16³ level
                    face_active_16 = []
                    for f_idx in range(6):
                        start = f_idx * tokens_per_face
                        end = (f_idx + 1) * tokens_per_face
                        active = mask_binary[:, start:end].any(axis=1)
                        face_active_16.append(active.reshape(voxel_resolution, voxel_resolution, voxel_resolution).astype(bool))

                    # --- Heatmap ---
                    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
                    ax.imshow(mask_binary, aspect='auto', cmap=cmap_2d,
                              interpolation='nearest', vmin=0, vmax=1)
                    for f_idx in range(1, 6):
                        ax.axvline(x=f_idx * tokens_per_face, color='white',
                                   linewidth=1, linestyle='--', alpha=0.8)
                    for f_idx in range(6):
                        cx = f_idx * tokens_per_face + tokens_per_face // 2
                        ax.text(cx, 50, face_names[f_idx].upper(), ha='center', va='top',
                                fontsize=9, fontweight='bold', color='white',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='#2c3e50', alpha=0.8))
                    ax.set_title(f'Cross-Attn Mask | cam=({cam_np[0]:.2f}, {cam_np[1]:.2f}, {cam_np[2]:.2f})')
                    ax.set_xlabel(f'Tokens (6 x {tokens_per_face})')
                    ax.set_ylabel(f'Voxels ({voxel_resolution}³)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'mask_heatmap.jpg'), dpi=300, bbox_inches='tight')
                    plt.close()

                    # --- 3D Voxel Visualization (2x3 grid) ---
                    fig = plt.figure(figsize=(18, 12))
                    for f_idx in range(6):
                        ax3d = fig.add_subplot(2, 3, f_idx + 1, projection='3d')
                        ax3d.set_box_aspect([1, 1, 1])
                        active_idx = np.where(face_active_16[f_idx].ravel())[0]
                        for s in [-1, 1]:
                            for t in [-1, 1]:
                                ax3d.plot([-1,1],[s,s],[t,t], 'gray', alpha=0.2, lw=0.5)
                                ax3d.plot([s,s],[-1,1],[t,t], 'gray', alpha=0.2, lw=0.5)
                                ax3d.plot([s,s],[t,t],[-1,1], 'gray', alpha=0.2, lw=0.5)
                        if len(active_idx) > 0:
                            apos = voxel_pos[active_idx]
                            ax3d.scatter(apos[:,0], apos[:,1], apos[:,2],
                                         c=face_colors[f_idx], s=15, alpha=0.6, marker='s')
                        ax3d.scatter([cam_np[0]], [cam_np[1]], [cam_np[2]],
                                    c='red', s=80, marker='o', zorder=10)
                        fd = face_dirs[f_idx]
                        ax3d.quiver(cam_np[0], cam_np[1], cam_np[2],
                                    fd[0]*0.4, fd[1]*0.4, fd[2]*0.4,
                                    color=face_colors[f_idx], arrow_length_ratio=0.3, linewidth=2)
                        pct = 100 * len(active_idx) / len(voxel_pos)
                        ax3d.set_title(f'{face_names[f_idx].upper()} ({len(active_idx)}, {pct:.0f}%)', fontsize=10)
                        ax3d.set_xlim([-1.2, 1.2]); ax3d.set_ylim([-1.2, 1.2]); ax3d.set_zlim([-1.2, 1.2])
                        ax3d.set_xticklabels([]); ax3d.set_yticklabels([]); ax3d.set_zticklabels([])
                    plt.suptitle(
                        f'Active Voxels per Face | cam=({cam_np[0]:.2f}, {cam_np[1]:.2f}, {cam_np[2]:.2f})',
                        fontsize=12
                    )
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'mask_3d_voxel.jpg'), dpi=300, bbox_inches='tight')
                    plt.close()

                # === Stage 2: Asset-aware cross-attention mask ===
                if is_stage2:
                    # Compute face-level spatial mask at voxel_resolution³ (e.g. 32³)
                    # Optimized: compute [N_voxels, 6] face mask directly instead of
                    # full [N_voxels, 6*tokens_per_face] to avoid ~800MB tensor at 32³
                    import torch.nn.functional as F_spatial
                    cam_t = cam.cpu().float()
                    if cam_t.dim() == 1:
                        cam_t = cam_t.unsqueeze(0)
                    coords_t = torch.linspace(-1 + 1/voxel_resolution, 1 - 1/voxel_resolution, voxel_resolution)
                    xx_t, yy_t, zz_t = torch.meshgrid(coords_t, coords_t, coords_t, indexing='ij')
                    vp_t = torch.stack([xx_t, yy_t, zz_t], dim=-1).reshape(-1, 3)  # [N, 3]
                    dirs_t = F_spatial.normalize(vp_t.unsqueeze(0) - cam_t.unsqueeze(1), p=2, dim=-1)  # [1, N, 3]
                    face_dirs_t = torch.tensor([
                        [0,1,0],[1,0,0],[0,-1,0],[-1,0,0],[0,0,1],[0,0,-1]
                    ], dtype=torch.float32)
                    cos_sim = torch.einsum('bnd,fd->bnf', dirs_t, face_dirs_t).squeeze(0).numpy()  # [N, 6]
                    fov_rad = np.radians(fov_degrees)
                    cos_threshold = np.cos(fov_rad / 2)

                    # Per-face active mask at voxel_resolution³ level
                    face_active_16 = []
                    for f_idx in range(6):
                        active = cos_sim[:, f_idx] >= cos_threshold
                        face_active_16.append(active.reshape(voxel_resolution, voxel_resolution, voxel_resolution))

                    # Face-level density for heatmap: fraction of voxels visible per face
                    face_density = (cos_sim >= cos_threshold).astype(float).mean(axis=0)  # [6]

                    # Dominant face assignment
                    spatial_face_active = (cos_sim >= cos_threshold)  # [N, 6]
                    dom_face = np.argmax(cos_sim, axis=1)  # [N]
                    any_active = spatial_face_active.any(axis=1)  # [N]

                    # --- mask_3d_voxel.jpg (2x3 grid, dense voxel_resolution³) ---
                    # Adaptive point size: smaller for higher resolution (32³ has 8x more points than 16³)
                    scatter_s = max(2, 15 * (16 / voxel_resolution) ** 2)
                    scatter_alpha = max(0.3, 0.6 * (16 / voxel_resolution))
                    fig = plt.figure(figsize=(18, 12))
                    for f_idx in range(6):
                        ax3d = fig.add_subplot(2, 3, f_idx + 1, projection='3d')
                        ax3d.set_box_aspect([1, 1, 1])
                        active_idx = np.where(face_active_16[f_idx].ravel())[0]
                        for s in [-1, 1]:
                            for t in [-1, 1]:
                                ax3d.plot([-1,1],[s,s],[t,t], 'gray', alpha=0.2, lw=0.5)
                                ax3d.plot([s,s],[-1,1],[t,t], 'gray', alpha=0.2, lw=0.5)
                                ax3d.plot([s,s],[t,t],[-1,1], 'gray', alpha=0.2, lw=0.5)
                        if len(active_idx) > 0:
                            apos = voxel_pos[active_idx]
                            ax3d.scatter(apos[:,0], apos[:,1], apos[:,2],
                                         c=face_colors[f_idx], s=scatter_s, alpha=scatter_alpha, marker='s')
                        ax3d.scatter([cam_np[0]], [cam_np[1]], [cam_np[2]],
                                    c='red', s=80, marker='o', zorder=10)
                        fd = face_dirs[f_idx]
                        ax3d.quiver(cam_np[0], cam_np[1], cam_np[2],
                                    fd[0]*0.4, fd[1]*0.4, fd[2]*0.4,
                                    color=face_colors[f_idx], arrow_length_ratio=0.3, linewidth=2)
                        pct = 100 * len(active_idx) / len(voxel_pos)
                        ax3d.set_title(f'{face_names[f_idx].upper()} ({len(active_idx)}, {pct:.0f}%)', fontsize=10)
                        ax3d.set_xlim([-1.2, 1.2]); ax3d.set_ylim([-1.2, 1.2]); ax3d.set_zlim([-1.2, 1.2])
                        ax3d.set_xticklabels([]); ax3d.set_yticklabels([]); ax3d.set_zticklabels([])
                    plt.suptitle(
                        f'Active Voxels per Face (Overall, {voxel_resolution}³) | cam=({cam_np[0]:.2f}, {cam_np[1]:.2f}, {cam_np[2]:.2f})',
                        fontsize=12
                    )
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'mask_3d_voxel.jpg'), dpi=300, bbox_inches='tight')
                    plt.close()

                    # dom_face and any_active already computed above from face-level cos_sim

                    def _draw_combined_scatter(ax3d, elev, azim, title_str):
                        """Draw 3D scatter with face-colored dense voxel grid points."""
                        ax3d.set_box_aspect([1, 1, 1])
                        ax3d.view_init(elev=elev, azim=azim)
                        for s in [-1, 1]:
                            for t in [-1, 1]:
                                ax3d.plot([-1,1],[s,s],[t,t], 'gray', alpha=0.15, lw=0.5)
                                ax3d.plot([s,s],[-1,1],[t,t], 'gray', alpha=0.15, lw=0.5)
                                ax3d.plot([s,s],[t,t],[-1,1], 'gray', alpha=0.15, lw=0.5)
                        for f_idx in range(6):
                            mask_f = (dom_face == f_idx) & any_active
                            idxs = np.where(mask_f)[0]
                            if len(idxs) > 0:
                                apos = voxel_pos[idxs]
                                ax3d.scatter(apos[:,0], apos[:,1], apos[:,2],
                                             c=face_colors[f_idx], s=scatter_s, alpha=scatter_alpha, marker='s',
                                             label=face_names[f_idx].upper())
                        ax3d.scatter([cam_np[0]], [cam_np[1]], [cam_np[2]],
                                    c='red', s=100, marker='*', zorder=10, label='cam')
                        for f_idx in range(6):
                            fd = face_dirs[f_idx]
                            ax3d.quiver(cam_np[0], cam_np[1], cam_np[2],
                                        fd[0]*0.3, fd[1]*0.3, fd[2]*0.3,
                                        color=face_colors[f_idx], arrow_length_ratio=0.25, linewidth=1.5)
                        ax3d.set_xlim([-1.2, 1.2]); ax3d.set_ylim([-1.2, 1.2]); ax3d.set_zlim([-1.2, 1.2])
                        ax3d.set_xticklabels([]); ax3d.set_yticklabels([]); ax3d.set_zticklabels([])
                        ax3d.set_title(title_str, fontsize=11)

                    # --- mask_voxel_topdown.jpg (dense grid) ---
                    fig = plt.figure(figsize=(8, 8))
                    ax3d = fig.add_subplot(111, projection='3d')
                    _draw_combined_scatter(ax3d, elev=90, azim=0,
                        title_str=f'Top-Down (Dense {voxel_resolution}³) | cam=({cam_np[0]:.2f}, {cam_np[1]:.2f}, {cam_np[2]:.2f})')
                    ax3d.legend(loc='upper left', fontsize=8, markerscale=1.5)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'mask_voxel_topdown.jpg'), dpi=300, bbox_inches='tight')
                    plt.close()

                    # --- mask_voxel_camera.jpg (dense grid) ---
                    cam_r = np.sqrt(cam_np[0]**2 + cam_np[1]**2 + cam_np[2]**2)
                    if cam_r < 1e-6:
                        cam_r = 1.0
                    cam_azim = np.degrees(np.arctan2(cam_np[0], cam_np[1]))  # atan2(x, y), front=+Y
                    cam_elev = np.degrees(np.arcsin(np.clip(cam_np[2] / cam_r, -1, 1)))
                    fig = plt.figure(figsize=(8, 8))
                    ax3d = fig.add_subplot(111, projection='3d')
                    _draw_combined_scatter(ax3d, elev=cam_elev, azim=cam_azim,
                        title_str=f'Camera (Dense {voxel_resolution}³) | cam=({cam_np[0]:.2f}, {cam_np[1]:.2f}, {cam_np[2]:.2f})')
                    ax3d.legend(loc='upper left', fontsize=8, markerscale=1.5)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'mask_voxel_camera.jpg'), dpi=300, bbox_inches='tight')
                    plt.close()

                    # --- mask_active_voxels.jpg (overall + per-part active voxels, top-down) ---
                    gt_parts_key = f'{meta_prefix}gt_part_coords'
                    gt_parts_list = metadata.get(gt_parts_key, {}).get('value', None)
                    if gt_parts_list is not None and i < len(gt_parts_list) and gt_parts_list[i] is not None:
                        part_coords_list = gt_parts_list[i]  # list of [M_k, 3] tensors
                        part_labels = ['overall']
                        if asset_names_list and i < len(asset_names_list):
                            for name in asset_names_list[i]:
                                short = name.split('/')[-1].replace('.npz', '').replace('.glb', '')
                                if len(short) > 20:
                                    short = short[:17] + '...'
                                part_labels.append(short)
                        while len(part_labels) < len(part_coords_list):
                            part_labels.append(f'part_{len(part_labels)}')

                        n_parts = len(part_coords_list)
                        n_cols = min(4, n_parts)
                        n_rows = (n_parts + n_cols - 1) // n_cols
                        fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
                        sparse_scatter_s = max(3, 30 * (16 / voxel_resolution))

                        for p_idx, p_coords in enumerate(part_coords_list):
                            ax3d = fig.add_subplot(n_rows, n_cols, p_idx + 1, projection='3d')
                            ax3d.set_box_aspect([1, 1, 1])
                            ax3d.view_init(elev=90, azim=0)
                            if p_coords is None or len(p_coords) == 0:
                                ax3d.set_title(f'{part_labels[p_idx]} (0)', fontsize=9)
                                ax3d.set_xlim([-1.2, 1.2]); ax3d.set_ylim([-1.2, 1.2]); ax3d.set_zlim([-1.2, 1.2])
                                continue
                            p_pos = (2.0 * p_coords.numpy().astype(float) / (voxel_resolution - 1)) - 1.0
                            p_dirs = p_pos - cam_np[None, :]
                            p_norms = np.linalg.norm(p_dirs, axis=1, keepdims=True)
                            p_dirs = p_dirs / np.maximum(p_norms, 1e-8)
                            p_cos = p_dirs @ face_dirs.T  # [M, 6]
                            p_dom = np.argmax(p_cos, axis=1)
                            # Draw bounding box
                            for s in [-1, 1]:
                                for t in [-1, 1]:
                                    ax3d.plot([-1,1],[s,s],[t,t], 'gray', alpha=0.1, lw=0.3)
                                    ax3d.plot([s,s],[-1,1],[t,t], 'gray', alpha=0.1, lw=0.3)
                                    ax3d.plot([s,s],[t,t],[-1,1], 'gray', alpha=0.1, lw=0.3)
                            for f_idx in range(6):
                                idxs = np.where(p_dom == f_idx)[0]
                                if len(idxs) > 0:
                                    apos = p_pos[idxs]
                                    ax3d.scatter(apos[:,0], apos[:,1], apos[:,2],
                                                 c=face_colors[f_idx], s=sparse_scatter_s, alpha=0.6, marker='s')
                            ax3d.scatter([cam_np[0]], [cam_np[1]], [cam_np[2]],
                                        c='red', s=60, marker='*', zorder=10)
                            ax3d.set_title(f'{part_labels[p_idx]} ({len(p_pos)})', fontsize=9)
                            ax3d.set_xlim([-1.2, 1.2]); ax3d.set_ylim([-1.2, 1.2]); ax3d.set_zlim([-1.2, 1.2])
                            ax3d.set_xticklabels([]); ax3d.set_yticklabels([]); ax3d.set_zticklabels([])
                        # Add face color legend
                        from matplotlib.patches import Patch
                        from matplotlib.lines import Line2D
                        legend_patches = [
                            Patch(facecolor=face_colors[fi], edgecolor='gray', label=face_names[fi].capitalize())
                            for fi in range(6)
                        ]
                        legend_patches.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                                                     markersize=12, label='Camera Center'))
                        fig.legend(handles=legend_patches, loc='lower center',
                                   ncol=7, fontsize=9, frameon=True, fancybox=True,
                                   bbox_to_anchor=(0.5, -0.01))
                        plt.suptitle(
                            f'Active Voxels (Overall + Parts) | cam=({cam_np[0]:.2f}, {cam_np[1]:.2f}, {cam_np[2]:.2f})',
                            fontsize=12
                        )
                        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                        plt.savefig(os.path.join(save_dir, 'mask_active_voxels.jpg'), dpi=300, bbox_inches='tight')
                        plt.close()

                    # --- Asset-aware heatmap ---
                    if obbs_list is not None and asset_names_list is not None:
                        sample_has_layout = (
                            has_layout_list[i] if has_layout_list is not None and i < len(has_layout_list) else False
                        )
                        self._visualize_asset_aware_heatmap(
                            cam=cam,
                            cam_np=cam_np,
                            obbs=obbs_list[i] if i < len(obbs_list) else None,
                            asset_names=asset_names_list[i] if i < len(asset_names_list) else None,
                            tokens_per_face=tokens_per_face,
                            fov_degrees=fov_degrees,
                            face_names=face_names,
                            save_dir=save_dir,
                            has_layout=sample_has_layout,
                        )

    @torch.no_grad()
    def _visualize_asset_aware_heatmap(
        self,
        cam: torch.Tensor,
        cam_np: np.ndarray,
        obbs,
        asset_names: List[str],
        tokens_per_face: int,
        fov_degrees: float,
        face_names: List[str],
        save_dir: str,
        has_layout: bool = False,
    ):
        """
        Visualize asset-aware cross-attention mask as a heatmap.

        Generates cross_mask_heatmap_asset_aware.jpg showing:
        - Row 0 (overall): spatial attention mask (Stage 1-style, per-voxel density)
        - Row 1 (layout, if has_layout): same spatial mask as overall
        - Row N..M (per asset): only bbox-projected tokens active

        X-axis: 6 * tokens_per_face (image tokens)
        Y-axis: [overall | (layout) | asset_0 | asset_1 | ...] (one row-band per part)
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from ..utils.asset_attention_mask import create_per_part_cross_attn_masks

        if obbs is None or asset_names is None:
            return
        if not isinstance(obbs, torch.Tensor):
            obbs = torch.tensor(obbs, dtype=torch.float32)
        obbs = obbs.cpu()
        if obbs.shape[0] == 0:
            return

        num_assets = obbs.shape[0]
        asset_start = 2 if has_layout else 1
        num_parts = asset_start + num_assets  # overall + (layout) + assets
        expand_pixels = getattr(self, 'expand_pixels', 28)
        voxel_resolution = getattr(self, 'voxel_resolution', 32)

        # Per-part masks from bbox projection (includes layout slot if has_layout)
        asset_masks = create_per_part_cross_attn_masks(
            obbs=obbs,
            camera_center=cam.cpu().float(),
            num_parts=num_parts,
            tokens_per_face=tokens_per_face,
            fov_degrees=fov_degrees,
            expand_pixels=expand_pixels,
            has_layout=has_layout,
        )
        # asset_masks: [overall, (layout), asset0, asset1, ...] boolean tensors

        # Overall/layout spatial mask: compute face-level density efficiently
        import torch.nn.functional as _F
        cam_t = cam.cpu().float()
        if cam_t.dim() == 1:
            cam_t = cam_t.unsqueeze(0)
        coords_t = torch.linspace(-1 + 1/voxel_resolution, 1 - 1/voxel_resolution, voxel_resolution)
        xx_t, yy_t, zz_t = torch.meshgrid(coords_t, coords_t, coords_t, indexing='ij')
        vp_t = torch.stack([xx_t, yy_t, zz_t], dim=-1).reshape(-1, 3)  # [N, 3]
        dirs_t = _F.normalize(vp_t.unsqueeze(0) - cam_t.unsqueeze(1), p=2, dim=-1)
        face_dirs_t = torch.tensor([
            [0,1,0],[1,0,0],[0,-1,0],[-1,0,0],[0,0,1],[0,0,-1]
        ], dtype=torch.float32)
        cos_sim_h = torch.einsum('bnd,fd->bnf', dirs_t, face_dirs_t).squeeze(0).numpy()  # [N, 6]
        fov_rad_h = np.radians(fov_degrees)
        cos_threshold_h = np.cos(fov_rad_h / 2)

        # Per-voxel face visibility: [N_voxels, 6] binary mask
        face_visible_all = (cos_sim_h >= cos_threshold_h).astype(float)  # [N, 6]

        total_tokens = 6 * tokens_per_face

        # Build spatial rows (used for both overall and layout)
        dom_face = np.argmax(cos_sim_h, axis=1)  # [N]
        sort_idx = np.argsort(dom_face)  # group by dominant face
        n_voxels = cos_sim_h.shape[0]

        rows_per_part = max(8, 200 // num_parts)
        total_rows = num_parts * rows_per_part
        heatmap = np.zeros((total_rows, total_tokens))

        subsample_indices = sort_idx[np.linspace(0, n_voxels - 1, rows_per_part, dtype=int)]
        face_visible_sub = face_visible_all[subsample_indices]  # [rows_per_part, 6]
        spatial_rows = np.repeat(face_visible_sub, tokens_per_face, axis=1)  # [rows_per_part, total_tokens]

        # Overall rows (part 0): per-voxel spatial stripe pattern
        for r in range(rows_per_part):
            heatmap[r, :] = spatial_rows[r]

        # Layout rows (part 1, if has_layout): same spatial pattern as overall
        if has_layout:
            row_start = 1 * rows_per_part
            for r in range(rows_per_part):
                heatmap[row_start + r, :] = spatial_rows[r]

        # Asset rows: binary bbox projection masks
        for asset_idx in range(num_assets):
            mask_idx = asset_idx + asset_start  # index into asset_masks
            if mask_idx >= len(asset_masks):
                break
            mask_np = asset_masks[mask_idx].cpu().numpy().astype(float)
            row_start = (asset_idx + asset_start) * rows_per_part
            row_end = row_start + rows_per_part
            for r in range(row_start, row_end):
                heatmap[r, :] = mask_np

        # Row labels
        row_labels = [f'overall (spatial {voxel_resolution}\u00b3)']
        if has_layout:
            row_labels.append(f'layout (spatial {voxel_resolution}\u00b3)')
        for name in asset_names:
            short = name.split('/')[-1].replace('.npz', '').replace('.glb', '')
            if len(short) > 20:
                short = short[:17] + '...'
            row_labels.append(short)

        # --- Draw heatmap ---
        fig, ax = plt.subplots(1, 1, figsize=(16, max(4, 0.6 * num_parts + 2)))
        ax.imshow(heatmap, aspect='auto', cmap='Blues', vmin=0, vmax=1, interpolation='nearest')

        # Face boundary lines + face labels at the TOP (just below title)
        for f_idx in range(1, 6):
            ax.axvline(x=f_idx * tokens_per_face, color='white', linewidth=1, linestyle='--', alpha=0.8)
        for f_idx in range(6):
            cx = f_idx * tokens_per_face + tokens_per_face // 2
            ax.text(cx, -rows_per_part * 0.15, face_names[f_idx].upper(), ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#2c3e50')

        # Part boundary lines and labels
        for part_idx in range(1, num_parts):
            y = part_idx * rows_per_part
            ax.axhline(y=y - 0.5, color='#333333', linewidth=0.8, linestyle='-', alpha=0.5)
        for part_idx in range(num_parts):
            cy = part_idx * rows_per_part + rows_per_part // 2
            if part_idx < asset_start:
                # Overall or layout: show per-face coverage stats
                face_coverage = face_visible_all.mean(axis=0)  # [6] avg fraction per face
                pct = 100 * face_coverage.mean()
                label = f'{row_labels[part_idx]} ({n_voxels} voxels, avg {pct:.0f}%)'
            else:
                mask_idx = part_idx
                n_active = int(asset_masks[mask_idx].sum().item()) if mask_idx < len(asset_masks) else 0
                pct = 100 * n_active / total_tokens
                label = f'{row_labels[part_idx]} ({n_active}, {pct:.0f}%)'
            ax.text(-50, cy, label, ha='right', va='center', fontsize=7, color='#333333')

        layout_str = ' + 1 layout' if has_layout else ''
        ax.set_title(
            f'Asset-Aware Cross-Attn Mask | cam=({cam_np[0]:.2f}, {cam_np[1]:.2f}, {cam_np[2]:.2f})\n'
            f'{num_parts} parts (1 overall{layout_str} + {num_assets} assets)',
            fontsize=11, pad=15
        )
        ax.set_xlabel(f'Tokens (6 x {tokens_per_face} = {total_tokens})')
        ax.set_ylabel('Parts')
        ax.set_yticks([])
        ax.set_xlim(-0.5, total_tokens - 0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cross_mask_heatmap_asset_aware.jpg'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    @torch.no_grad()
    def _render_face_masked_voxels(self, gt_latent, face_active_16, cam_np,
                                    face_names, face_colors_rgb, save_dir):
        """
        Render actual GT voxels colored by which cubemap face they attend to.

        Produces two images:
          - mask_voxel_topdown.jpg: 2x3 grid from top-down view
          - mask_voxel_camera.jpg: 2x3 grid from camera-center looking at origin

        Each subplot shows:
          - All occupied voxels in dark gray (context)
          - Face-active occupied voxels in the face's color
        """
        from ..representations import Voxel
        from ..renderers import VoxelRenderer
        from ..utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
        from PIL import Image as PILImage, ImageDraw, ImageFont

        # Decode GT latent to 64³ occupancy
        z = gt_latent.unsqueeze(0).cuda()  # [1, 8, 16, 16, 16]
        decoded = self.dataset.decode_latent(z)  # [1, 1, 64, 64, 64]
        occupancy_64 = (decoded[0, 0] > 0).cpu().numpy()  # [64, 64, 64] bool
        resolution = 64

        # Upscale face masks from 16³ to 64³
        scale = resolution // 16  # 4
        face_active_64 = []
        for fa16 in face_active_16:
            fa64 = np.repeat(np.repeat(np.repeat(
                fa16.astype(bool), scale, axis=0), scale, axis=1), scale, axis=2)
            face_active_64.append(fa64)

        # Get all occupied voxel coords
        all_occ_coords = torch.nonzero(torch.tensor(occupancy_64), as_tuple=False)  # [N, 3]
        if all_occ_coords.shape[0] == 0:
            return
        all_occ_color = torch.full((all_occ_coords.shape[0], 3), 0.25)  # dark gray

        # Setup renderer
        renderer = VoxelRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.ssaa = 4
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)

        # Camera setups: topdown and from-camera
        # 1) Top-down
        td_yaw = [0]
        td_pitch = [90 / 180 * np.pi]
        td_exts, td_ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(td_yaw, td_pitch, 2, 30)

        # 2) From camera center looking at origin
        cx, cy, cz = cam_np[0], cam_np[1], cam_np[2]
        r_cam = np.sqrt(cx**2 + cy**2 + cz**2)
        if r_cam < 1e-6:
            r_cam = 1.0
        # Camera is inside the scene (close to origin), so we position the render camera
        # farther out along the same direction for a good view
        cam_yaw = np.arctan2(cx, cy)  # atan2(x, y) since front is +Y
        cam_pitch = np.arcsin(np.clip(cz / r_cam, -1, 1))
        cam_exts, cam_ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(
            [cam_yaw], [cam_pitch], 2, 30
        )

        tile_size = 256
        viewpoints = [
            ('mask_voxel_topdown.jpg', td_exts[0], td_ints[0], 'Top-Down View'),
            ('mask_voxel_camera.jpg', cam_exts[0], cam_ints[0],
             f'Camera View (yaw={np.degrees(cam_yaw):.0f}°, pitch={np.degrees(cam_pitch):.0f}°)'),
        ]

        for filename, ext, intr, view_title in viewpoints:
            # Build 2x3 grid: one subplot per face
            grid = torch.zeros(3, tile_size * 2, tile_size * 3)

            for f_idx in range(6):
                row, col = f_idx // 3, f_idx % 3

                # Face-active occupied voxels
                face_occ = occupancy_64 & face_active_64[f_idx]
                face_coords = torch.nonzero(torch.tensor(face_occ), as_tuple=False)

                # Build color array: gray for all, face color for active
                if face_coords.shape[0] > 0:
                    # Combine: all occupied (gray) + face-active (colored, drawn on top)
                    combined_coords = torch.cat([all_occ_coords, face_coords], dim=0)
                    fc = face_colors_rgb[f_idx]
                    face_color = torch.tensor([[fc[0], fc[1], fc[2]]]).expand(face_coords.shape[0], 3)
                    combined_color = torch.cat([all_occ_color, face_color.float()], dim=0)
                else:
                    combined_coords = all_occ_coords
                    combined_color = all_occ_color

                combined_coords = combined_coords.cuda()
                combined_color = combined_color.cuda()
                rep = Voxel(
                    origin=[-0.5, -0.5, -0.5],
                    voxel_size=1 / resolution,
                    coords=combined_coords,
                    attrs=combined_color,
                    layout={'color': slice(0, 3)},
                )
                res = renderer.render(rep, ext, intr, colors_overwrite=combined_color)
                tile = res['color'].cpu()  # [3, 512, 512]

                # Resize to tile_size
                tile = F.interpolate(tile.unsqueeze(0), size=tile_size, mode='bilinear', align_corners=False)[0]

                # Add face label
                tile_np = (tile.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                pil_tile = PILImage.fromarray(tile_np)
                draw = ImageDraw.Draw(pil_tile)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
                except Exception:
                    font = ImageFont.load_default()

                n_face_occ = face_coords.shape[0] if face_coords.shape[0] > 0 else 0
                label = f'{face_names[f_idx].upper()} ({n_face_occ})'
                draw.text((4, 4), label, fill=(255, 255, 255), font=font)
                tile = torch.tensor(np.array(pil_tile)).permute(2, 0, 1).float() / 255.0

                grid[:, row * tile_size:(row + 1) * tile_size, col * tile_size:(col + 1) * tile_size] = tile

            # Save grid
            utils.save_image(grid, os.path.join(save_dir, filename), normalize=False)
            print(f'  Saved {filename}')

    def _make_row_label_column(self, labels: list, row_heights: list, col_width: int = 40) -> torch.Tensor:
        """Create a narrow white column with vertically-centered text labels for each row.

        Args:
            labels: list of text labels, one per row
            row_heights: list of pixel heights for each row
            col_width: width of the label column in pixels

        Returns:
            Tensor [3, total_h, col_width]
        """
        from PIL import Image as PILImage, ImageDraw, ImageFont
        total_h = sum(row_heights)
        col = PILImage.new('RGB', (col_width, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(col)
        font_size = min(col_width - 4, 16)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

        y_offset = 0
        for label, rh in zip(labels, row_heights):
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = (col_width - tw) // 2
            y = y_offset + (rh - th) // 2
            draw.text((x, y), label, fill=(0, 0, 0), font=font)
            y_offset += rh

        col_tensor = torch.tensor(np.array(col)).permute(2, 0, 1).float() / 255.0
        return col_tensor

    @torch.no_grad()
    def _create_comparison_images(self, samples):
        """
        Create GT vs Prediction comparison images by concatenating them.

        Non-interior: GT on top, pred on bottom (vertical concat per sample).
        Interior: cubemap row (real) + GT rendered row + pred rendered row.
        A narrow white column with row labels is prepended on the left.
        """
        comparison_pairs = [
            ('train_sample_gt', 'train_sample', 'train_comparison_exterior', False),
            ('train_gt_topdown', 'train_pred_topdown', 'train_comparison_topdown', False),
            ('train_gt_topdown_cam', 'train_pred_topdown_cam', 'train_comparison_topdown_cam', False),
            ('train_gt_interior', 'train_pred_interior', 'train_comparison_interior', True),
            # eval variants
            ('eval_sample_gt', 'eval_sample', 'eval_comparison_exterior', False),
            ('eval_gt_topdown', 'eval_pred_topdown', 'eval_comparison_topdown', False),
            ('eval_gt_topdown_cam', 'eval_pred_topdown_cam', 'eval_comparison_topdown_cam', False),
            ('eval_gt_interior', 'eval_pred_interior', 'eval_comparison_interior', True),
            # shaded/base_color exterior variants (texture mode — visualize_sample returns dict)
            ('train_sample_gt_shaded', 'train_sample_shaded', 'train_comparison_exterior_shaded', False),
            ('train_sample_gt_base_color', 'train_sample_base_color', 'train_comparison_exterior_base_color', False),
            ('eval_sample_gt_shaded', 'eval_sample_shaded', 'eval_comparison_exterior_shaded', False),
            ('eval_sample_gt_base_color', 'eval_sample_base_color', 'eval_comparison_exterior_base_color', False),
            # base_color variants for topdown/interior (texture mode — keys exist only when vis returns dict)
            ('train_gt_topdown_shaded', 'train_pred_topdown_shaded', 'train_comparison_topdown_shaded', False),
            ('train_gt_topdown_base_color', 'train_pred_topdown_base_color', 'train_comparison_topdown_base_color', False),
            ('train_gt_topdown_cam_shaded', 'train_pred_topdown_cam_shaded', 'train_comparison_topdown_cam_shaded', False),
            ('train_gt_topdown_cam_base_color', 'train_pred_topdown_cam_base_color', 'train_comparison_topdown_cam_base_color', False),
            ('train_gt_interior_shaded', 'train_pred_interior_shaded', 'train_comparison_interior_shaded', True),
            ('train_gt_interior_base_color', 'train_pred_interior_base_color', 'train_comparison_interior_base_color', True),
            ('eval_gt_topdown_shaded', 'eval_pred_topdown_shaded', 'eval_comparison_topdown_shaded', False),
            ('eval_gt_topdown_base_color', 'eval_pred_topdown_base_color', 'eval_comparison_topdown_base_color', False),
            ('eval_gt_topdown_cam_shaded', 'eval_pred_topdown_cam_shaded', 'eval_comparison_topdown_cam_shaded', False),
            ('eval_gt_topdown_cam_base_color', 'eval_pred_topdown_cam_base_color', 'eval_comparison_topdown_cam_base_color', False),
            ('eval_gt_interior_shaded', 'eval_pred_interior_shaded', 'eval_comparison_interior_shaded', True),
            ('eval_gt_interior_base_color', 'eval_pred_interior_base_color', 'eval_comparison_interior_base_color', True),
        ]

        label_col_width = 40

        for gt_key, pred_key, out_key, is_interior in comparison_pairs:
            if gt_key not in samples or pred_key not in samples:
                continue
            if samples[gt_key]['type'] != 'image' or samples[pred_key]['type'] != 'image':
                continue

            gt_imgs = samples[gt_key]['value']
            pred_imgs = samples[pred_key]['value']
            B = min(gt_imgs.shape[0], pred_imgs.shape[0])
            gt_imgs = gt_imgs[:B]
            pred_imgs = pred_imgs[:B]

            if is_interior:
                # Interior format: [B, 3, label_h + 2*tile_size, 6*tile_size]
                # Extract: label strip, cubemap row (from GT), GT rendered, pred rendered
                total_w = gt_imgs.shape[3]
                tile_size = total_w // 6
                label_h = gt_imgs.shape[2] - 2 * tile_size

                # Row labels for interior: face labels row, Input cubemap, GT rendered, Pred rendered
                row_labels = ['', 'Input', 'GT', 'Pred']
                row_heights = [label_h, tile_size, tile_size, tile_size]
                label_col = self._make_row_label_column(row_labels, row_heights, label_col_width).to(gt_imgs.device)

                comparisons = []
                for i in range(B):
                    label_strip = gt_imgs[i, :, :label_h, :]
                    cubemap_row = gt_imgs[i, :, label_h:label_h + tile_size, :]
                    gt_rendered = gt_imgs[i, :, label_h + tile_size:, :]
                    pred_rendered = pred_imgs[i, :, label_h + tile_size:, :]
                    content = torch.cat([label_strip, cubemap_row, gt_rendered, pred_rendered], dim=1)
                    composite = torch.cat([label_col, content], dim=2)
                    comparisons.append(composite)
                samples[out_key] = {'value': torch.stack(comparisons), 'type': 'image'}
            else:
                # Vertical concat: GT on top, pred on bottom
                gt_h = gt_imgs.shape[2]
                pred_h = pred_imgs.shape[2]
                row_labels = ['GT', 'Pred']
                row_heights = [gt_h, pred_h]
                label_col = self._make_row_label_column(row_labels, row_heights, label_col_width).to(gt_imgs.device)

                content = torch.cat([gt_imgs, pred_imgs], dim=2)
                # Prepend label column to each sample
                label_col_batch = label_col.unsqueeze(0).expand(B, -1, -1, -1)
                comparison = torch.cat([label_col_batch, content], dim=3)
                samples[out_key] = {'value': comparison, 'type': 'image'}

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=16, batch_size=4, verbose=False): # num_samples=36
        """
        Sample images from the model.
        NOTE: This function should be called by all processes.
        """
        if self.is_master:
            print(f'\nSampling {num_samples} images...', end='')

        if suffix is None:
            suffix = f'step{self.step:07d}'

        # Assign tasks
        num_samples_per_process = int(np.ceil(num_samples / self.world_size))
        amp_context = partial(torch.autocast, device_type='cuda', dtype=self.mix_precision_dtype) if self.mix_precision_mode == 'amp' else nullcontext
        with amp_context():
            samples = self.run_snapshot(num_samples_per_process, batch_size=batch_size, verbose=verbose) # trellis2/trainers/flow_matching/flow_matching.py -> run_snapshot

        # Extract metadata entries (camera_center, cond, paths) before image conversion
        metadata = {}
        metadata_keys = [k for k in samples.keys() if k.startswith('_')]
        for k in metadata_keys:
            metadata[k] = samples.pop(k)

        # Generate topdown/interior visualizations BEFORE converting samples to images
        # (visualization functions need raw sample tensors for decoding)
        self._generate_additional_vis(samples, metadata, self.dataset, prefix_filter='train')
        if self.eval_dataset is not None:
            self._generate_additional_vis(samples, metadata, self.eval_dataset, prefix_filter='eval')

        # Generate per-sample visualizations on ALL processes BEFORE preprocessing/gather.
        # Each process has its own portion of samples (different sample_paths), so file names
        # won't conflict. Running on all processes means all samples get visualized, not just
        # the master's portion (which is only num_samples/world_size).
        self._generate_per_sample_vis(samples, metadata, self.dataset, suffix, prefix_filter='train')
        if self.eval_dataset is not None:
            self._generate_per_sample_vis(samples, metadata, self.eval_dataset, suffix, prefix_filter='eval')

        # Preprocess images (convert remaining samples to rendered images)
        for key in tqdm(list(samples.keys()), desc='Preprocess images', total=len(samples)):
            if samples[key]['type'] == 'sample':
                vis = self.visualize_sample(samples[key]['value'])
                if isinstance(vis, dict):
                    for k, v in vis.items():
                        samples[f'{key}_{k}'] = {'value': v, 'type': 'image'}
                    del samples[key]
                else:
                    samples[key] = {'value': vis, 'type': 'image'}

        # Gather results across processes
        if self.world_size > 1:
            for key in list(samples.keys()):
                if samples[key]['type'] not in ('image', 'number'):
                    continue
                samples[key]['value'] = samples[key]['value'].contiguous()
                if self.is_master:
                    all_images = [torch.empty_like(samples[key]['value']) for _ in range(self.world_size)]
                else:
                    all_images = []
                dist.gather(samples[key]['value'], all_images, dst=0)
                if self.is_master:
                    samples[key]['value'] = torch.cat(all_images, dim=0)[:num_samples]

        # Create GT vs Prediction comparison images (on master after gather)
        if self.is_master:
            self._create_comparison_images(samples)

        # Save images and metadata
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'samples', suffix), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'samples', suffix, 'concat'), exist_ok=True)
            for key in samples.keys():
                if samples[key]['type'] == 'image':
                    # Interior images are wide composites: use nrow=1
                    if 'interior' in key:
                        nrow = 1
                    else:
                        nrow = int(np.sqrt(num_samples))
                    # Comparison images go into concat/ subfolder
                    if 'comparison_' in key:
                        save_path = os.path.join(self.output_dir, 'samples', suffix, 'concat', f'{key}_{suffix}.jpg')
                    else:
                        save_path = os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg')
                    utils.save_image(
                        samples[key]['value'],
                        save_path,
                        nrow=nrow,
                        normalize=True,
                        value_range=self.dataset.value_range,
                    )
                elif samples[key]['type'] == 'number':
                    min = samples[key]['value'].min()
                    max = samples[key]['value'].max()
                    images = (samples[key]['value'] - min) / (max - min)
                    images = utils.make_grid(
                        images,
                        nrow=int(np.sqrt(num_samples)),
                        normalize=False,
                    )
                    save_image_with_notes(
                        images,
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                        notes=f'{key} min: {min}, max: {max}',
                    )

            # Save data paths as JSON
            paths_json = {}
            for k, v in metadata.items():
                if isinstance(v, dict) and v.get('type') == 'paths':
                    json_key = k.strip('_')
                    paths_json[json_key] = v['value']
            if paths_json:
                with open(os.path.join(self.output_dir, 'samples', suffix, f'data_paths_{suffix}.json'), 'w') as f:
                    json.dump(paths_json, f, indent=2)

            # Visualize cross-attention mask (Stage 1: spatial attention heatmap + 3D voxels)
            try:
                self._visualize_cross_attn_mask(metadata, suffix)
            except Exception as e:
                print(f'\nWarning: Failed to visualize cross-attn mask: {e}')

        if self.is_master:
            print(' Done.')

    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        assert self.is_master, 'update_ema() should be called only by the rank 0 process.'
        for i, ema_rate in enumerate(self.ema_rate):
            for master_param, ema_param in zip(self.master_params, self.ema_params[i]):
                ema_param.detach().mul_(ema_rate).add_(master_param, alpha=1.0 - ema_rate)

    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        if self.is_master:
            print('\nPerforming DDP check...')

        if self.is_master:
            print('Checking if parameters are consistent across processes...')
        dist.barrier()
        try:
            for p in self.master_params:
                # split to avoid OOM
                for i in range(0, p.numel(), 10000000):
                    sub_size = min(10000000, p.numel() - i)
                    sub_p = p.detach().view(-1)[i:i+sub_size]
                    # gather from all processes
                    sub_p_gather = [torch.empty_like(sub_p) for _ in range(self.world_size)]
                    dist.all_gather(sub_p_gather, sub_p)
                    # check if equal
                    assert all([torch.equal(sub_p, sub_p_gather[i]) for i in range(self.world_size)]), 'parameters are not consistent across processes'
        except AssertionError as e:
            if self.is_master:
                print(f'\n\033[91mError: {e}\033[0m')
                print('DDP check failed.')
            raise e

        dist.barrier()
        if self.is_master:
            print('Done.')

    @abstractmethod
    def training_losses(**mb_data):
        """
        Compute training losses.
        """
        pass

    def load_data(self):
        """
        Load data.
        """
        if self.prefetch_data:
            if self._data_prefetched is None:
                self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
            data = self._data_prefetched
            self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        else:
            data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        
        # if the data is a dict, we need to split it into multiple dicts with batch_size_per_gpu
        if isinstance(data, dict):
            if self.batch_split == 1:
                data_list = [data]
            else:
                batch_size = list(data.values())[0].shape[0]
                data_list = [
                    {k: v[i * batch_size // self.batch_split:(i + 1) * batch_size // self.batch_split] for k, v in data.items()}
                    for i in range(self.batch_split)
                ]
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError('Data must be a dict or a list of dicts.')
        
        return data_list

    def run_step(self, data_list):
        """
        Run a training step.
        """
        step_log = {'loss': {}, 'status': {}}
        amp_context = partial(torch.autocast, device_type='cuda', dtype=self.mix_precision_dtype) if self.mix_precision_mode == 'amp' else nullcontext
        elastic_controller_context = self.elastic_controller.record if self.elastic_controller_config is not None else nullcontext

        # Train
        losses = []
        statuses = []
        elastic_controller_logs = []
        zero_grad(self.model_params)
        for i, mb_data in enumerate(data_list):
            ## sync at the end of each batch split
            sync_contexts = [self.training_models[name].no_sync for name in self.training_models] if i != len(data_list) - 1 and self.world_size > 1 else [nullcontext]
            with nested_contexts(*sync_contexts), elastic_controller_context():
                with amp_context():
                    loss, status = self.training_losses(**mb_data)
                    l = loss['loss'] / len(data_list)
                ## backward
                if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
                    self.scaler.scale(l).backward()
                elif self.mix_precision_mode == 'inflat_all' and self.mix_precision_dtype == torch.float16:
                    scaled_l = l * (2 ** self.log_scale)
                    scaled_l.backward()
                else:
                    l.backward()
            ## log
            losses.append(dict_foreach(loss, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            statuses.append(dict_foreach(status, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            if self.elastic_controller_config is not None:
                elastic_controller_logs.append(self.elastic_controller.log())
        ## gradient clip
        if self.grad_clip is not None:
            if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
                self.scaler.unscale_(self.optimizer)
            elif self.mix_precision_mode == 'inflat_all':
                model_grads_to_master_grads(self.model_params, self.master_params)
                if self.mix_precision_dtype == torch.float16:
                    self.master_params[0].grad.mul_(1.0 / (2 ** self.log_scale))
            if isinstance(self.grad_clip, float):
                grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params, self.grad_clip)
            else:
                grad_norm = self.grad_clip(self.master_params)
            if torch.isfinite(grad_norm):
                statuses[-1]['grad_norm'] = grad_norm.item()
        ## step
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            prev_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif self.mix_precision_mode == 'inflat_all':
            if self.mix_precision_dtype == torch.float16:
                prev_scale = 2 ** self.log_scale
                if not any(not p.grad.isfinite().all() for p in self.model_params):
                    if self.grad_clip is None:
                        model_grads_to_master_grads(self.model_params, self.master_params)
                        self.master_params[0].grad.mul_(1.0 / (2 ** self.log_scale))
                    self.optimizer.step()
                    master_params_to_model_params(self.model_params, self.master_params)
                    self.log_scale += self.fp16_scale_growth
                else:
                    self.log_scale -= 1
            else:
                prev_scale = 1.0
                if self.grad_clip is None:
                    model_grads_to_master_grads(self.model_params, self.master_params)
                if not any(not p.grad.isfinite().all() for p in self.master_params):
                    self.optimizer.step()
                    master_params_to_model_params(self.model_params, self.master_params)
                else:
                    print('\n\033[93mWarning: NaN detected in gradients. Skipping update.\033[0m')
        else:
            prev_scale = 1.0
            if not any(not p.grad.isfinite().all() for p in self.model_params):
                self.optimizer.step()
            else:
                print('\n\033[93mWarning: NaN detected in gradients. Skipping update.\033[0m') 
        ## adjust learning rate
        if self.lr_scheduler_config is not None:
            statuses[-1]['lr'] = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step()

        # Logs
        step_log['loss'] = dict_reduce(losses, lambda x: np.mean(x))
        step_log['status'] = dict_reduce(statuses, lambda x: np.mean(x), special_func={'min': lambda x: np.min(x), 'max': lambda x: np.max(x)})
        if self.elastic_controller_config is not None:
            step_log['elastic'] = dict_reduce(elastic_controller_logs, lambda x: np.mean(x))
        if self.grad_clip is not None:
            step_log['grad_clip'] = self.grad_clip if isinstance(self.grad_clip, float) else self.grad_clip.log()
            
        # Check grad and norm of each param
        if self.log_param_stats:
            param_norms = {}
            param_grads = {}
            for model_name, model in self.models.items():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param_norms[f'{model_name}.{name}'] = param.norm().item()
                        if param.grad is not None and torch.isfinite(param.grad).all():
                            param_grads[f'{model_name}.{name}'] = param.grad.norm().item() / prev_scale
            step_log['param_norms'] = param_norms
            step_log['param_grads'] = param_grads

        # Update exponential moving average
        if self.is_master:
            self.update_ema()

        return step_log

    def save_logs(self):
        log_str = '\n'.join([
            f'{step}: {json.dumps(dict_foreach(log, lambda x: float(x)))}' for step, log in self.log
        ])
        with open(os.path.join(self.output_dir, 'log.txt'), 'a') as log_file:
            log_file.write(log_str + '\n')

        # show with mlflow
        log_show = [l for _, l in self.log if not dict_any(l, lambda x: np.isnan(x))]
        log_show = dict_reduce(log_show, lambda x: np.mean(x))
        log_show = dict_flatten(log_show, sep='/')
        for key, value in log_show.items():
            self.writer.add_scalar(key, value, self.step)
        self.log = []
        
    def check_abort(self):
        """
        Check if training should be aborted due to certain conditions.
        """
        # 1. If log_scale in inflat_all mode is less than 0
        if self.mix_precision_dtype == torch.float16 and \
           self.mix_precision_mode == 'inflat_all' and \
           self.log_scale < 0:
            if self.is_master:
                print ('\n\n\033[91m')
                print (f'ABORT: log_scale in inflat_all mode is less than 0 at step {self.step}.')
                print ('This indicates that the model is diverging. You should look into the model and the data.')
                print ('\033[0m')
                self.save(non_blocking=False)
                self.save_logs()
            if self.world_size > 1:
                dist.barrier()
            raise ValueError('ABORT: log_scale in inflat_all mode is less than 0.')

    def run(self):
        """
        Run training.
        """
        #NOTE:
        self.snapshot_batch_size = 4
        # if self.is_master:
        #     print('\nStarting training...')
        #     self.snapshot_dataset(batch_size=self.snapshot_batch_size)
        # # Sync all ranks after snapshot_dataset (runs only on master, can take long)
        # if self.world_size > 1:
        #     dist.barrier()
        # if self.step == 0:
        #     self.snapshot(suffix='init', batch_size=self.snapshot_batch_size)
        #     print('Skipping snapshot init')
        # else: # resume
        #     self.snapshot(suffix=f'resume_step{self.step:07d}', batch_size=self.snapshot_batch_size)

        time_last_print = 0.0
        time_elapsed = 0.0

        # Calculate epoch info
        # 1 step = batch_size samples (batch_size_per_gpu × world_size)
        # 1 epoch = len(dataset) / batch_size steps
        dataset_size = len(self.dataset)
        samples_per_step = self.batch_size  # = batch_size_per_gpu × world_size
        steps_per_epoch = dataset_size / samples_per_step

        if self.is_master:
            print(f'\nDataset size: {dataset_size}')
            print(f'Samples per step: {samples_per_step} (batch_size_per_gpu={self.batch_size_per_gpu} × world_size={self.world_size})')
            print(f'Steps per epoch: {steps_per_epoch:.1f}')
            print(f'Total epochs: {self.max_steps / steps_per_epoch:.2f}')

        # Initialize tqdm progress bar
        pbar = None
        if self.is_master:
            pbar = tqdm(
                initial=self.step,
                total=self.max_steps,
                desc='Training',
                unit='step',
                dynamic_ncols=True,
                disable=False
            )
        
        while self.step < self.max_steps: 
            time_start = time.time()

            data_list = self.load_data()
            step_log = self.run_step(data_list)

            time_end = time.time()
            time_elapsed += time_end - time_start

            self.step += 1

            # Print progress (original code - commented out)
            if self.is_master and self.step % self.i_print == 0:
                speed = self.i_print / (time_elapsed - time_last_print) * 3600
                columns = [
                    f'Step: {self.step}/{self.max_steps} ({self.step / self.max_steps * 100:.2f}%)',
                    f'Elapsed: {time_elapsed / 3600:.2f} h',
                    f'Speed: {speed:.2f} steps/h',
                    f'ETA: {(self.max_steps - self.step) / speed:.2f} h',
                ]
                print(' | '.join([c.ljust(25) for c in columns]), flush=True)
                time_last_print = time_elapsed
            
            # Update progress bar with tqdm
            if self.is_master and pbar is not None:
                pbar.update(1)

                # Calculate current epoch
                current_epoch = self.step / steps_per_epoch
                total_samples = self.step * samples_per_step

                # Update progress bar description with current info
                if self.step % self.i_print == 0:
                    speed = self.i_print / (time_elapsed - time_last_print) * 3600 if time_elapsed > time_last_print else 0
                    eta_seconds = (self.max_steps - self.step) / speed * 3600 if speed > 0 else 0

                    # Format postfix info
                    postfix_dict = {
                        'Epoch': f'{current_epoch:.2f}',
                        'Samples': f'{total_samples}',
                        'Speed': f'{speed:.0f} steps/h',
                        'ETA': f'{eta_seconds / 3600:.2f}h' if eta_seconds > 0 else 'N/A'
                    }

                    # Add loss info if available
                    if step_log is not None:
                        for key, value in step_log.items():
                            if isinstance(value, (int, float)):
                                postfix_dict[key] = f'{value:.4f}'

                    pbar.set_postfix(postfix_dict)
                    time_last_print = time_elapsed

            # Check ddp
            if self.parallel_mode == 'ddp' and self.world_size > 1 and self.i_ddpcheck is not None and self.step % self.i_ddpcheck == 0:
                self.check_ddp()

            # # Sample images
            # if self.step % self.i_sample == 0:
            #     self.snapshot()

            if self.is_master:
                self.log.append((self.step, {}))

                # Log time
                self.log[-1][1]['time'] = {
                    'step': time_end - time_start,
                    'elapsed': time_elapsed,
                }

                # Log losses
                if step_log is not None:
                    self.log[-1][1].update(step_log)

                # Log scale
                if self.mix_precision_dtype == torch.float16:
                    if self.mix_precision_mode == 'amp':
                        self.log[-1][1]['scale'] = self.scaler.get_scale()
                    elif self.mix_precision_mode == 'inflat_all':
                        self.log[-1][1]['log_scale'] = self.log_scale

                # Save log
                if self.step % self.i_log == 0:
                    self.save_logs()

                # Save checkpoint
                if self.step % self.i_save == 0:
                    self.save()

            # Sample images
            if self.step % self.i_sample == 0:
                self.snapshot()
                    
            # Check abort
            self.check_abort()

        # Close progress bar
        if self.is_master and pbar is not None:
            pbar.close()
        
        self.snapshot(suffix='final', batch_size=self.snapshot_batch_size)
        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            self.writer.close()
            print('Training finished.')
            
    def profile(self, wait=2, warmup=3, active=5):
        """
        Profile the training loop.
        """
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, 'profile')),
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(wait + warmup + active):
                self.run_step()
                prof.step()
