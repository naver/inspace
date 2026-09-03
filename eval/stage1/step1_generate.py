# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Step 1: Generate sparse structure voxels from trained checkpoint.

Loads a trained sparse structure generation model, runs inference on test data,
decodes generated latents to [1,64,64,64] voxels, and saves results.

Output structure:
    {output_dir}/{uuid}/{room_name}/{view_idx:04d}.npz
    Each NPZ contains:
        'voxel': [1, 64, 64, 64] bool - decoded binary voxel grid
        'z': [8, 16, 16, 16] float16 - raw latent

Usage:
    # Single process
    python eval/step1_generate.py \
        --config configs/gen/erp_ss_flow_img_dit_L_16l8_bf16_spatial.json \
        --ckpt_dir results/erp_ss_flow_img_dit_L_16l8_bf16_spatial \
        --data_dir datasets/ERP_3D_FRONT_test \
        --gpu_id 0

    # Multi-process (4 workers)
    for i in 0 1 2 3; do
        python eval/step1_generate.py \
            --config configs/gen/erp_ss_flow_img_dit_L_16l8_bf16_spatial.json \
            --ckpt_dir results/erp_ss_flow_img_dit_L_16l8_bf16_spatial \
            --data_dir datasets/ERP_3D_FRONT_test \
            --rank $i --world_size 4 --gpu_id $i &
    done
"""

import os
import sys
import json
import glob
import argparse
from tqdm import tqdm

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trellis2 import models, datasets
from trellis2.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from trellis2.trainers.flow_matching.mixins.erp_image_conditioned import (
    ERPImageEncoder,
    create_spatial_attention_mask,
)


def find_latest_ckpt(ckpt_dir, use_ema=True, ema_rate=0.9999):
    """Find the latest checkpoint step in ckpt_dir/ckpts/."""
    ckpts_dir = os.path.join(ckpt_dir, 'ckpts')
    if use_ema:
        pattern = f'denoiser_ema{ema_rate}_step*.pt'
    else:
        pattern = 'denoiser_step*.pt'
    files = glob.glob(os.path.join(ckpts_dir, pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No checkpoint files matching '{pattern}' in {ckpts_dir}")
    steps = [int(os.path.basename(f).split('step')[-1].split('.')[0]) for f in files]
    return max(steps)


def load_denoiser(config, ckpt_dir, ckpt_step, device='cuda', use_ema=True, ema_rate=0.9999):
    """
    Load the denoiser model from checkpoint.

    Args:
        config: Parsed config dict
        ckpt_dir: Checkpoint directory
        ckpt_step: Checkpoint step number
        device: Device to load on
        use_ema: Whether to load EMA weights (recommended for inference)
        ema_rate: EMA rate

    Returns:
        Loaded denoiser model in eval mode
    """
    # Build model from config
    model_config = config['models']['denoiser']
    denoiser = getattr(models, model_config['name'])(**model_config['args'])
    denoiser = denoiser.to(device)

    # Load checkpoint weights (prefer EMA)
    if use_ema:
        ckpt_path = os.path.join(ckpt_dir, 'ckpts', f'denoiser_ema{ema_rate}_step{ckpt_step:07d}.pt')
    else:
        ckpt_path = os.path.join(ckpt_dir, 'ckpts', f'denoiser_step{ckpt_step:07d}.pt')

    if not os.path.exists(ckpt_path):
        # Fallback: try the other variant
        alt_path = os.path.join(ckpt_dir, 'ckpts', f'denoiser_step{ckpt_step:07d}.pt')
        if use_ema and os.path.exists(alt_path):
            print(f"EMA checkpoint not found, falling back to: {alt_path}")
            ckpt_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    denoiser.load_state_dict(state_dict)
    denoiser.eval()

    print(f"Loaded denoiser from: {ckpt_path}")
    return denoiser


def load_ss_decoder(pretrained_ss_dec, device='cuda'):
    """Load the SS decoder for latent -> voxel decoding."""
    decoder = models.from_pretrained(pretrained_ss_dec)
    decoder = decoder.to(device).eval()
    print(f"Loaded SS decoder from: {pretrained_ss_dec}")
    return decoder


@torch.no_grad()
def decode_latent(decoder, z, batch_size=4):
    """Decode latent [B, 8, 16, 16, 16] -> voxel [B, 1, 64, 64, 64]."""
    ss = []
    for i in range(0, z.shape[0], batch_size):
        ss.append(decoder(z[i:i+batch_size]))
    return torch.cat(ss, dim=0)


@torch.no_grad()
def generate_samples(
    denoiser,
    erp_encoder,
    sampler,
    data,
    device='cuda',
    steps=12,
    rescale_t=5.0,
    guidance_strength=7.5,
    guidance_rescale=0.7,
    guidance_interval=(0.6, 1.0),
    use_spatial_attention=False,
    spatial_attention_kwargs=None,
    use_initial_voxel=False,
    initial_voxel_t_noise=0.5,
    sigma_min=1e-5,
):
    """
    Generate samples from noise using the denoiser.

    Args:
        denoiser: The denoiser model
        erp_encoder: ERP image encoder
        sampler: FlowEulerGuidanceIntervalSampler
        data: Dict with 'cond' (cubemap images) and optionally 'camera_center', 'initial_voxel_latent'
        device: Device to use for computation
        steps: Number of denoising steps
        rescale_t: Time rescale factor for flow matching
        guidance_strength: CFG guidance strength
        guidance_interval: Interval for applying guidance (tuple of start, end)
        use_spatial_attention: Whether to use spatial attention mask
        spatial_attention_kwargs: Dict with spatial attention parameters
        use_initial_voxel: Whether to use initial voxel latent (SDEdit-style)
        initial_voxel_t_noise: Noise level for SDEdit (0=clean, 1=pure noise). Default 0.5.
        sigma_min: Minimum noise level from flow matching training.

    Returns:
        Generated latent tensor [B, 8, 16, 16, 16]
    """
    # Encode cubemap images
    cond = erp_encoder(data['cond'].to(device))  # [B, 6*N, 1024]
    neg_cond = torch.zeros_like(cond)

    # Generate noise or mix initial voxel latent with noise (SDEdit-style)
    B = data['cond'].shape[0]
    if use_initial_voxel and 'initial_voxel_latent' in data:
        # SDEdit: x_t = (1 - t) * x_init + (sigma_min + (1 - sigma_min) * t) * gaussian_noise
        # Same formula as training (erp_image_conditioned.py line 276, 357)
        x_init = data['initial_voxel_latent'].to(device)
        gaussian_noise = torch.randn_like(x_init)
        t = initial_voxel_t_noise
        noise = (1 - t) * x_init + (sigma_min + (1 - sigma_min) * t) * gaussian_noise
    else:
        noise = torch.randn(B, 8, 16, 16, 16, device=device)

    # Build extra kwargs for sampler (passed through to model inference)
    extra_kwargs = {}

    # Add spatial attention mask if needed
    if use_spatial_attention and 'camera_center' in data:
        cross_attn_mask = create_spatial_attention_mask(
            camera_center=data['camera_center'].to(device),
            **spatial_attention_kwargs,
        )
        extra_kwargs['cross_attn_mask'] = cross_attn_mask

    # Run sampling with guidance interval (autocast for flash_attn compatibility)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        res = sampler.sample(
            denoiser,
            noise=noise,
            cond=cond,
            neg_cond=neg_cond,
            steps=steps,
            rescale_t=rescale_t,
            guidance_strength=guidance_strength,
            guidance_interval=guidance_interval,
            guidance_rescale=guidance_rescale,
            verbose=False,
            **extra_kwargs,
        )

    return res.samples


def main():
    parser = argparse.ArgumentParser(description='Generate SS voxels from trained checkpoint')
    # parser.add_argument('--config', type=str, required=True,
    #                     help='Training config JSON path')
    # parser.add_argument('--ckpt_dir', type=str, required=True,
    #                     help='Checkpoint directory (e.g., results/erp_ss_flow_...)')
    parser.add_argument('--ckpt_step', type=str, default='latest',
                        help='Checkpoint step (integer or "latest")')
    # parser.add_argument('--data_dir', type=str, required=True,
    #                     help='Test data directory')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Output directory (default: evals/ss_generated/{config_name})')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--steps', type=int, default=12,
                        help='Number of denoising steps')
    parser.add_argument('--rescale_t', type=float, default=5.0,
                        help='Time rescale factor for flow matching')
    parser.add_argument('--guidance_strength', type=float, default=7.5,
                        help='CFG guidance strength')
    parser.add_argument('--guidance_rescale', type=float, default=0.7,
                        help='Guidance rescale factor')
    parser.add_argument('--guidance_interval', type=float, nargs=2, default=[0.6, 1.0],
                        help='Guidance interval (start end)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU to use')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Max samples to process (-1 for all)')
    parser.add_argument('--rank', type=int, default=0,
                        help='Process rank for distributed processing')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total number of processes')
    parser.add_argument('--use_initial_voxel', action='store_true', default=False,
                        help='Use initial voxel latent (depth-based) with SDEdit-style noise mixing')
    parser.add_argument('--initial_voxel_t_noise', type=float, default=0.5,
                        help='Noise level for SDEdit (0=clean, 1=pure noise). Default 0.5')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip already generated samples')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA weights (recommended for inference)')
    parser.add_argument('--no_ema', dest='use_ema', action='store_false',
                        help='Use non-EMA weights')
    parser.add_argument('--ema_rate', type=float, default=0.9999,
                        help='EMA rate for checkpoint loading')
    args = parser.parse_args()

    args.config = 'configs/gen/erp_ss_flow_img_dit_L_16l8_bf16_spatial.json'
    args.ckpt_dir = 'results/erp_ss_flow_img_dit_L_16l8_bf16_spatial'

    # args.config = 'configs/gen/erp_ss_flow_img_dit_L_16l8_bf16.json'
    # args.ckpt_dir = 'results/erp_ss_flow_img_dit_L_16l8_bf16'
    
    # inversion 0.3
    # CUDA_VISIBLE_DEVICES=0 python eval/step1_generate.py --gpu_id 0 --use_initial_voxel --initial_voxel_t_noise 0.3
    # python eval/step1_generate.py --gpu_id 0 --use_initial_voxel --initial_voxel_t_noise 0.3

    # inversion 0.5
    # CUDA_VISIBLE_DEVICES=0 python eval/step1_generate.py --gpu_id 0 --use_initial_voxel --initial_voxel_t_noise 0.5
    # python eval/step1_generate.py --gpu_id 0 --use_initial_voxel --initial_voxel_t_noise 0.5

    # inversion 0.7
    # CUDA_VISIBLE_DEVICES=0 python eval/step1_generate.py --gpu_id 0 --use_initial_voxel --initial_voxel_t_noise 0.7

    # t=0.3 → evals/ss_generated/erp_ss_flow_img_dit_L_16l8_bf16_spatial_da2_inversion_0.3
    # t=0.5 → evals/ss_generated/erp_ss_flow_img_dit_L_16l8_bf16_spatial_da2_inversion_0.5
    # t=0.7 → evals/ss_generated/erp_ss_flow_img_dit_L_16l8_bf16_spatial_da2_inversion_0.7
    # No inversion → evals/ss_generated/erp_ss_flow_img_dit_L_16l8_bf16_spatial

    args.ckpt_step = 'latest'
    args.data_dir = 'datasets/ERP_3D_FRONT_test'
    # args.gpu_id = 4
    args.max_samples = -1
    args.rank = 0
    args.world_size = 1
    args.skip_existing = True
    args.use_ema = True

    # SDEdit initial voxel settings
    # args.use_initial_voxel = True
    # args.initial_voxel_t_noise = 0.5  # try: 0.3, 0.5, 0.7

    # Auto-generate output dir based on config name and noise level
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    if args.use_initial_voxel:
        args.output_dir = f'evals/ss_generated/{config_name}_da2_inversion_{args.initial_voxel_t_noise}'
    else:
        args.output_dir = f'evals/ss_generated/{config_name}'


    # Set GPU
    device = f'cuda:{args.gpu_id}'

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    config_name = os.path.splitext(os.path.basename(args.config))[0]

    # Set output directory
    if args.output_dir == '':
        args.output_dir = os.path.join('evals', 'ss_generated', config_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine checkpoint step
    if args.ckpt_step == 'latest':
        ckpt_step = find_latest_ckpt(args.ckpt_dir, use_ema=args.use_ema, ema_rate=args.ema_rate)
    else:
        ckpt_step = int(args.ckpt_step)
    print(f"Using checkpoint step: {ckpt_step} (EMA={args.use_ema})")

    # Save eval config
    if args.rank == 0:
        eval_config = {
            'config': args.config,
            'ckpt_dir': args.ckpt_dir,
            'ckpt_step': ckpt_step,
            'data_dir': args.data_dir,
            'steps': args.steps,
            'rescale_t': args.rescale_t,
            'guidance_strength': args.guidance_strength,
            'guidance_rescale': args.guidance_rescale,
            'guidance_interval': args.guidance_interval,
            'use_initial_voxel': args.use_initial_voxel,
        }
        with open(os.path.join(args.output_dir, 'eval_config.json'), 'w') as f:
            json.dump(eval_config, f, indent=2)

    # Load denoiser model
    print("Loading denoiser model...")
    denoiser = load_denoiser(config, args.ckpt_dir, ckpt_step, device,
                             use_ema=args.use_ema, ema_rate=args.ema_rate)

    # Load ERP encoder
    print("Loading ERP encoder...")
    trainer_config = config['trainer']['args']
    erp_encoder = ERPImageEncoder(
        image_cond_model=trainer_config['image_cond_model'],
        feature_dim=1024,
    ).to(device)

    # Load view_pos_emb from checkpoint if it was saved
    # The view_pos_emb is part of the trainer state, not the denoiser
    # For eval, we use randomly initialized view_pos_emb (same as training start)
    # TODO: Save/load view_pos_emb separately if needed

    # Load SS decoder
    print("Loading SS decoder...")
    pretrained_ss_dec = config['dataset']['args'].get(
        'pretrained_ss_dec',
        'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16'
    )
    ss_decoder = load_ss_decoder(pretrained_ss_dec, device)

    # Check spatial attention config
    use_spatial_attention = trainer_config.get('use_spatial_attention', False)
    spatial_attention_kwargs = {}
    if use_spatial_attention:
        spatial_attention_kwargs = {
            'voxel_resolution': trainer_config.get('voxel_resolution', 16),
            'tokens_per_face': trainer_config.get('tokens_per_face', 1029),
            'fov_degrees': trainer_config.get('spatial_attention_fov', 120.0),
            'soft_mask': trainer_config.get('spatial_attention_soft', True),
            'soft_margin': trainer_config.get('spatial_attention_soft_margin', 0.1),
        }
        print(f"Spatial attention enabled: FOV={spatial_attention_kwargs['fov_degrees']}")

    # Create sampler
    sigma_min = trainer_config.get('sigma_min', 1e-5)
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=sigma_min)

    # Create dataset
    print("Creating dataset...")
    dataset_config = config['dataset']
    dataset_args = dict(dataset_config['args'])
    if args.use_initial_voxel:
        dataset_args['use_initial_voxel'] = True
        print("Initial voxel mode enabled: using depth voxel latent as starting point")
    dataset = getattr(datasets, dataset_config['name'])(
        args.data_dir,
        **dataset_args,
    )
    print(f"Dataset: {dataset}")

    # Get all instances
    all_instances = dataset.instances  # List of (root, house_id, room_name, view_idx)
    total = len(all_instances)

    # Split across ranks
    my_instances = all_instances[args.rank::args.world_size]
    print(f"Rank {args.rank}/{args.world_size}: processing {len(my_instances)}/{total} instances")

    my_instances = sorted(my_instances)

    # Apply max_samples limit
    if args.max_samples > 0:
        my_instances = my_instances[:args.max_samples]
        print(f"Limited to {len(my_instances)} samples")

    # Filter out already generated samples
    if args.skip_existing:
        original_count = len(my_instances)
        filtered = []
        for root, house_id, room_name, view_idx in my_instances:
            out_path = os.path.join(args.output_dir, house_id, room_name, f'{view_idx:04d}.npz')
            if not os.path.exists(out_path):
                filtered.append((root, house_id, room_name, view_idx))
        my_instances = filtered
        skipped = original_count - len(my_instances)
        if skipped > 0:
            print(f"Skipping {skipped} already generated samples")

    if len(my_instances) == 0:
        print("All samples already generated. Done.")
        return

    # Process in batches
    num_generated = 0
    num_failed = 0

    for batch_start in tqdm(range(0, len(my_instances), args.batch_size),
                            desc=f"Generating (rank {args.rank})",
                            disable=args.rank != 0):
        batch_instances = my_instances[batch_start:batch_start + args.batch_size]
        batch_size = len(batch_instances)

        try:
            # Load data for this batch
            batch_data = []
            for root, house_id, room_name, view_idx in batch_instances:
                sample = dataset.get_instance(root, house_id, room_name, view_idx)
                batch_data.append(sample)

            # Collate batch
            data = {}
            data['cond'] = torch.stack([s['cond'] for s in batch_data])  # [B, 6, 3, H, W]

            if use_spatial_attention:
                camera_centers = []
                for s in batch_data:
                    if 'camera_center' in s:
                        camera_centers.append(s['camera_center'])
                    else:
                        camera_centers.append(torch.zeros(3))
                data['camera_center'] = torch.stack(camera_centers)  # [B, 3]

            if args.use_initial_voxel:
                init_latents = []
                for s in batch_data:
                    if 'initial_voxel_latent' in s:
                        init_latents.append(s['initial_voxel_latent'])
                    else:
                        init_latents.append(torch.randn(8, 16, 16, 16))
                data['initial_voxel_latent'] = torch.stack(init_latents)  # [B, 8, 16, 16, 16]

            # Generate latents
            z = generate_samples(
                denoiser=denoiser,
                erp_encoder=erp_encoder,
                sampler=sampler,
                data=data,
                device=device,
                steps=args.steps,
                rescale_t=args.rescale_t,
                guidance_strength=args.guidance_strength,
                guidance_rescale=args.guidance_rescale,
                guidance_interval=tuple(args.guidance_interval),
                use_spatial_attention=use_spatial_attention,
                spatial_attention_kwargs=spatial_attention_kwargs,
                use_initial_voxel=args.use_initial_voxel,
                initial_voxel_t_noise=args.initial_voxel_t_noise,
                sigma_min=sigma_min,
            )  # [B, 8, 16, 16, 16]

            # Decode to voxels (decoder needs float32)
            voxels = decode_latent(ss_decoder, z.float())  # [B, 1, 64, 64, 64]
            voxels_binary = (voxels > 0).cpu()  # [B, 1, 64, 64, 64] bool

            # Save results
            for i, (root, house_id, room_name, view_idx) in enumerate(batch_instances):
                out_dir = os.path.join(args.output_dir, house_id, room_name)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f'{view_idx:04d}.npz')

                np.savez_compressed(
                    out_path,
                    voxel=voxels_binary[i].numpy(),  # [1, 64, 64, 64] bool
                    z=z[i].cpu().half().numpy(),      # [8, 16, 16, 16] float16
                )
                num_generated += 1

        except Exception as e:
            print(f"\nError processing batch starting at {batch_start}: {e}")
            import traceback
            traceback.print_exc()
            num_failed += batch_size

    # Summary
    print(f"\nRank {args.rank} Summary:")
    print(f"  Generated: {num_generated}")
    print(f"  Failed: {num_failed}")
    print(f"  Output: {args.output_dir}")

    # Clean up
    del denoiser, erp_encoder, ss_decoder
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()


