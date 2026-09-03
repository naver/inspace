# InSpace Evaluation

The evaluation runs in three steps on `datasets/ERP_3D_FRONT_test`, all driven by the
config-driven pipeline in [`pipeline/`](pipeline/):

```
recon_gt.py      ─┐
                  ├─►  compute_metrics.py   ─►  per-config metrics (JSON/CSV)
eval_pipeline.py ─┘
   (GT recon)        (batch inference)            (evaluation)
```

Everything writes under `evals/` (git-ignored). GT reconstructions and predictions share the same
`[-0.5, 0.5]` coordinate space, so comparison needs no ICP alignment.

---

## `pipeline/` — the inference + evaluation flow

| File | Role |
|------|------|
| `recon_gt.py` | **GT reconstruction** — decode GT latents → `scene/layout/assets` GLBs (upper bound). Run once. |
| `eval_pipeline.py` | **Batch inference** — end to end: Stage 1 (SS) → BBox → Stage 2 shape → Stage 2 texture → mesh + vis. |
| `compute_metrics.py` | **Evaluation** — 3D (voxel-IoU, Chamfer, F1; scene- and asset-level) + 2D (PSNR/SSIM/LPIPS). |
| `decode_glb.py` | Standalone re-decode of saved latents → GLB (no re-inference). |
| `analyze_by_camera.py` | Break metrics down by camera↔room geometry. |
| `run_*.sh` | Ready-to-run entry points (below). |

### Step 1 — GT reconstruction (once)

```sh
bash eval/pipeline/run_recon_gt.sh      # → evals/gt_recon/{scene}/{room}/meshes/*.glb
```

### Step 2 — batch inference (our results, with alpha variation)

`eval_pipeline.py` is a 2×2 matrix of `--noise_mode {random,sdedit}` × `--bbox_mode {gt,predicted}`,
plus the SDEdit strength `--sdedit_alpha` and `--enable_texture`:

| Script | Noise | BBox | α | Scope |
|--------|-------|------|---|-------|
| `run_random_gt.sh` | random | GT | – | all samples (no structure prior, upper-bound boxes) |
| `run_sdedit_predicted.sh` | SDEdit | predicted | 0.5 | **all** samples — main config |
| `run_texture_sdedit_predicted_0.{3,5,7}.sh` | SDEdit | predicted | 0.3 / 0.5 / 0.7 | **first 200** — α sweep (`--skip_existing`) |

```sh
bash eval/pipeline/run_sdedit_predicted.sh                 # full α=0.5 run over the whole test set
bash eval/pipeline/run_texture_sdedit_predicted_0.5.sh     # 200-sample α sweep (0.3 / 0.5 / 0.7)
```

Every run produces textured meshes (`--enable_texture` defaults on in `eval_pipeline.py`). Outputs →
`evals/stage12_pipeline/{config}/{scene}/{room}/` (`shape_latent.npz`, `texture_latent.npz`,
`bboxes.npz`, `meshes/`, `vis_pred/`, and `vis_concat/` with `--save_concat`). The α-sweep scripts use `--skip_existing` to reuse latents already
computed by the full run.

Key flags: `--max_samples N` (`-1` = all), `--num_vis N`, `--save_concat`, `--max_meshes N`,
`--skip_existing`, `--rank/--world_size` (multi-GPU sharding).

### Step 3 — metrics

```sh
bash eval/pipeline/run_metrics.sh       # loops over the 4 configs → metrics per config
```

`compute_metrics.py` matches predicted assets to GT by **voxel centroid** (from `shape_latent.npz`
`part_layouts`), which avoids the `bboxes.npz` ordering mismatch of the old bbox-center matching
(kept in `../legacy/` for reference). Scene-level metrics are independent of the matching method.
Select metrics with `--metrics voxel_iou chamfer f1 psnr ssim lpips`.

---

## `stage1/` — component evaluations (supplementary)

Not part of the end-to-end path above; these back specific ablation tables.

- **3D BBox estimator** — `bbox_inference_centerpoint_obb_grid.py` (newest; AABB/OBB AP + per-sample
  grid vis), plus `_obb.py` / `.py` variants. Evaluates the released `bbox_centerpoint` checkpoint.
- **Stage-1 (coarse geometry / SS) ablation** —
  `step1_generate.py` → `step2_metrics_vis.py` → `step3_metrics_vs_scene_properties.py`
  (SS voxels from a checkpoint; IoU/Dice/Chamfer + metric-vs-scene-property plots) → `evals/ss_generated/`.

---

## `viewers/` — visualization

- `gt_vs_pipeline_unified_viewer.py` — interactive Gradio viewer: GT recon + all pipeline modes,
  scene/layout/assets detail, ceiling removal, input ERP + cubemap.
- `create_stage12_comparison.py` — static comparison grids across methods, for paper figures.

Older/superseded viewers (per-mode, GT-recon-only, turntable-video, and the cross-method batch
viewer) are in `viewers/legacy/`.

<!-- ## `legacy/` — not for release use

Kept for reference, not runnable out of the box: the cross-method batch eval
(`eval_inspace_batch*.py`, which needs a curated `perspective_eval_dataset_selected.json` +
baseline outputs), the dataset-construction scripts, the old bbox-center metrics
(`compute_metrics_v1_bbox_match.py`), and the 3D-FRONT VAE eval. -->
