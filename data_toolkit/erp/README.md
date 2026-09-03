# InSpace Data Preparation Toolkit (ERP-FRONT)

This is the **only** data-preparation pipeline InSpace uses. It turns the raw
`ERP_3D_FRONT` indoor-scene dataset (360° panorama + 3D room + assets) into the
O-Voxels, latents, and image/depth conditions required to train and run the InSpace models.

The full 30K training set is [`ERP-FRONT-30K`](https://huggingface.co/datasets/GwanHyeong/ERP-FRONT-30K);
the examples below use the held-out `ERP_3D_FRONT_test` split, but every command works identically on the full set.

---

## Two complementary tracks

Each room is processed by **two tracks** that share one coordinate system (the room's
`normalization_info.json`). They are not "with vs without ceiling" — **both are ceiling-free**;
they differ in *what* they encode:

| Track | Files | Target mesh | Produces |
|-------|-------|-------------|----------|
| **Scene + assets** | `step1–step7 *_erp.py` | `full_room_wo_ceiling` + `individual_assets/*` | geometry/PBR O-Voxels, shape/PBR latents, **SS latent** (room only) |
| **Layout** | `step1–step6 *_layout_wo_ceiling.py` | `layout_wo_ceiling` (walls + floor + door + baseboard) | geometry/PBR O-Voxels + shape/PBR latents of the bare structure |

Because step 1 normalizes assets and layout with the **room's** bounding box, the scene,
its individual assets, and the structural layout all live in one aligned space, this is what enables
the asset-aware (OmniPart-style) generation and scene re-composition.

```
                              raw room  (ERP panorama + meshes + DA2 depth)
                                   │
        ┌──────────────────────────┼───────────────────────────┐
        │ SCENE + ASSETS           │ LAYOUT                     │ CONDITIONS (shared)
        │ step1_dump_mesh_erp      │ remove_ceiling_from_layout │ step8  ERP → 6× cubemap (FOV 120)
        │ step2_dump_pbr_erp       │ step1_..._layout_wo_ceiling│ step9  DA2 depth → PSG voxels
        │ step3_dual_grid_erp      │ step2 … step6 (layout)     │ step10 PSG voxels → SS latent (SDEdit seed)
        │ step4_voxelize_pbr_erp   │                            │
        │ step5_encode_shape_erp   │                            │
        │ step6_encode_pbr_erp     │                            │
        │ step7_encode_ss_erp      │ (no SS: room-only)         │
        └──────────────────────────┴───────────────────────────┘
```

---

## Input structure

```
datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/
├── mesh/
│   ├── full_room_wo_ceiling.obj      # scene mesh, ceiling removed  (scene track input)
│   ├── full_room.obj                 # scene mesh with ceiling (reference only)
│   ├── layout.obj                    # raw structural layout (→ layout_wo_ceiling)
│   ├── individual_assets/{asset}.glb # per-object meshes
│   └── *.png, *.mtl                  # textures
├── erp/
│   ├── {view}_colors.png             # ERP panorama  (model INPUT)
│   └── {view}_depth.npy              # DA2 depth      (condition)
├── 3d_bounding_box/{room}_scene_data.npz   # GT asset OBBs
└── camera_poses.json
```

---

## Pipeline

All commands take `--root <dataset dir>` and support distributed sharding via
`--rank <i> --world_size <n>` and resumability via `--skip_completed`.

### Scene + assets track

```bash
ROOT=datasets/ERP_3D_FRONT_test

# 1. Dump mesh → pickle. Normalizes room AND all assets by the ROOM's bbox → [-0.5, 0.5].
python data_toolkit/erp/step1_dump_mesh_erp.py --root $ROOT

# 2. Dump PBR (materials/UVs). 3D-FRONT has no PBR → metallic=0, roughness=0.5 fallbacks.
python data_toolkit/erp/step2_dump_pbr_erp.py --root $ROOT

# 3. Geometry → O-Voxels (Flexible Dual Grid). Default asset_mode=room_coord (keeps alignment).
python data_toolkit/erp/step3_dual_grid_erp.py --root $ROOT --resolution 512

# 4. Material → O-Voxels (PBR attributes).
python data_toolkit/erp/step4_voxelize_pbr_erp.py --root $ROOT --resolution 512

# 5. Encode shape (geometry) latents.
python data_toolkit/erp/step5_encode_shape_latent_erp.py --root $ROOT --resolution 512

# 6. Encode PBR (texture) latents.
python data_toolkit/erp/step6_encode_pbr_latent_erp.py --root $ROOT --resolution 512

# 7. Encode SS (sparse-structure) latents — FULL ROOM ONLY (first-stage generation target).
python data_toolkit/erp/step7_encode_ss_latent_erp.py --root $ROOT \
    --shape_latent_name shape_enc_next_dc_f16c32_fp16_512 --resolution 64
```

### Layout track

```bash
# 0. Strip ceiling from layout.obj → layout_wo_ceiling.obj (prerequisite).
python data_toolkit/erp/remove_ceiling_from_layout.py --root $ROOT

# 1–6. Same steps, layout target, reusing the room's normalization (no SS latent, no assets).
python data_toolkit/erp/step1_dump_mesh_layout_wo_ceiling.py --root $ROOT
python data_toolkit/erp/step2_dump_pbr_layout_wo_ceiling.py --root $ROOT
python data_toolkit/erp/step3_dual_grid_layout_wo_ceiling.py --root $ROOT --resolution 512
python data_toolkit/erp/step4_voxelize_pbr_layout_wo_ceiling.py --root $ROOT --resolution 512
python data_toolkit/erp/step5_encode_shape_layout_wo_ceiling.py --root $ROOT --resolution 512
python data_toolkit/erp/step6_encode_pbr_layout_wo_ceiling.py --root $ROOT --resolution 512
```

### Conditioning (shared by both tracks)

```bash
# 8. ERP panorama → 6 cubemap faces (FOV 120) — the image condition.
python data_toolkit/erp/step8_render_cubic_fov_120.py --root $ROOT --fov 120

# 9. Depth-Anything-2 depth → Partial Scene Geometry (PSG) voxels.
#    --remove_ceiling --ceiling_threshold 0.2 matches the demo default.
python data_toolkit/erp/step9_erp_depth_da2_to_voxels.py --root $ROOT --resolution 64 \
    --remove_ceiling --ceiling_threshold 0.2

# 10. Encode PSG voxels → SS latent (the SDEdit seed used at inference).
python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root $ROOT --resolution 64
```

---

## Output structure

```
datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/
├── mesh_dumps/            full_room_wo_ceiling.pickle, individual_assets/*.pickle, normalization_info.json
├── pbr_dumps/            (same layout as mesh_dumps)
├── dual_grid_{res}/       full_room_wo_ceiling.vxz, individual_assets_{room_coord,normalized}/*.vxz
├── pbr_voxels_{res}/     (geometry ↔ PBR O-Voxels)
├── shape_latents/{enc}_{res}/   full_room_wo_ceiling.npz + individual_assets_*/*.npz
├── pbr_latents/{enc}_{res}/     (texture latents)
├── ss_latents/{enc}_{ss_res}/   full_room_wo_ceiling.npz     # room only
├── cubic_fov_120/{view}/{front,right,back,left,top,bottom}.png
├── cubic_fov_120_concat/{view}_concat.png
└── depth_voxels_da2_{ss_res}/   PSG voxels + SS-latent seed (steps 9–10)
```

The layout track writes `layout_wo_ceiling.*` alongside the same `dual_grid_*`, `pbr_voxels_*`,
`shape_latents/*`, and `pbr_latents/*` folders.

---

## Modes

**Processing scope** (`--mode`, scene track): `all` (default) · `room_only` · `assets_only`.

**Asset voxelization** (`--asset_mode`, steps 3–6):
- `room_coord` (default) — assets keep their position in the room (OmniPart-style, for scene-level training).
- `normalized` — each asset re-normalized to its own `[-0.5, 0.5]` for max resolution (object-level training, loses spatial context).
- `both` — generate both.

---

## Distributed processing

Every step shards by rank; run one process per shard:

```bash
for r in 0 1 2 3; do
  python data_toolkit/erp/step3_dual_grid_erp.py --root $ROOT --resolution 512 \
      --rank $r --world_size 4 &
done; wait
```

---

## Notes

**3D-FRONT PBR.** 3D-FRONT ships no metallic/roughness maps, so `step2` uses fallbacks:
`metallicFactor=0.0`, `roughnessFactor=0.5`, base color from vertex colors/textures (gray `0.8` otherwise).

**Room-anchored normalization (step 1).** The room mesh is scaled to the `[-0.5, 0.5]` unit cube;
assets and layout are transformed with the **same** center/scale, preserving spatial alignment across
scene, assets, and layout.

**Cubemap layout.** The 6 faces (FOV 120) are arranged in a cross:

```
        top
  left  front  right  back
        bottom
```
`front` yaw 0 · `right` 90 · `back` 180 · `left` 270 · `top` pitch +90 · `bottom` pitch −90.
