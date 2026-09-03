# Datasets

The datasets are **not included** in this repository due to their size.

Expected structure:

```
datasets/
├── ERP_3D_FRONT/        # training set (~12,120 rooms)
└── ERP_3D_FRONT_test/   # test set (~1,144 rooms)
```

Each room follows:

```
{uuid}/{room_name}/
├── mesh/                       # full_room_wo_ceiling.obj, individual_assets/*.glb
├── erp/                        # {view}_colors.png, {view}_depth.npy
├── 3d_bounding_box/            # {room_name}_scene_data.npz
├── cubic_fov_120/{view}/       # 6 cubemap faces
├── ss_latents/, shape_latents/, pbr_latents/, dual_grid_*/, pbr_voxels_*/
└── camera_poses.json
```

## Download & train/test split

The processed dataset is released on **Hugging Face** as
[**ERP-FRONT-30K**](https://huggingface.co/datasets/GwanHyeong/ERP-FRONT-30K), sharded into
five parts (`ERP_3D_FRONT_1` … `ERP_3D_FRONT_5`) purely to keep each download manageable.
The shards are **not** pre-split into train/test, every scene sits in a flat `{uuid}/` pool.

```sh
hf download GwanHyeong/ERP-FRONT-30K --repo-type dataset --local-dir datasets/
```

After downloading, split the scenes into the train and test sets using
[`train_test_scene_mapping.json`](train_test_scene_mapping.json). Its keys:

| Key | Count | Meaning |
|-----|-------|---------|
| `train_scenes` | 5,462 | scene UUIDs that belong to the **training** set |
| `test_scenes`  | 500   | scene UUIDs that belong to the **test** set |
| `moved_scenes` | 221   | bookkeeping only, a subset of `train_scenes` (relocated during preprocessing) |
| `statistics`   | –     | totals for the above |

`train_scenes` and `test_scenes` are disjoint, so routing every `{uuid}` by whether it appears
in `test_scenes` fully reconstructs the split. Run this once from the repo root:

```python
import json, shutil
from pathlib import Path

root = Path("datasets")
test = set(json.load(open(root / "train_test_scene_mapping.json"))["test_scenes"])

train_dir, test_dir = root / "ERP_3D_FRONT", root / "ERP_3D_FRONT_test"
train_dir.mkdir(exist_ok=True); test_dir.mkdir(exist_ok=True)

moved = {"train": 0, "test": 0}
for i in range(1, 6):                       # ERP_3D_FRONT_1 … ERP_3D_FRONT_5
    shard = root / f"ERP_3D_FRONT_{i}"
    if not shard.is_dir():
        continue
    for scene in shard.iterdir():           # scene == one {uuid} directory
        if not scene.is_dir():
            continue
        split = "test" if scene.name in test else "train"
        shutil.move(str(scene), str((test_dir if split == "test" else train_dir) / scene.name))
        moved[split] += 1
    shard.rmdir()                           # remove the now-empty shard folder

print(moved)   # -> {'train': 5462, 'test': 500}
```

`shutil.move` is an instant rename when the shards and target dirs share a filesystem (they do
here), so no data is copied. The result matches the layout at the top of this file:
`datasets/ERP_3D_FRONT/` (train) and `datasets/ERP_3D_FRONT_test/` (test).

Alternatively, regenerate everything from raw 3D-FRONT with the scripts in `data_toolkit/erp/`
(see the top-level `README.md`).
