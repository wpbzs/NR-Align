# NR-Align

Code release for:

**NR-Align: Non-Rigid Alignment for Non-simultaneous Two-View 3D Coronary Reconstruction in Complex Cardiac Interventions**

This toolkit builds synthetic non-simultaneous two-view training cases, converts them into backprojection volumes, and trains a compact 3D alignment network.

It does not include private clinical data, hospital preprocessing code, clinical evaluation scripts, or the upstream prior reconstruction model.

## What Is Included

- `generate_cases.py`: create asynchronous non-rigid two-view vessel masks.
- `build_bp.py`: project masks and build SIRT backprojection volumes.
- `train_nr_align.py`: train the NR-Align correction network.
- `nr_align_models.py`: model definitions.
- `deform.py`: deformation and mask-processing utilities.
- `geometry.py`: metadata parsing and TIGRE geometry helpers.

## Install

```bash
conda create -n nr-align python=3.10 -y
conda activate nr-align
pip install -r requirements.txt
```

TIGRE is required only for `build_bp.py`:

```bash
python -c "import tigre; print('TIGRE OK')"
```

## Input Data

Prepare one binary 3D vessel mask and one geometry file per case:

```text
data/gt128/{ID}.npy       # shape=(128,128,128), uint8 {0,1}, XYZ order
data/meta/{ID}.npz        # projection geometry metadata
```

Generated training cases are written as:

```text
data/cases/ID001/amp6/k00/
|-- view1_mask.npy
|-- view2_mask.npy
|-- meta.json
|-- bp012_nr.npy
`-- bp012_gt.npy
```

`bp012_nr.npy` is the non-simultaneous two-view input. `bp012_gt.npy` is the aligned target.

## Quick Start

Generate non-rigid two-view cases:

```bash
python scripts/generate_cases.py \
  --ids 1-100 \
  --gt_root data/gt128 \
  --meta_npz_dir data/meta \
  --out_root data/cases \
  --amp_list_mm 6,10,14,18 \
  --ks 0-2 \
  --variant_map sym_small,sym_large,same_side_small \
  --device cuda:0 \
  --overwrite 1
```

Build backprojection volumes:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/build_bp.py \
  --data_root data/cases \
  --gt_dir data/gt128 \
  --meta_npz_dir data/meta \
  --ids 1-100 \
  --amps 6,10,14,18 \
  --ks 0-2 \
  --nDetector 512 \
  --sirt_iters 1 \
  --save_proj 1 \
  --overwrite 1
```

Train NR-Align:

```bash
python scripts/train_nr_align.py \
  --data_root data/cases \
  --work_dir runs/nr_align_dual_ds_amp6_18 \
  --ids 1-100 \
  --amps 6,10,14,18 \
  --ks 0-2 \
  --variant dual_ds \
  --epochs 120 \
  --batch 1 \
  --grad_accum 4 \
  --lr 2e-4 \
  --wd 1e-4 \
  --base 24 \
  --amp 1 \
  --seed 0
```

Training writes:

```text
runs/nr_align_dual_ds_amp6_18/
|-- nr_align_best.pt
|-- nr_align_ckpt_ep*.pt
|-- nr_align_run_meta.json
`-- nr_align_train_history.json
```

## Model Variants

- `single`: union output only.
- `dual`: union and overlap outputs.
- `dual_ds`: union and overlap outputs with 64^3 / 32^3 deep supervision.

`dual_ds` is the recommended default.

## More Details

- [Data format](docs/DATA_FORMAT.md)
- [Parameters](docs/PARAMETERS.md)
- [Recommended public parameters](configs/recommended_public_params.md)

Choose an open-source license before publishing.
