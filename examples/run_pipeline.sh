#!/usr/bin/env bash
set -euo pipefail

# Example end-to-end run. Edit paths and ID range before running.

GT_ROOT="data/gt128"
META_DIR="data/meta"
CASE_ROOT="data/cases"
RUN_DIR="runs/nr_align_dual_ds_amp6_18"
IDS="1-100"
AMPS="6,10,14,18"
KS="0-2"

python scripts/generate_cases.py \
  --ids "$IDS" \
  --gt_root "$GT_ROOT" \
  --meta_npz_dir "$META_DIR" \
  --out_root "$CASE_ROOT" \
  --amp_list_mm "$AMPS" \
  --ks "$KS" \
  --variant_map sym_small,sym_large,same_side_small \
  --device cuda:0 \
  --overwrite 1

CUDA_VISIBLE_DEVICES=0 python scripts/build_bp.py \
  --data_root "$CASE_ROOT" \
  --gt_dir "$GT_ROOT" \
  --meta_npz_dir "$META_DIR" \
  --ids "$IDS" \
  --amps "$AMPS" \
  --ks "$KS" \
  --nDetector 512 \
  --sirt_iters 1 \
  --save_proj 1 \
  --overwrite 1

python scripts/train_nr_align.py \
  --data_root "$CASE_ROOT" \
  --work_dir "$RUN_DIR" \
  --ids "$IDS" \
  --amps "$AMPS" \
  --ks "$KS" \
  --variant dual_ds \
  --epochs 120 \
  --batch 1 \
  --grad_accum 4 \
  --lr 2e-4 \
  --wd 1e-4 \
  --base 24 \
  --amp 1 \
  --seed 0
