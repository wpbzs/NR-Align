#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-async physiology-aware non-rigid generator (final strategy version).

Design goals
------------
- Keep the SAME safe backbone as your strict ONE-FIX generator:
  SVF + exp(scaling& squaring) + SDF warp + topology-preserving gates.
- Upgrade to dual-sided asynchronous deformation (both views deformed).
- Make all critical knobs CLI-configurable (no code edits for thresholds/strengths).
- Add physiologically interpretable Delta-t sampling variants:
  * sym_small      : small pair phase gap, views on opposite sides of anchor phase
  * sym_large      : large pair phase gap, views on opposite sides of anchor phase
  * same_side_small: small pair phase gap, both views on same side of anchor phase
- Add center/axis jitter to avoid always rotating around GT COM exactly.

Output layout (same as before, so your BP builder/eval can be reused):
  out_root/IDxxx/ampYY/kZZ/{view1_mask.npy, view2_mask.npy, meta.json}

NOTE:
- This script intentionally imports helper functions from your existing
  deform.py to avoid duplicating the SDF/SVF/gate implementations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception as e:
    raise RuntimeError("This script requires PyTorch.") from e

# Reuse proven helper stack from your strict generator
import deform as base


# ---------------------------- local utils ----------------------------

def amp_tag(a: float) -> str:
    # Keep folder names compatible with the existing amp-folder convention: amp5 / amp10 / amp15 (no zero padding)
    if abs(a - round(a)) < 1e-6:
        return f"amp{int(round(a))}"
    return f"amp{a:.1f}".replace('.', 'p')


def wrap01(x: float) -> float:
    return float(x % 1.0)


def parse_range(s: str) -> Tuple[float, float]:
    vals = [float(x.strip()) for x in str(s).split(',') if x.strip()]
    if len(vals) != 2:
        raise ValueError(f"Expected 'min,max', got: {s}")
    lo, hi = vals
    if hi < lo:
        lo, hi = hi, lo
    return float(lo), float(hi)


def phase_amp(t: float, phase_offset: float = 0.0) -> float:
    return float(math.sin(2.0 * math.pi * float(t) + float(phase_offset)))


def sample_variant_dt_ms(rng: np.random.RandomState, variant: str, args) -> Tuple[float, str]:
    v = variant.lower().strip()
    if v == 'sym_small':
        lo, hi = parse_range(args.dt_ms_sym_small)
    elif v == 'sym_large':
        lo, hi = parse_range(args.dt_ms_sym_large)
    elif v in ('same_side_small', 'same_small'):
        lo, hi = parse_range(args.dt_ms_same_side_small)
        v = 'same_side_small'
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return float(rng.uniform(lo, hi)), v


def dt_ms_to_phase_delta(dt_ms: float, period_ms: float) -> float:
    # normalized cycle fraction [0,1) (but keep actual fraction before wrap for readability if >1 cycle)
    return float(dt_ms / max(float(period_ms), 1e-6))


def sample_dual_phases(rng: np.random.RandomState, variant: str, args) -> Dict[str, float]:
    """
    Produce (t_resp_v1,t_resp_v2,t_card_v1,t_card_v2) using a shared physical delta-t (ms),
    mapped to respiratory and cardiac phase deltas via their periods.
    """
    dt_ms, variant_norm = sample_variant_dt_ms(rng, variant, args)

    # Anchor phases (dataset-level "canonical" phase around which async offsets are sampled)
    t0_resp = wrap01(float(args.anchor_resp))
    t0_card = wrap01(float(args.anchor_card))

    d_resp = dt_ms_to_phase_delta(dt_ms, float(args.resp_period_ms))
    d_card = dt_ms_to_phase_delta(dt_ms, float(args.card_period_ms))

    # Optional per-sample jitter in the effective dt mapping (keeps distributions less discrete)
    if float(args.dt_phase_jitter_frac) > 0:
        j = 1.0 + float(rng.uniform(-args.dt_phase_jitter_frac, args.dt_phase_jitter_frac))
        d_resp *= j
        d_card *= j

    if variant_norm in ('sym_small', 'sym_large'):
        # Opposite sides of the same anchor (what你说的“分布在GT两边”)
        t_resp_v1 = wrap01(t0_resp - 0.5 * d_resp)
        t_resp_v2 = wrap01(t0_resp + 0.5 * d_resp)
        t_card_v1 = wrap01(t0_card - 0.5 * d_card)
        t_card_v2 = wrap01(t0_card + 0.5 * d_card)
        same_side = 0
        side_sign = 0
        center_shift_ms = 0.0
    else:
        # Same-side: both phases are on the same side of the anchor, still separated by dt
        # Choose a center shift > 0.5*dt to keep both on same side.
        shift_lo, shift_hi = parse_range(args.same_side_center_shift_ms)
        # ensure physically valid shift wrt dt
        shift_lo = max(shift_lo, 0.5 * dt_ms + 1e-3)
        shift_hi = max(shift_hi, shift_lo + 1e-3)
        center_shift_ms = float(rng.uniform(shift_lo, shift_hi))
        side_sign = -1.0 if rng.rand() < 0.5 else 1.0

        c_resp = wrap01(t0_resp + side_sign * dt_ms_to_phase_delta(center_shift_ms, float(args.resp_period_ms)))
        c_card = wrap01(t0_card + side_sign * dt_ms_to_phase_delta(center_shift_ms, float(args.card_period_ms)))

        t_resp_v1 = wrap01(c_resp - 0.5 * d_resp)
        t_resp_v2 = wrap01(c_resp + 0.5 * d_resp)
        t_card_v1 = wrap01(c_card - 0.5 * d_card)
        t_card_v2 = wrap01(c_card + 0.5 * d_card)
        same_side = 1

    return {
        'variant': variant_norm,
        'dt_ms': float(dt_ms),
        'd_resp_cycle': float(d_resp),
        'd_card_cycle': float(d_card),
        't_resp_v1': float(t_resp_v1),
        't_resp_v2': float(t_resp_v2),
        't_card_v1': float(t_card_v1),
        't_card_v2': float(t_card_v2),
        'anchor_resp': float(t0_resp),
        'anchor_card': float(t0_card),
        'same_side': int(same_side),
        'same_side_sign': float(side_sign),
        'same_side_center_shift_ms': float(center_shift_ms),
    }


def sample_center_axis_jitter(rng: np.random.RandomState, center_zyx: Tuple[float, float, float], args) -> Tuple[Tuple[float, float, float], np.ndarray]:
    """Return jittered center (zyx) and unit twist axis n (zyx)."""
    cz, cy, cx = center_zyx
    cj = float(args.center_jitter_vox)
    center_j = (
        float(cz + rng.uniform(-cj, cj)),
        float(cy + rng.uniform(-cj, cj)),
        float(cx + rng.uniform(-cj, cj)),
    )

    # Start from +z axis in zyx coordinates => [1,0,0]
    # Tilt via small random y/x components; normalize.
    tilt_deg = float(args.axis_tilt_deg)
    tilt_rad = math.radians(tilt_deg)
    ty = math.tan(float(rng.uniform(-tilt_rad, tilt_rad)))
    tx = math.tan(float(rng.uniform(-tilt_rad, tilt_rad)))
    n = np.array([1.0, ty, tx], dtype=np.float32)
    n /= max(float(np.linalg.norm(n)), 1e-8)
    return center_j, n


def _torch_coords_zyx(D: int, H: int, W: int, center_zyx: Tuple[float, float, float], device):
    cz, cy, cx = center_zyx
    z = (torch.arange(D, device=device, dtype=torch.float32) - float(cz)).view(D, 1, 1)
    y = (torch.arange(H, device=device, dtype=torch.float32) - float(cy)).view(1, H, 1)
    x = (torch.arange(W, device=device, dtype=torch.float32) - float(cx)).view(1, 1, W)
    zz = z.expand(D, H, W)
    yy = y.expand(D, H, W)
    xx = x.expand(D, H, W)
    return zz, yy, xx


def build_physio_svf_axisjitter_vox(
    D: int,
    H: int,
    W: int,
    center_zyx: Tuple[float, float, float],
    axis_zyx_unit: np.ndarray,
    voxel_mm: float,
    device,
    resp_mm: float,
    resp_lateral_mm: float,
    twist_deg: float,
    twist_mod: float,
    win_sigma_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return base SVFs in voxel units (1,3,D,H,W), channels (dz,dy,dx):
      - v_resp: localized translation field
      - v_card: localized torsion-like field around jittered axis
      - win   : localization window (1,1,D,H,W)

    This is the dual-async upgrade of the base physio model; still smooth + exp(SVF) safe.
    """
    zz, yy, xx = _torch_coords_zyx(D, H, W, center_zyx, device)
    sig = float(win_sigma_ratio) * float(W)
    win = torch.exp(-(zz * zz + yy * yy + xx * xx) / (2.0 * sig * sig)).clamp(0, 1)

    # axis n in zyx
    n = torch.tensor(axis_zyx_unit.astype(np.float32), device=device, dtype=torch.float32)
    n = n / (torch.norm(n) + 1e-8)

    # Build a stable orthonormal frame (n, e1, e2)
    tmp = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    if torch.abs(torch.dot(n, tmp)) > 0.95:
        tmp = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
    e1 = tmp - torch.dot(tmp, n) * n
    e1 = e1 / (torch.norm(e1) + 1e-8)
    e2 = torch.cross(n, e1, dim=0)
    e2 = e2 / (torch.norm(e2) + 1e-8)

    # Respiration = localized translation with dominant z + lateral component in local frame
    resp_vox = float(resp_mm) / float(voxel_mm)
    lat_vox = float(resp_lateral_mm) / float(voxel_mm)

    # dominant direction: mostly +z in world, plus a small component orthogonal to axis (e1)
    zhat = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    tdir = zhat + (lat_vox / max(resp_vox, 1e-6)) * e1
    tdir = tdir / (torch.norm(tdir) + 1e-8)

    dz = (resp_vox * tdir[0]) * win
    dy = (resp_vox * tdir[1]) * win
    dx = (resp_vox * tdir[2]) * win
    v_resp = torch.stack([dz, dy, dx], dim=0)[None].to(torch.float32)

    # Cardiac torsion around axis n: v = omega * (n x r) * localized modulation
    # r in zyx coords
    rz, ry, rx = zz, yy, xx
    # n x r (component-wise in zyx basis treated as 3D vector)
    cxr_z = n[1] * rx - n[2] * ry
    cxr_y = n[2] * rz - n[0] * rx
    cxr_x = n[0] * ry - n[1] * rz

    # longitudinal coordinate along axis for gentle modulation
    s = n[0] * rz + n[1] * ry + n[2] * rx
    s_norm = s / (0.33 * float(D) + 1e-6)
    zprof = 0.85 + 0.15 * torch.tanh(s_norm)
    mod = (1.0 + float(twist_mod) * zprof).clamp(0.7, 1.3)

    omega = float(twist_deg) * math.pi / 180.0
    dz = omega * cxr_z * win * mod
    dy = omega * cxr_y * win * mod
    dx = omega * cxr_x * win * mod
    v_card = torch.stack([dz, dy, dx], dim=0)[None].to(torch.float32)

    return v_resp, v_card, win[None, None].to(torch.float32)


def p95_pair_gap_mm_torch(d1_vox: torch.Tensor, d2_vox: torch.Tensor, roi_t: torch.Tensor, voxel_mm: float) -> float:
    diff = d2_vox - d1_vox
    return base.p95_disp_mm_torch(diff, roi_t, voxel_mm)


def p95_pair_center_mm_torch(d1_vox: torch.Tensor, d2_vox: torch.Tensor, roi_t: torch.Tensor, voxel_mm: float) -> float:
    cen = 0.5 * (d1_vox + d2_vox)
    return base.p95_disp_mm_torch(cen, roi_t, voxel_mm)


def outside_pair_sum(mask1_256: np.ndarray, mask2_256: np.ndarray, z0: int, y0: int, x0: int) -> float:
    return float(base.outside_frac_128_in_256(mask1_256, z0, y0, x0) + base.outside_frac_128_in_256(mask2_256, z0, y0, x0))


def _save_case(case_dir: Path, v1_zyx: np.ndarray, v2_zyx: np.ndarray, meta_case: Dict[str, Any]):
    base.ensure_dir(case_dir)
    np.save(str(case_dir / 'view1_mask.npy'), base.zyx_to_xyz(v1_zyx).astype(np.uint8))
    np.save(str(case_dir / 'view2_mask.npy'), base.zyx_to_xyz(v2_zyx).astype(np.uint8))
    with open(str(case_dir / 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(base._jsonify(meta_case), f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()

    # dataset indexing
    ap.add_argument('--ids', type=str, default='1-5')
    ap.add_argument('--amp_list_mm', type=str, default='5,10,15')
    ap.add_argument('--ks', type=str, default='0-2', help='k indices to generate; mapped to variant_map cyclically')
    ap.add_argument('--variant_map', type=str, default='sym_small,sym_large,same_side_small')

    # io
    ap.add_argument('--gt_root', type=str, required=True)
    ap.add_argument('--meta_npz_dir', type=str, required=True)
    ap.add_argument('--out_root', type=str, required=True)
    ap.add_argument('--overwrite', type=int, default=0)

    # compute/device
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--voxel_mm', type=float, default=0.78125)

    # exp(SVF) + smoothing + postproc (same safety stack)
    ap.add_argument('--ss_steps', type=int, default=6)
    ap.add_argument('--use_divfree', type=int, default=1)
    ap.add_argument('--aa_sigma', type=float, default=0.7)
    ap.add_argument('--iso_offset', type=float, default=0.45)
    ap.add_argument('--connfix', type=int, default=1)
    ap.add_argument('--connfix_close_iters', type=int, default=1)

    # localization / base motion strength
    ap.add_argument('--win_sigma_ratio', type=float, default=0.50)
    ap.add_argument('--resp_base_mm', type=float, default=8.0)
    ap.add_argument('--resp_lateral_mm', type=float, default=2.0)
    ap.add_argument('--card_twist_base_deg', type=float, default=8.0)
    ap.add_argument('--card_twist_mod', type=float, default=0.10)

    # per-view asymmetry (dual but asymmetric)
    ap.add_argument('--v1_amp_scale', type=float, default=1.00)
    ap.add_argument('--v2_amp_scale', type=float, default=1.10)
    ap.add_argument('--amp_jitter', type=float, default=0.08)
    ap.add_argument('--v1_lat_scale', type=float, default=1.00)
    ap.add_argument('--v2_lat_scale', type=float, default=1.15)
    ap.add_argument('--v1_twist_scale', type=float, default=1.00)
    ap.add_argument('--v2_twist_scale', type=float, default=1.15)

    # center / axis jitter (your requested upgrade)
    ap.add_argument('--center_jitter_vox', type=float, default=1.5)
    ap.add_argument('--axis_tilt_deg', type=float, default=10.0)
    ap.add_argument('--per_view_center_jitter_extra_vox', type=float, default=0.5)
    ap.add_argument('--per_view_axis_tilt_extra_deg', type=float, default=4.0)

    # physiology timing model
    ap.add_argument('--anchor_resp', type=float, default=0.25)
    ap.add_argument('--anchor_card', type=float, default=0.25)
    ap.add_argument('--resp_period_ms', type=float, default=4000.0)
    ap.add_argument('--card_period_ms', type=float, default=850.0)
    ap.add_argument('--dt_ms_sym_small', type=str, default='40,90')
    ap.add_argument('--dt_ms_sym_large', type=str, default='140,240')
    ap.add_argument('--dt_ms_same_side_small', type=str, default='50,110')
    ap.add_argument('--same_side_center_shift_ms', type=str, default='80,220')
    ap.add_argument('--dt_phase_jitter_frac', type=float, default=0.08)

    # scaling target and pair difficulty gates
    ap.add_argument('--pair_scale_target_mode', type=str, default='gap', choices=['gap', 'max_view_p95'])
    ap.add_argument('--gate_pair_gap', type=int, default=1)
    ap.add_argument('--pair_gap_p95_mm_min', type=float, default=3.0)
    ap.add_argument('--pair_gap_p95_mm_max', type=float, default=20.0)
    ap.add_argument('--gate_pair_center', type=int, default=1)
    ap.add_argument('--pair_center_p95_mm_max', type=float, default=12.0,
                    help='Keeps pair center not drifting too far from GT anchor (physio plausibility)')
    ap.add_argument('--gate_pair_outside_sum', type=int, default=1)
    ap.add_argument('--pair_outside_sum_max', type=float, default=0.04)

    # per-view ROI and translation clamp
    ap.add_argument('--roi_dilate_big', type=int, default=3)
    ap.add_argument('--roi_dilate_crop', type=int, default=2)
    ap.add_argument('--clamp_translation', type=int, default=1)
    ap.add_argument('--clamp_safety_vox', type=int, default=8)

    # classic strict gates (per-view)
    ap.add_argument('--gate_detJ', type=int, default=1)
    ap.add_argument('--detJ_min_thr', type=float, default=0.0)
    ap.add_argument('--detJ_nonpos_max', type=float, default=0.0)

    ap.add_argument('--gate_outside', type=int, default=1)
    ap.add_argument('--outside_frac_max', type=float, default=0.02)

    ap.add_argument('--gate_border', type=int, default=1)
    ap.add_argument('--border_width', type=int, default=1)
    ap.add_argument('--border_frac_max', type=float, default=0.0)

    ap.add_argument('--gate_warpback', type=int, default=1)
    ap.add_argument('--warpback_p95_mm_max', type=float, default=2.0)
    ap.add_argument('--warpback_inv_mode', type=str, default='svf', choices=['svf', 'fixedpoint'])
    ap.add_argument('--inv_iters', type=int, default=12)

    # search
    ap.add_argument('--max_tries', type=int, default=12000)
    ap.add_argument('--print_every', type=int, default=50)

    args = ap.parse_args()

    if not base._HAS_SCIPY:
        raise RuntimeError('SciPy is required (SDF distance transform).')

    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))

    ids = base.parse_ints(args.ids)
    amps = base.parse_floats(args.amp_list_mm)
    ks = base.parse_ints(args.ks)
    variants = [v.strip() for v in str(args.variant_map).split(',') if v.strip()]
    if len(variants) <= 0:
        raise ValueError('--variant_map cannot be empty')

    gt_root = Path(args.gt_root)
    meta_dir = Path(args.meta_npz_dir)
    out_root = Path(args.out_root)
    base.ensure_dir(out_root)

    # fixed crop in 256^3
    z0 = y0 = x0 = (256 - 128) // 2
    base_grid_256 = base.make_base_grid_zyx(256, 256, 256, device=device)

    for ID in ids:
        gt_path = gt_root / f"{ID}.npy"
        npz_path = meta_dir / f"{ID}.npz"
        if not gt_path.exists():
            print(f"[skip] missing GT: {gt_path}", flush=True)
            continue
        if not npz_path.exists():
            print(f"[skip] missing meta npz: {npz_path}", flush=True)
            continue

        gt_xyz = np.load(str(gt_path)).astype(np.uint8)
        if gt_xyz.shape != (128, 128, 128):
            raise ValueError(f"GT must be 128^3 XYZ, got {gt_xyz.shape}: {gt_path}")
        gt_zyx = base.xyz_to_zyx(gt_xyz)
        if int((gt_zyx > 0).sum()) == 0:
            print(f"[skip] empty GT: {gt_path}")
            continue

        meta_geo = base.load_npz_meta(npz_path)

        # Embed GT in 256^3 center
        ref256 = np.zeros((256, 256, 256), dtype=np.uint8)
        ref256[z0:z0+128, y0:y0+128, x0:x0+128] = (gt_zyx > 0).astype(np.uint8)

        # ROIs
        roi_big = base.binary_dilation(ref256, iterations=int(args.roi_dilate_big)).astype(np.uint8)
        roi_crop = base.binary_dilation((gt_zyx > 0).astype(np.uint8), iterations=int(args.roi_dilate_crop)).astype(np.uint8)
        roi_big_t = torch.from_numpy(roi_big).to(device=device, dtype=torch.uint8)
        roi_crop_t = torch.from_numpy(roi_crop).to(device=device, dtype=torch.uint8)

        # SDF reference (256)
        sdf_ref = base.mask_to_sdf(ref256).astype(np.float32)
        sdf_ref_t = torch.from_numpy(sdf_ref)[None, None].to(device=device, dtype=torch.float32)

        # GT bbox margins in 128 crop (for translation clamp)
        margins_128 = base.bbox_margins_zyx(gt_zyx)
        com_128 = base.center_of_mass_zyx(gt_zyx)
        center_base_256 = (com_128[0] + z0, com_128[1] + y0, com_128[2] + x0)

        for amp in amps:
            atag = amp_tag(float(amp))
            for idx_k, k in enumerate(ks):
                variant = variants[idx_k % len(variants)]
                case_dir = out_root / f"ID{ID:03d}" / atag / f"k{int(k):02d}"
                meta_path = case_dir / 'meta.json'
                if meta_path.exists() and int(args.overwrite) == 0:
                    print(f"[skip] exists {meta_path}", flush=True)
                    continue

                seed_case = int(args.seed + ID * 100003 + int(round(float(amp) * 100)) * 97 + int(k) * 11)
                rng = np.random.RandomState(seed_case)

                rejects = {
                    'v1_nan': 0, 'v1_empty': 0, 'v1_detJ': 0, 'v1_outside': 0, 'v1_border': 0, 'v1_warpback': 0,
                    'v2_nan': 0, 'v2_empty': 0, 'v2_detJ': 0, 'v2_outside': 0, 'v2_border': 0, 'v2_warpback': 0,
                    'pair_gap': 0, 'pair_center': 0, 'pair_outside': 0,
                }
                ok = False

                with torch.inference_mode():
                    for attempt in range(1, int(args.max_tries) + 1):
                        # --- timing / phase sampling ---
                        ph = sample_dual_phases(rng, variant, args)

                        # --- shared center+axis jitter (pair-level), plus tiny per-view extra jitter ---
                        pair_center, pair_axis = sample_center_axis_jitter(rng, center_base_256, args)
                        center_v1_zyx, a1 = pair_center, pair_axis.copy()
                        center_v2_zyx, a2 = pair_center, pair_axis.copy()

                        ex = float(args.per_view_center_jitter_extra_vox)
                        if ex > 0:
                            center_v1_zyx = (center_v1_zyx[0] + rng.uniform(-ex, ex), center_v1_zyx[1] + rng.uniform(-ex, ex), center_v1_zyx[2] + rng.uniform(-ex, ex))
                            center_v2_zyx = (center_v2_zyx[0] + rng.uniform(-ex, ex), center_v2_zyx[1] + rng.uniform(-ex, ex), center_v2_zyx[2] + rng.uniform(-ex, ex))

                        ex_tilt = float(args.per_view_axis_tilt_extra_deg)
                        if ex_tilt > 0:
                            # apply small extra tilt by perturbing y/x then renormalizing
                            for aa in (a1, a2):
                                t = math.radians(ex_tilt)
                                aa[1] += np.float32(math.tan(rng.uniform(-t, t)))
                                aa[2] += np.float32(math.tan(rng.uniform(-t, t)))
                                aa /= max(np.linalg.norm(aa), 1e-8)

                        # --- per-view asymmetry in base strength ---
                        j1 = 1.0 + float(rng.uniform(-args.amp_jitter, args.amp_jitter))
                        j2 = 1.0 + float(rng.uniform(-args.amp_jitter, args.amp_jitter))
                        resp1 = float(args.resp_base_mm) * float(args.v1_amp_scale) * j1
                        resp2 = float(args.resp_base_mm) * float(args.v2_amp_scale) * j2
                        lat1  = float(args.resp_lateral_mm) * float(args.v1_lat_scale)
                        lat2  = float(args.resp_lateral_mm) * float(args.v2_lat_scale)
                        tw1   = float(args.card_twist_base_deg) * float(args.v1_twist_scale)
                        tw2   = float(args.card_twist_base_deg) * float(args.v2_twist_scale)

                        # --- build local bases ---
                        v1_resp0, v1_card0, _ = build_physio_svf_axisjitter_vox(
                            256, 256, 256, center_v1_zyx, a1, float(args.voxel_mm), device,
                            resp1, lat1, tw1, float(args.card_twist_mod), float(args.win_sigma_ratio)
                        )
                        v2_resp0, v2_card0, _ = build_physio_svf_axisjitter_vox(
                            256, 256, 256, center_v2_zyx, a2, float(args.voxel_mm), device,
                            resp2, lat2, tw2, float(args.card_twist_mod), float(args.win_sigma_ratio)
                        )

                        # phase amplitudes (resp phase offset 0, card phase +pi/2 to mimic old generator style)
                        a_resp1 = phase_amp(ph['t_resp_v1'], 0.0)
                        a_resp2 = phase_amp(ph['t_resp_v2'], 0.0)
                        a_card1 = phase_amp(ph['t_card_v1'], math.pi / 2.0)
                        a_card2 = phase_amp(ph['t_card_v2'], math.pi / 2.0)

                        v1_vox = (a_resp1 * v1_resp0 + a_card1 * v1_card0).to(torch.float32)
                        v2_vox = (a_resp2 * v2_resp0 + a_card2 * v2_card0).to(torch.float32)

                        if int(args.use_divfree) == 1:
                            v1_vox = base.project_divergence_free_vox_zyx(v1_vox)
                            v2_vox = base.project_divergence_free_vox_zyx(v2_vox)

                        # --- shared scaling to target amp ---
                        scale = 1.0
                        u1 = u2 = None
                        v1_norm = v2_norm = None
                        d1_vox = d2_vox = None
                        p95_big_v1 = p95_big_v2 = pair_gap_big = None

                        for _ in range(2):
                            v1_s = v1_vox * float(scale)
                            v2_s = v2_vox * float(scale)
                            v1_norm = base.vox_zyx_to_norm_xyz(v1_s, 256, 256, 256).to(torch.float32)
                            v2_norm = base.vox_zyx_to_norm_xyz(v2_s, 256, 256, 256).to(torch.float32)
                            u1 = base.svf_exp_scaling_squaring(v1_norm, base_grid_256, n_steps=int(args.ss_steps)).to(torch.float32)
                            u2 = base.svf_exp_scaling_squaring(v2_norm, base_grid_256, n_steps=int(args.ss_steps)).to(torch.float32)
                            d1_vox = base.norm_xyz_to_vox_zyx(u1, 256, 256, 256)[0]
                            d2_vox = base.norm_xyz_to_vox_zyx(u2, 256, 256, 256)[0]

                            if (not torch.isfinite(d1_vox).all()) or (not torch.isfinite(d2_vox).all()):
                                u1 = None
                                break

                            p95_big_v1 = base.p95_disp_mm_torch(d1_vox, roi_big_t, float(args.voxel_mm))
                            p95_big_v2 = base.p95_disp_mm_torch(d2_vox, roi_big_t, float(args.voxel_mm))
                            pair_gap_big = p95_pair_gap_mm_torch(d1_vox, d2_vox, roi_big_t, float(args.voxel_mm))

                            if str(args.pair_scale_target_mode) == 'gap':
                                cur = float(pair_gap_big)
                            else:
                                cur = float(max(p95_big_v1, p95_big_v2))
                            if (not np.isfinite(cur)) or cur < 1e-4:
                                u1 = None
                                break
                            scale *= float(amp) / float(cur)

                        if u1 is None:
                            rejects['v1_nan'] += 1
                            if attempt % int(args.print_every) == 0:
                                print(f"[hb] ID={ID:03d} amp={amp:g} k={int(k):02d} attempt={attempt}/{args.max_tries} rejects={rejects}", flush=True)
                            continue

                        # --- translation clamp (per-view) to reduce truncation while preserving pair dynamics ---
                        clamp_logs = {}
                        if int(args.clamp_translation) == 1:
                            for tag, uu, dd in [('v1', u1, d1_vox), ('v2', u2, d2_vox)]:
                                dc = dd[:, z0:z0+128, y0:y0+128, x0:x0+128]
                                t_mean = base.roi_mean_disp_vox_torch(dc, roi_crop_t)
                                t_clamped = base.clamp_translation_vox(t_mean, margins_128, int(args.clamp_safety_vox))
                                delta = (t_clamped[0]-t_mean[0], t_clamped[1]-t_mean[1], t_clamped[2]-t_mean[2])
                                dzn = float(delta[0]) * (2.0 / 255.0)
                                dyn = float(delta[1]) * (2.0 / 255.0)
                                dxn = float(delta[2]) * (2.0 / 255.0)
                                uu[:, 0:1] = uu[:, 0:1] + dxn
                                uu[:, 1:2] = uu[:, 1:2] + dyn
                                uu[:, 2:3] = uu[:, 2:3] + dzn
                                dd[0] = dd[0] + float(delta[0])
                                dd[1] = dd[1] + float(delta[1])
                                dd[2] = dd[2] + float(delta[2])
                                clamp_logs[f'{tag}_t_mean_vox'] = [float(x) for x in t_mean]
                                clamp_logs[f'{tag}_t_clamped_vox'] = [float(x) for x in t_clamped]
                                clamp_logs[f'{tag}_delta_vox'] = [float(x) for x in delta]

                        # --- pair crop metrics after clamp ---
                        d1_crop = d1_vox[:, z0:z0+128, y0:y0+128, x0:x0+128]
                        d2_crop = d2_vox[:, z0:z0+128, y0:y0+128, x0:x0+128]
                        pair_gap_p95 = p95_pair_gap_mm_torch(d1_crop, d2_crop, roi_crop_t, float(args.voxel_mm))
                        pair_center_p95 = p95_pair_center_mm_torch(d1_crop, d2_crop, roi_crop_t, float(args.voxel_mm))

                        if int(args.gate_pair_gap) == 1:
                            if (pair_gap_p95 < float(args.pair_gap_p95_mm_min) - 1e-12) or (pair_gap_p95 > float(args.pair_gap_p95_mm_max) + 1e-12):
                                rejects['pair_gap'] += 1
                                if attempt % int(args.print_every) == 0:
                                    print(f"[hb] ID={ID:03d} amp={amp:g} k={int(k):02d} attempt={attempt}/{args.max_tries} rejects={rejects}", flush=True)
                                continue

                        if int(args.gate_pair_center) == 1 and (pair_center_p95 > float(args.pair_center_p95_mm_max) + 1e-12):
                            rejects['pair_center'] += 1
                            if attempt % int(args.print_every) == 0:
                                print(f"[hb] ID={ID:03d} amp={amp:g} k={int(k):02d} attempt={attempt}/{args.max_tries} rejects={rejects}", flush=True)
                            continue

                        # --- per-view validation + rasterization ---
                        per_view = {}
                        ok_views = True
                        out_masks_256 = {}
                        out_masks_128 = {}
                        wb_logs = {}
                        det_logs = {}
                        border_logs = {}
                        outside_logs = {}
                        p95crop_logs = {}

                        for tag, uu, vv_norm, dd in [('v1', u1, v1_norm, d1_vox), ('v2', u2, v2_norm, d2_vox)]:
                            dcrop = dd[:, z0:z0+128, y0:y0+128, x0:x0+128]
                            p95_crop_mm = base.p95_disp_mm_torch(dcrop, roi_crop_t, float(args.voxel_mm))
                            detJ_min, detJ_nonpos = base.jacobian_det_stats_crop_torch(dcrop, roi_crop_t)

                            if int(args.gate_detJ) == 1:
                                if (not np.isfinite(detJ_min)) or (not np.isfinite(detJ_nonpos)) or detJ_min < float(args.detJ_min_thr) - 1e-12 or detJ_nonpos > float(args.detJ_nonpos_max) + 1e-12:
                                    rejects[f'{tag}_detJ'] += 1
                                    ok_views = False
                                    break

                            grid_fwd = base_grid_256 + uu.permute(0, 2, 3, 4, 1)
                            sdf_w = F.grid_sample(sdf_ref_t, grid_fwd, mode='bilinear', padding_mode='border', align_corners=True)
                            sdf_w = base.gaussian_blur3d(sdf_w, float(args.aa_sigma))
                            big_def_t = (sdf_w <= float(args.iso_offset)).to(torch.uint8)
                            big_def = big_def_t[0, 0].detach().cpu().numpy()
                            if int(big_def.sum()) == 0:
                                rejects[f'{tag}_empty'] += 1
                                ok_views = False
                                break

                            outside_frac = base.outside_frac_128_in_256(big_def, z0, y0, x0)
                            if int(args.gate_outside) == 1 and outside_frac > float(args.outside_frac_max) + 1e-12:
                                rejects[f'{tag}_outside'] += 1
                                ok_views = False
                                break

                            m128 = big_def[z0:z0+128, y0:y0+128, x0:x0+128].astype(np.uint8)
                            if int(args.connfix) == 1:
                                m128 = base.binary_closing(m128, iterations=int(args.connfix_close_iters)).astype(np.uint8)
                                m128 = base.keep_largest_cc(m128)
                            if int(m128.sum()) == 0:
                                rejects[f'{tag}_empty'] += 1
                                ok_views = False
                                break

                            bfrac, _ = base.border_frac_zyx(m128, border_width=int(args.border_width))
                            if int(args.gate_border) == 1 and bfrac > float(args.border_frac_max) + 1e-12:
                                rejects[f'{tag}_border'] += 1
                                ok_views = False
                                break

                            warpback_p95_mm = float('nan')
                            if int(args.gate_warpback) == 1:
                                if str(args.warpback_inv_mode).lower() == 'svf':
                                    u0_inv = base.svf_exp_scaling_squaring((-vv_norm).to(torch.float32), base_grid_256, n_steps=int(args.ss_steps)).to(torch.float32)
                                    if int(args.clamp_translation) == 1:
                                        # translation part already absorbed into uu, reconstruct residual translation from uu-u0 approx is overkill
                                        # use fixedpoint if exact inverse of clamped field is needed; here we accept svf inverse of unclamped + fixedpoint fallback optional.
                                        u_inv = base.invert_disp_norm_xyz(uu, base_grid_256, n_iter=max(8, int(args.inv_iters))) if int(args.inv_iters) > 0 else u0_inv
                                    else:
                                        u_inv = u0_inv
                                else:
                                    u_inv = base.invert_disp_norm_xyz(uu, base_grid_256, n_iter=int(args.inv_iters)).to(torch.float32)

                                comp1 = base.compose_disp_norm_xyz(uu, u_inv, base_grid_256)
                                comp2 = base.compose_disp_norm_xyz(u_inv, uu, base_grid_256)
                                comp1_crop = comp1[0, :, z0:z0+128, y0:y0+128, x0:x0+128]
                                comp2_crop = comp2[0, :, z0:z0+128, y0:y0+128, x0:x0+128]
                                wb1 = base.p95_disp_mm_normxyz_crop_torch(comp1_crop, roi_crop_t, float(args.voxel_mm), full_DHW=(256,256,256))
                                wb2 = base.p95_disp_mm_normxyz_crop_torch(comp2_crop, roi_crop_t, float(args.voxel_mm), full_DHW=(256,256,256))
                                warpback_p95_mm = float(max(wb1, wb2))
                                if (not np.isfinite(warpback_p95_mm)) or (warpback_p95_mm > float(args.warpback_p95_mm_max) + 1e-12):
                                    rejects[f'{tag}_warpback'] += 1
                                    ok_views = False
                                    break

                            out_masks_256[tag] = big_def.astype(np.uint8)
                            out_masks_128[tag] = m128.astype(np.uint8)
                            wb_logs[f'{tag}_warpback_p95_mm'] = float(warpback_p95_mm)
                            det_logs[f'{tag}_detJ_min'] = float(detJ_min)
                            det_logs[f'{tag}_detJ_nonpos_frac'] = float(detJ_nonpos)
                            border_logs[f'{tag}_border_frac'] = float(bfrac)
                            outside_logs[f'{tag}_outside_frac'] = float(outside_frac)
                            p95crop_logs[f'{tag}_p95_crop_mm'] = float(p95_crop_mm)
                            per_view[tag] = {
                                'p95_crop_mm': float(p95_crop_mm),
                                'detJ_min': float(detJ_min),
                                'detJ_nonpos_frac': float(detJ_nonpos),
                                'outside_frac': float(outside_frac),
                                'border_frac': float(bfrac),
                                'warpback_p95_mm': float(warpback_p95_mm),
                            }

                        if not ok_views:
                            if attempt % int(args.print_every) == 0:
                                print(f"[hb] ID={ID:03d} amp={amp:g} k={int(k):02d} attempt={attempt}/{args.max_tries} rejects={rejects}", flush=True)
                            continue

                        if int(args.gate_pair_outside_sum) == 1:
                            ps = outside_pair_sum(out_masks_256['v1'], out_masks_256['v2'], z0, y0, x0)
                            if ps > float(args.pair_outside_sum_max) + 1e-12:
                                rejects['pair_outside'] += 1
                                if attempt % int(args.print_every) == 0:
                                    print(f"[hb] ID={ID:03d} amp={amp:g} k={int(k):02d} attempt={attempt}/{args.max_tries} rejects={rejects}", flush=True)
                                continue
                        else:
                            ps = outside_pair_sum(out_masks_256['v1'], out_masks_256['v2'], z0, y0, x0)

                        # success: dual deformed views
                        meta_case: Dict[str, Any] = {}
                        meta_case.update(meta_geo)
                        meta_case.update({
                            'id': int(ID), 'ID': int(ID), 'k': int(k),
                            'mode': 'dual_async_physio',
                            'variant': str(ph['variant']),
                            'amp_target': float(amp),
                            'seed_case': int(seed_case),
                            'attempt': int(attempt),
                            'npz_path': str(npz_path),
                            'gt_path': str(gt_path),
                            'pair_scale_target_mode': str(args.pair_scale_target_mode),
                            'pair_gap_p95_mm': float(pair_gap_p95),
                            'pair_center_p95_mm': float(pair_center_p95),
                            'pair_outside_sum': float(ps),
                            'p95_big_v1_mm': float(p95_big_v1),
                            'p95_big_v2_mm': float(p95_big_v2),
                            'pair_gap_big_mm': float(pair_gap_big),
                            'center_base_256_zyx': [float(x) for x in center_base_256],
                            'pair_center_256_zyx': [float(x) for x in pair_center],
                            'pair_axis_zyx_unit': [float(x) for x in pair_axis.tolist()],
                            'v1_center_256_zyx': [float(x) for x in center_v1_zyx],
                            'v2_center_256_zyx': [float(x) for x in center_v2_zyx],
                            'v1_axis_zyx_unit': [float(x) for x in a1.tolist()],
                            'v2_axis_zyx_unit': [float(x) for x in a2.tolist()],
                            'resp_period_ms': float(args.resp_period_ms),
                            'card_period_ms': float(args.card_period_ms),
                            'dt_phase_jitter_frac': float(args.dt_phase_jitter_frac),
                            'v1_amp_scale': float(args.v1_amp_scale),
                            'v2_amp_scale': float(args.v2_amp_scale),
                            'amp_jitter': float(args.amp_jitter),
                            'v1_lat_scale': float(args.v1_lat_scale),
                            'v2_lat_scale': float(args.v2_lat_scale),
                            'v1_twist_scale': float(args.v1_twist_scale),
                            'v2_twist_scale': float(args.v2_twist_scale),
                            'center_jitter_vox': float(args.center_jitter_vox),
                            'axis_tilt_deg': float(args.axis_tilt_deg),
                            'per_view_center_jitter_extra_vox': float(args.per_view_center_jitter_extra_vox),
                            'per_view_axis_tilt_extra_deg': float(args.per_view_axis_tilt_extra_deg),
                            'use_divfree': int(args.use_divfree),
                            'ss_steps': int(args.ss_steps),
                            'aa_sigma': float(args.aa_sigma),
                            'iso_offset': float(args.iso_offset),
                            'connfix': int(args.connfix),
                            'connfix_close_iters': int(args.connfix_close_iters),
                            'win_sigma_ratio': float(args.win_sigma_ratio),
                            'resp_base_mm': float(args.resp_base_mm),
                            'resp_lateral_mm': float(args.resp_lateral_mm),
                            'card_twist_base_deg': float(args.card_twist_base_deg),
                            'card_twist_mod': float(args.card_twist_mod),
                            'gate_pair_gap': int(args.gate_pair_gap),
                            'gate_pair_center': int(args.gate_pair_center),
                            'gate_pair_outside_sum': int(args.gate_pair_outside_sum),
                            'gate_detJ': int(args.gate_detJ),
                            'gate_outside': int(args.gate_outside),
                            'gate_border': int(args.gate_border),
                            'gate_warpback': int(args.gate_warpback),
                            'rejects': rejects,
                        })
                        meta_case.update(ph)
                        meta_case.update(clamp_logs)
                        meta_case.update(p95crop_logs)
                        meta_case.update(det_logs)
                        meta_case.update(border_logs)
                        meta_case.update(outside_logs)
                        meta_case.update(wb_logs)

                        _save_case(case_dir, out_masks_128['v1'], out_masks_128['v2'], meta_case)

                        print(
                            f"[ok] ID={ID:03d} amp={amp:g} k={int(k):02d} var={ph['variant']} attempt={attempt} "
                            f"dt={ph['dt_ms']:.1f}ms gap={pair_gap_p95:.2f}mm cen={pair_center_p95:.2f}mm "
                            f"out=({outside_logs['v1_outside_frac']:.3f},{outside_logs['v2_outside_frac']:.3f}) "
                            f"wb=({wb_logs['v1_warpback_p95_mm']:.2f},{wb_logs['v2_warpback_p95_mm']:.2f})",
                            flush=True,
                        )
                        ok = True
                        break

                if not ok:
                    print(f"[fail] ID={ID:03d} amp={amp:g} k={int(k):02d} exhausted max_tries={args.max_tries} rejects={rejects}", flush=True)

        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
