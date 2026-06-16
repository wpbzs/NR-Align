#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    deform.py

ONE-FIX dataset generator (128^3 paired masks) using the SAME deformation strategy as:
  nr256_gen_physio_bspline_df_v9_2_gated_truncation_gate.py

Key constraints (as requested):
- Single file; DO NOT import the generator as a module.
- Diffeomorphic-ish deformation via SVF exp (scaling & squaring).
- Physio model: localized translation (resp) + localized torsion (cardiac).
- Gates: detJ (ROI crop), outside(truncation), border_frac, warpback_p95 (composition residual p95 in mm).
- Output structure:
    out_root/IDxxx/ampYY/kZZ/{view1_mask.npy, view2_mask.npy, meta.json}
  view*_mask.npy are XYZ uint8 {0,1}, shape (128,128,128).
- ONE-FIX: default fixed_view=1 => view1=GT (reference), view2=deformed.

Requires:
- numpy
- torch
- scipy (for distance_transform_edt; binary ops have a torch fallback, but SDF needs scipy)
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
    raise RuntimeError("This script requires PyTorch (torch).") from e

# --- SciPy (SDF needs distance_transform_edt) ---
try:
    from scipy.ndimage import distance_transform_edt, binary_dilation as _scipy_binary_dilation, binary_closing as _scipy_binary_closing, label as _scipy_label
    _HAS_SCIPY = True
except Exception:
    distance_transform_edt = None
    _scipy_binary_dilation = None
    _scipy_binary_closing = None
    _scipy_label = None
    _HAS_SCIPY = False


# ---------------------------- small utils ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def parse_ints(s: str) -> List[int]:
    s = str(s).strip()
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(list(range(a, b + step, step)))
        else:
            out.append(int(part))
    seen = set()
    ret = []
    for x in out:
        if x not in seen:
            ret.append(x)
            seen.add(x)
    return ret

def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]

def xyz_to_zyx(a_xyz: np.ndarray) -> np.ndarray:
    return np.transpose(a_xyz, (2, 1, 0))

def zyx_to_xyz(a_zyx: np.ndarray) -> np.ndarray:
    return np.transpose(a_zyx, (2, 1, 0))

def center_of_mass_zyx(mask_zyx: np.ndarray) -> Tuple[float, float, float]:
    idx = np.argwhere(mask_zyx > 0)
    if idx.size == 0:
        D, H, W = mask_zyx.shape
        return (D / 2.0, H / 2.0, W / 2.0)
    z, y, x = idx.mean(axis=0)
    return float(z), float(y), float(x)

def bbox_margins_zyx(mask_zyx: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    """
    Return margins (voxels) from foreground bbox to each face:
      (low_z, high_z, low_y, high_y, low_x, high_x)
    """
    idx = np.argwhere(mask_zyx > 0)
    D, H, W = mask_zyx.shape
    if idx.size == 0:
        return (D // 2, D // 2, H // 2, H // 2, W // 2, W // 2)
    zmin, ymin, xmin = idx.min(axis=0)
    zmax, ymax, xmax = idx.max(axis=0)
    low_z = int(zmin)
    high_z = int((D - 1) - zmax)
    low_y = int(ymin)
    high_y = int((H - 1) - ymax)
    low_x = int(xmin)
    high_x = int((W - 1) - xmax)
    return (low_z, high_z, low_y, high_y, low_x, high_x)

def clamp_translation_vox(
    t_dzdy_dx_vox: Tuple[float, float, float],
    margins_zyx: Tuple[int, int, int, int, int, int],
    safety_vox: int,
) -> Tuple[float, float, float]:
    """
    Clamp global translation (dz,dy,dx) so the 128^3 bbox stays within crop with a safety margin.
    """
    dz, dy, dx = t_dzdy_dx_vox
    low_z, high_z, low_y, high_y, low_x, high_x = margins_zyx

    allow_pos_z = max(0.0, float(high_z - safety_vox))
    allow_neg_z = max(0.0, float(low_z - safety_vox))
    allow_pos_y = max(0.0, float(high_y - safety_vox))
    allow_neg_y = max(0.0, float(low_y - safety_vox))
    allow_pos_x = max(0.0, float(high_x - safety_vox))
    allow_neg_x = max(0.0, float(low_x - safety_vox))

    def clamp_one(t, allow_neg, allow_pos):
        if t >= 0.0:
            return min(t, allow_pos)
        else:
            return max(t, -allow_neg)

    dzc = clamp_one(dz, allow_neg_z, allow_pos_z)
    dyc = clamp_one(dy, allow_neg_y, allow_pos_y)
    dxc = clamp_one(dx, allow_neg_x, allow_pos_x)
    return (float(dzc), float(dyc), float(dxc))

def roi_mean_disp_vox(disp_vox_zyx: np.ndarray, roi_zyx: np.ndarray) -> Tuple[float, float, float]:
    """
    disp_vox_zyx: (3,D,H,W) channels (dz,dy,dx) in vox
    roi_zyx: (D,H,W) {0,1}
    """
    w = (roi_zyx > 0).astype(np.float32)
    s = float(w.sum())
    if s <= 1e-6:
        return (0.0, 0.0, 0.0)
    dz = float((disp_vox_zyx[0] * w).sum() / s)
    dy = float((disp_vox_zyx[1] * w).sum() / s)
    dx = float((disp_vox_zyx[2] * w).sum() / s)
    return (dz, dy, dx)

def dice_bool(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0)
    b = (b > 0)
    ia = int(a.sum())
    ib = int(b.sum())
    if ia + ib == 0:
        return 1.0
    inter = int((a & b).sum())
    return float(2.0 * inter / (ia + ib + 1e-8))

def border_frac_zyx(mask_zyx: np.ndarray, border_width: int = 2) -> Tuple[float, int]:
    """
    Return:
      - border_frac: fraction of foreground voxels that lie in the border shell (width=border_width)
      - touch_border: 1 if any foreground touches the border shell else 0
    """
    m = (mask_zyx > 0)
    fg = int(m.sum())
    if fg == 0:
        return 0.0, 0
    bw = int(max(1, border_width))
    D, H, W = m.shape
    border = np.zeros_like(m, dtype=bool)
    border[:bw, :, :] = True
    border[-bw:, :, :] = True
    border[:, :bw, :] = True
    border[:, -bw:, :] = True
    border[:, :, :bw] = True
    border[:, :, -bw:] = True
    on = m & border
    touch = 1 if bool(on.any()) else 0
    frac = float(on.sum() / float(fg))
    return frac, touch

def _jsonify(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.ndarray,)):
        if v.shape == ():
            return _jsonify(v.item())
        if v.size == 1:
            return _jsonify(v.reshape(-1)[0].item())
        if v.size <= 256:
            return v.tolist()
        return {"_array": True, "shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonify(val) for k, val in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        return str(v)
    except Exception:
        return None

def load_npz_meta(npz_path: Path) -> Dict[str, Any]:
    """
    Load meta npz into a JSON-friendly dict. Also tries to alias common geometry keys to legacy geometry-key names.
    """
    meta: Dict[str, Any] = {}
    with np.load(str(npz_path), allow_pickle=True) as npz:
        for k in npz.files:
            meta[k] = _jsonify(npz[k])

    # Best-effort aliases for keys build_bp scripts often expect
    alias_map = {
        "DSD_view1": ["DSD_view1", "DSD1", "DSD_v1", "DSD_A", "DSD_a"],
        "DSO_view1": ["DSO_view1", "DSO1", "DSO_v1", "DSO_A", "DSO_a"],
        "angles1_deg": ["angles1_deg", "angles1", "angle1_deg", "anglesA_deg", "angles_a_deg", "alpha1_deg"],
        "DSD_view2": ["DSD_view2", "DSD2", "DSD_v2", "DSD_B", "DSD_b"],
        "DSO_view2": ["DSO_view2", "DSO2", "DSO_v2", "DSO_B", "DSO_b"],
        "angles2_proj_deg": ["angles2_proj_deg", "angles2_proj", "angles2_deg", "angles2", "angle2_deg", "anglesB_deg", "angles_b_deg", "alpha2_deg"],
        "angles2_bp_deg": ["angles2_bp_deg", "angles2_bp", "angles2_bpdeg", "angles2_recon_deg"],
        "offOrigin_dirty": ["offOrigin_dirty", "offOrigin", "offOrigin_vox", "offOrigin_mm", "off_origin_dirty"],
        "d_spacing": ["d_spacing", "dDetector", "dDetector_new", "dDetector_mm"],
        "v_size": ["v_size", "nVoxel", "nVoxelXYZ", "volume_size", "vsize"],
    }
    for std, cands in alias_map.items():
        if std in meta:
            continue
        for c in cands:
            if c in meta:
                meta[std] = meta[c]
                meta[f"_alias_{std}"] = c
                break
    return meta


# ---------------------------- morphology / SDF ----------------------------

def _torch_binary_dilation(vol_zyx_u8: np.ndarray, iterations: int) -> np.ndarray:
    """Fallback binary dilation using 3x3x3 conv on CPU."""
    it = int(max(0, iterations))
    if it == 0:
        return (vol_zyx_u8 > 0).astype(np.uint8)

    x = torch.from_numpy((vol_zyx_u8 > 0).astype(np.float32))[None, None]  # 1,1,D,H,W
    k = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32)
    for _ in range(it):
        y = F.conv3d(x, k, padding=1)
        x = (y > 0.5).to(torch.float32)
    return (x[0, 0].numpy() > 0.5).astype(np.uint8)

def binary_dilation(vol_zyx_u8: np.ndarray, iterations: int) -> np.ndarray:
    if _HAS_SCIPY:
        return _scipy_binary_dilation(vol_zyx_u8 > 0, iterations=int(iterations)).astype(np.uint8)
    return _torch_binary_dilation(vol_zyx_u8, iterations)

def binary_closing(vol_zyx_u8: np.ndarray, iterations: int) -> np.ndarray:
    if _HAS_SCIPY:
        return _scipy_binary_closing(vol_zyx_u8 > 0, iterations=int(iterations)).astype(np.uint8)
    x = binary_dilation(vol_zyx_u8, int(iterations))
    inv = (x == 0).astype(np.uint8)
    inv_d = binary_dilation(inv, int(iterations))
    return (inv_d == 0).astype(np.uint8)

def keep_largest_cc(vol_zyx_u8: np.ndarray) -> np.ndarray:
    if not _HAS_SCIPY:
        return (vol_zyx_u8 > 0).astype(np.uint8)
    lab, n = _scipy_label((vol_zyx_u8 > 0).astype(np.uint8))
    if n <= 1:
        return (vol_zyx_u8 > 0).astype(np.uint8)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    k = int(sizes.argmax())
    return (lab == k).astype(np.uint8)

def mask_to_sdf(mask_zyx_u8: np.ndarray) -> np.ndarray:
    if distance_transform_edt is None:
        raise RuntimeError("SciPy is required for distance_transform_edt (SDF). Please install scipy in your environment.")
    m = (mask_zyx_u8 > 0)
    out = distance_transform_edt(~m).astype(np.float32)  # vox
    inn = distance_transform_edt(m).astype(np.float32)   # vox
    return out - inn  # inside negative


# ---------------------------- torch / SVF exp ----------------------------

def make_base_grid_zyx(D: int, H: int, W: int, device):
    """grid_sample grid in normalized XYZ order: (x,y,z). shape (1,D,H,W,3)."""
    z = torch.linspace(-1.0, 1.0, D, device=device, dtype=torch.float32)
    y = torch.linspace(-1.0, 1.0, H, device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, W, device=device, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=-1)[None]  # 1,D,H,W,3
    return grid

def grid_sample(vol, grid, mode="bilinear"):
    return F.grid_sample(vol, grid, mode=mode, padding_mode="border", align_corners=True)

def vox_zyx_to_norm_xyz(disp_vox_zyx, D: int, H: int, W: int):
    """disp_vox_zyx: (N,3,D,H,W) channels (dz,dy,dx) in vox -> norm xyz (dx,dy,dz)."""
    dz = disp_vox_zyx[:, 0:1]
    dy = disp_vox_zyx[:, 1:2]
    dx = disp_vox_zyx[:, 2:3]
    dxn = dx * (2.0 / max(W - 1, 1))
    dyn = dy * (2.0 / max(H - 1, 1))
    dzn = dz * (2.0 / max(D - 1, 1))
    return torch.cat([dxn, dyn, dzn], dim=1)

def norm_xyz_to_vox_zyx(disp_norm_xyz, D: int, H: int, W: int):
    """disp_norm_xyz: (N,3,D,H,W) dxn,dyn,dzn -> vox (dz,dy,dx)."""
    dxn = disp_norm_xyz[:, 0:1]
    dyn = disp_norm_xyz[:, 1:2]
    dzn = disp_norm_xyz[:, 2:3]
    dx = dxn * (max(W - 1, 1) / 2.0)
    dy = dyn * (max(H - 1, 1) / 2.0)
    dz = dzn * (max(D - 1, 1) / 2.0)
    return torch.cat([dz, dy, dx], dim=1)

def compose_disp_norm_xyz(u_xyz, v_xyz, base_grid):
    """u,v: (N,3,D,H,W) in norm xyz; return u∘(Id+v)+v."""
    grid = base_grid + v_xyz.permute(0, 2, 3, 4, 1)
    u_warp = grid_sample(u_xyz, grid, mode="bilinear")
    return u_warp + v_xyz

def svf_exp_scaling_squaring(v_xyz, base_grid, n_steps: int = 6):
    """exp(v) for stationary velocity field (normalized xyz)."""
    u = v_xyz / float(2 ** n_steps)
    for _ in range(int(n_steps)):
        u = compose_disp_norm_xyz(u, u, base_grid)
    return u

def invert_disp_norm_xyz(u_xyz, base_grid, n_iter: int = 8):
    """
    Approximate inverse displacement in normalized xyz using fixed-point iteration:
        inv <- - warp(u, Id + inv)
    """
    inv = -u_xyz.clone()
    for _ in range(int(max(1, n_iter))):
        grid = base_grid + inv.permute(0, 2, 3, 4, 1)
        u_warp = grid_sample(u_xyz, grid, mode="bilinear")
        inv = -u_warp
    return inv

def p95_disp_mm_norm_xyz(disp_norm_xyz_np: np.ndarray, roi_zyx: np.ndarray, voxel_mm: float, full_shape_zyx=None) -> float:
    """
    disp_norm_xyz_np: (3,D,H,W) in normalized xyz (dxn,dyn,dzn), but roi is zyx.
    Convert to vox magnitude then mm.
    """
    dxn, dyn, dzn = disp_norm_xyz_np[0], disp_norm_xyz_np[1], disp_norm_xyz_np[2]
    if full_shape_zyx is None:
        D, H, W = roi_zyx.shape
    else:
        D, H, W = int(full_shape_zyx[0]), int(full_shape_zyx[1]), int(full_shape_zyx[2])
    dx = dxn * (max(W - 1, 1) / 2.0)
    dy = dyn * (max(H - 1, 1) / 2.0)
    dz = dzn * (max(D - 1, 1) / 2.0)
    mag = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
    vals = mag[roi_zyx > 0]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    return float(np.percentile(vals, 95) * float(voxel_mm))

def gaussian_blur3d(vol_zyx_t, sigma: float):
    """Separable Gaussian blur for tensor (1,1,D,H,W) float32."""
    if sigma <= 0:
        return vol_zyx_t
    radius = int(math.ceil(3.0 * float(sigma)))
    xs = torch.arange(-radius, radius + 1, device=vol_zyx_t.device, dtype=torch.float32)
    k = torch.exp(-(xs ** 2) / (2.0 * float(sigma) * float(sigma)))
    k = k / (k.sum() + 1e-8)
    kx = k.view(1, 1, 1, 1, -1)
    ky = k.view(1, 1, 1, -1, 1)
    kz = k.view(1, 1, -1, 1, 1)
    pad = radius
    vol = F.conv3d(vol_zyx_t, kx, padding=(0, 0, pad))
    vol = F.conv3d(vol, ky, padding=(0, pad, 0))
    vol = F.conv3d(vol, kz, padding=(pad, 0, 0))
    return vol


# ---------------------------- divergence-free projection ----------------------------

def project_divergence_free_vox_zyx(v_vox_zyx):
    """
    Helmholtz-Hodge projection onto divergence-free fields using FFT (periodic BC).

    v_vox_zyx: (1,3,D,H,W) channels (dz,dy,dx) in voxel units (float32).
    returns same shape, divergence-free in spectral sense.
    """
    assert v_vox_zyx.dtype == torch.float32
    _, _, D, H, W = v_vox_zyx.shape

    vz = v_vox_zyx[:, 0]  # dz
    vy = v_vox_zyx[:, 1]  # dy
    vx = v_vox_zyx[:, 2]  # dx

    Vx = torch.fft.fftn(vx, dim=(-3, -2, -1))
    Vy = torch.fft.fftn(vy, dim=(-3, -2, -1))
    Vz = torch.fft.fftn(vz, dim=(-3, -2, -1))

    kx = (2.0 * math.pi) * torch.fft.fftfreq(W, d=1.0, device=v_vox_zyx.device, dtype=torch.float32).view(1, 1, 1, W)
    ky = (2.0 * math.pi) * torch.fft.fftfreq(H, d=1.0, device=v_vox_zyx.device, dtype=torch.float32).view(1, 1, H, 1)
    kz = (2.0 * math.pi) * torch.fft.fftfreq(D, d=1.0, device=v_vox_zyx.device, dtype=torch.float32).view(1, D, 1, 1)

    I = torch.complex(torch.tensor(0.0, device=v_vox_zyx.device), torch.tensor(1.0, device=v_vox_zyx.device))
    div_hat = I * (kx * Vx + ky * Vy + kz * Vz)
    denom = (kx * kx + ky * ky + kz * kz).to(torch.complex64)

    denom[..., 0, 0, 0] = torch.complex(torch.tensor(1.0, device=v_vox_zyx.device), torch.tensor(0.0, device=v_vox_zyx.device))
    phi_hat = -div_hat / denom
    phi_hat[..., 0, 0, 0] = 0.0 + 0.0j

    Gx = I * kx * phi_hat
    Gy = I * ky * phi_hat
    Gz = I * kz * phi_hat

    Vx_df = Vx - Gx
    Vy_df = Vy - Gy
    Vz_df = Vz - Gz

    vx_df = torch.fft.ifftn(Vx_df, dim=(-3, -2, -1)).real
    vy_df = torch.fft.ifftn(Vy_df, dim=(-3, -2, -1)).real
    vz_df = torch.fft.ifftn(Vz_df, dim=(-3, -2, -1)).real

    v_df = torch.stack([vz_df, vy_df, vx_df], dim=1)
    return v_df


# ---------------------------- physio SVF basis ----------------------------

def build_window(D: int, H: int, W: int, center_zyx: Tuple[float, float, float], device, sigma_ratio: float):
    cz, cy, cx = center_zyx
    z = (torch.arange(D, device=device, dtype=torch.float32) - float(cz)).view(D, 1, 1)
    y = (torch.arange(H, device=device, dtype=torch.float32) - float(cy)).view(1, H, 1)
    x = (torch.arange(W, device=device, dtype=torch.float32) - float(cx)).view(1, 1, W)
    zz = z.expand(D, H, W)
    yy = y.expand(D, H, W)
    xx = x.expand(D, H, W)
    sig = float(sigma_ratio) * float(W)
    win = torch.exp(-(zz ** 2 + yy ** 2 + xx ** 2) / (2.0 * sig * sig)).clamp(0, 1)
    return win, zz, yy, xx

def build_physio_svf_vox_zyx_v2(
    D: int, H: int, W: int,
    center_zyx: Tuple[float, float, float],
    voxel_mm: float,
    device,
    resp_mm: float,
    resp_lateral_mm: float,
    twist_deg: float,
    twist_mod: float,
    win_sigma_ratio: float,
):
    """
    Return base SVFs in voxel units (1,3,D,H,W) channels (dz,dy,dx), float32.
    """
    win, zz, yy, xx = build_window(D, H, W, center_zyx, device, win_sigma_ratio)

    # Respiration: translation (mostly z)
    resp_vox = float(resp_mm) / float(voxel_mm)
    lat_vox = float(resp_lateral_mm) / float(voxel_mm)
    dz = resp_vox * win
    dy = (0.5 * lat_vox) * win
    dx = (0.5 * lat_vox) * win
    v_resp = torch.stack([dz, dy, dx], dim=0)[None].to(torch.float32)

    # Cardiac: torsion (rigid rotation around z) with SMALL bounded modulation
    omega = (float(twist_deg) * math.pi / 180.0)  # radians
    zprof = 0.85 + 0.15 * torch.tanh(zz / (D / 3.0))
    mod = (1.0 + float(twist_mod) * zprof).clamp(0.7, 1.3)  # bounded
    dy = (-omega) * xx * win * mod
    dx = (omega) * yy * win * mod
    dz = torch.zeros_like(dx)
    v_card = torch.stack([dz, dy, dx], dim=0)[None].to(torch.float32)

    win = win[None, None].to(torch.float32)
    return v_resp, v_card, win


# ---------------------------- audit metrics ----------------------------

def p95_disp_mm(disp_vox_zyx_np: np.ndarray, roi_zyx: np.ndarray, voxel_mm: float) -> float:
    dz, dy, dx = disp_vox_zyx_np
    mag = np.sqrt(dz * dz + dy * dy + dx * dx).astype(np.float32)
    vals = mag[roi_zyx > 0]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    return float(np.percentile(vals, 95) * float(voxel_mm))

def jacobian_det_stats_crop(disp_vox_zyx_crop: np.ndarray, roi_crop: np.ndarray) -> Tuple[float, float]:
    """
    Compute detJ on a central-diff interior and return:
      detJ_min, detJ_nonpos_frac (fraction <=0)
    """
    u = disp_vox_zyx_crop.astype(np.float32)
    dz, dy, dx = u[0], u[1], u[2]

    def cdiff(a, axis):
        return (np.take(a, range(2, a.shape[axis]), axis=axis) - np.take(a, range(0, a.shape[axis] - 2), axis=axis)) * 0.5

    duz_dz = cdiff(dz, 0)[:, 1:-1, 1:-1]
    duz_dy = cdiff(dz, 1)[1:-1, :, 1:-1]
    duz_dx = cdiff(dz, 2)[1:-1, 1:-1, :]

    duy_dz = cdiff(dy, 0)[:, 1:-1, 1:-1]
    duy_dy = cdiff(dy, 1)[1:-1, :, 1:-1]
    duy_dx = cdiff(dy, 2)[1:-1, 1:-1, :]

    dux_dz = cdiff(dx, 0)[:, 1:-1, 1:-1]
    dux_dy = cdiff(dx, 1)[1:-1, :, 1:-1]
    dux_dx = cdiff(dx, 2)[1:-1, 1:-1, :]

    a11 = 1.0 + duz_dz
    a12 = duz_dy
    a13 = duz_dx
    a21 = duy_dz
    a22 = 1.0 + duy_dy
    a23 = duy_dx
    a31 = dux_dz
    a32 = dux_dy
    a33 = 1.0 + dux_dx

    detJ = (
        a11 * (a22 * a33 - a23 * a32) -
        a12 * (a21 * a33 - a23 * a31) +
        a13 * (a21 * a32 - a22 * a31)
    ).astype(np.float32)

    roi_i = roi_crop[1:-1, 1:-1, 1:-1].astype(bool)
    vals = detJ[roi_i]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.min()), float((vals <= 0).mean())

def outside_frac_128_in_256(mask256_zyx_u8: np.ndarray, z0: int, y0: int, x0: int) -> float:
    m = (mask256_zyx_u8 > 0)
    total = int(m.sum())
    if total == 0:
        return 0.0
    inside = int(m[z0:z0 + 128, y0:y0 + 128, x0:x0 + 128].sum())
    outside = total - inside
    return float(outside / max(total, 1))


# ---------------------------- GPU metric helpers (NO strategy change) ----------------------------

def _quantile_linear(vals_1d: torch.Tensor, q: float) -> torch.Tensor:
    """Linear interpolation quantile matching numpy.percentile(..., interpolation='linear')."""
    vals = vals_1d
    n = int(vals.numel())
    if n <= 0:
        return torch.tensor(0.0, device=vals.device, dtype=vals.dtype)
    # sort
    v, _ = torch.sort(vals)
    if n == 1:
        return v[0]
    qq = float(q)
    qq = 0.0 if qq < 0.0 else (1.0 if qq > 1.0 else qq)
    pos = qq * float(n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return v[lo]
    w = float(pos - lo)
    return v[lo] * (1.0 - w) + v[hi] * w

def p95_disp_mm_torch(disp_vox_zyx_t: torch.Tensor, roi_zyx_t: torch.Tensor, voxel_mm: float) -> float:
    """disp_vox_zyx_t: (3,D,H,W) in vox; roi_zyx_t: (D,H,W) bool/int."""
    # magnitude in vox
    mag = torch.sqrt(torch.sum(disp_vox_zyx_t * disp_vox_zyx_t, dim=0))
    vals = mag[roi_zyx_t > 0]
    vals = vals[torch.isfinite(vals)]
    if int(vals.numel()) == 0:
        return 0.0
    q = _quantile_linear(vals.flatten(), 0.95)
    return float(q.item() * float(voxel_mm))

def roi_mean_disp_vox_torch(disp_vox_zyx_t: torch.Tensor, roi_zyx_t: torch.Tensor) -> Tuple[float, float, float]:
    """disp_vox_zyx_t: (3,D,H,W) in vox; roi_zyx_t: (D,H,W) bool/int."""
    w = (roi_zyx_t > 0).to(disp_vox_zyx_t.dtype)
    s = float(w.sum().item())
    if s <= 1e-6:
        return (0.0, 0.0, 0.0)
    dz = float((disp_vox_zyx_t[0] * w).sum().item() / s)
    dy = float((disp_vox_zyx_t[1] * w).sum().item() / s)
    dx = float((disp_vox_zyx_t[2] * w).sum().item() / s)
    return (dz, dy, dx)

def jacobian_det_stats_crop_torch(disp_vox_zyx_crop_t: torch.Tensor, roi_crop_t: torch.Tensor) -> Tuple[float, float]:
    """
    torch version of jacobian_det_stats_crop (same central-diff interior convention).
    disp_vox_zyx_crop_t: (3,D,H,W) in vox (float32)
    roi_crop_t: (D,H,W) uint8/bool
    returns detJ_min, detJ_nonpos_frac on ROI interior.
    """
    dz = disp_vox_zyx_crop_t[0]
    dy = disp_vox_zyx_crop_t[1]
    dx = disp_vox_zyx_crop_t[2]

    # interior central differences -> shape (D-2,H-2,W-2)
    duz_dz = (dz[2:, 1:-1, 1:-1] - dz[:-2, 1:-1, 1:-1]) * 0.5
    duz_dy = (dz[1:-1, 2:, 1:-1] - dz[1:-1, :-2, 1:-1]) * 0.5
    duz_dx = (dz[1:-1, 1:-1, 2:] - dz[1:-1, 1:-1, :-2]) * 0.5

    duy_dz = (dy[2:, 1:-1, 1:-1] - dy[:-2, 1:-1, 1:-1]) * 0.5
    duy_dy = (dy[1:-1, 2:, 1:-1] - dy[1:-1, :-2, 1:-1]) * 0.5
    duy_dx = (dy[1:-1, 1:-1, 2:] - dy[1:-1, 1:-1, :-2]) * 0.5

    dux_dz = (dx[2:, 1:-1, 1:-1] - dx[:-2, 1:-1, 1:-1]) * 0.5
    dux_dy = (dx[1:-1, 2:, 1:-1] - dx[1:-1, :-2, 1:-1]) * 0.5
    dux_dx = (dx[1:-1, 1:-1, 2:] - dx[1:-1, 1:-1, :-2]) * 0.5

    a11 = 1.0 + duz_dz
    a12 = duz_dy
    a13 = duz_dx
    a21 = duy_dz
    a22 = 1.0 + duy_dy
    a23 = duy_dx
    a31 = dux_dz
    a32 = dux_dy
    a33 = 1.0 + dux_dx

    detJ = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)

    roi_i = (roi_crop_t[1:-1, 1:-1, 1:-1] > 0)
    vals = detJ[roi_i]
    vals = vals[torch.isfinite(vals)]
    if int(vals.numel()) == 0:
        return float("nan"), float("nan")
    det_min = float(vals.min().item())
    nonpos = float((vals <= 0).to(torch.float32).mean().item())
    return det_min, nonpos

def p95_disp_mm_normxyz_crop_torch(comp_norm_xyz_crop_t: torch.Tensor, roi_crop_t: torch.Tensor, voxel_mm: float, full_DHW: Tuple[int,int,int]=(256,256,256)) -> float:
    """
    comp_norm_xyz_crop_t: (3,128,128,128) in normalized xyz (dxn,dyn,dzn) but crop in zyx order.
    Convert to vox using FULL 256^3 scaling like the numpy version, then p95 in mm within roi_crop.
    """
    D, H, W = int(full_DHW[0]), int(full_DHW[1]), int(full_DHW[2])
    dxn = comp_norm_xyz_crop_t[0]
    dyn = comp_norm_xyz_crop_t[1]
    dzn = comp_norm_xyz_crop_t[2]
    dx = dxn * (max(W - 1, 1) / 2.0)
    dy = dyn * (max(H - 1, 1) / 2.0)
    dz = dzn * (max(D - 1, 1) / 2.0)
    mag = torch.sqrt(dx * dx + dy * dy + dz * dz)
    vals = mag[roi_crop_t > 0]
    vals = vals[torch.isfinite(vals)]
    if int(vals.numel()) == 0:
        return 0.0
    q = _quantile_linear(vals.flatten(), 0.95)
    return float(q.item() * float(voxel_mm))

# ---------------------------- main ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--ids", type=str, default="1-5")
    ap.add_argument("--ks", type=str, default="0-1")
    ap.add_argument("--amp_list_mm", type=str, default="6,10,14,18")

    ap.add_argument("--gt_root", type=str, required=True)
    ap.add_argument("--meta_npz_dir", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)

    ap.add_argument("--voxel_mm", type=float, default=0.78125)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", type=int, default=0)

    # exp(SVF)
    ap.add_argument("--ss_steps", type=int, default=6)
    ap.add_argument("--use_divfree", type=int, default=1)

    # physio knobs (match v9_2 defaults)
    ap.add_argument("--win_sigma_ratio", type=float, default=0.50)
    ap.add_argument("--resp_base_mm", type=float, default=8.0)
    ap.add_argument("--resp_lateral_mm", type=float, default=2.0)
    ap.add_argument("--card_twist_base_deg", type=float, default=8.0)
    ap.add_argument("--card_twist_mod", type=float, default=0.10)

    # sdf warp -> blur -> threshold (match v9_2 defaults)
    ap.add_argument("--aa_sigma", type=float, default=0.7)
    ap.add_argument("--iso_offset", type=float, default=+0.45)

    # ROI for p95/detJ stats (match v9_2 defaults)
    ap.add_argument("--roi_dilate_big", type=int, default=3)
    ap.add_argument("--roi_dilate_crop", type=int, default=2)

    # translation clamp (match v9_2 defaults)
    ap.add_argument("--clamp_translation", type=int, default=1)
    ap.add_argument("--clamp_safety_vox", type=int, default=8)

    # post threshold connectivity fix (match v9_2 defaults)
    ap.add_argument("--connfix", type=int, default=1)
    ap.add_argument("--connfix_close_iters", type=int, default=1)

    # sampling / progress
    ap.add_argument("--max_tries", type=int, default=2000)
    ap.add_argument("--print_every", type=int, default=50)

    # gates (your strict settings as defaults)
    ap.add_argument("--gate_detJ", type=int, default=1)
    ap.add_argument("--detJ_min_thr", type=float, default=0.0)
    ap.add_argument("--detJ_nonpos_max", type=float, default=0.0)

    ap.add_argument("--gate_border", type=int, default=1)
    ap.add_argument("--border_width", type=int, default=1)
    ap.add_argument("--border_frac_max", type=float, default=0.0)

    ap.add_argument("--gate_outside", type=int, default=1)
    ap.add_argument("--outside_frac_max", type=float, default=0.0)

    ap.add_argument("--gate_warpback", type=int, default=1)
    ap.add_argument("--warpback_p95_mm_max", type=float, default=2.0)
    ap.add_argument("--warpback_inv_mode", type=str, default="svf", choices=["svf", "fixedpoint"])
    ap.add_argument("--inv_iters", type=int, default=12)

    ap.add_argument("--warpback_use_dice_gate", type=int, default=0)
    ap.add_argument("--warpback_dice_min", type=float, default=0.985)

    # one-fix
    ap.add_argument("--fixed_view", type=int, default=1, choices=[1, 2])

    args = ap.parse_args()

    if not _HAS_SCIPY:
        raise RuntimeError("SciPy not found. Install scipy (required for SDF via distance_transform_edt).")

    device = torch.device(args.device)
    ids = parse_ints(args.ids)
    ks = parse_ints(args.ks)
    amps = parse_floats(args.amp_list_mm)

    gt_root = Path(args.gt_root)
    meta_dir = Path(args.meta_npz_dir)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    # fixed crop window in 256 (center embed)
    z0 = y0 = x0 = (256 - 128) // 2
    base_grid_256 = make_base_grid_zyx(256, 256, 256, device=device)

    torch.manual_seed(int(args.seed))

    for ID in ids:
        gt_path = gt_root / f"{ID}.npy"
        npz_path = meta_dir / f"{ID}.npz"
        if not gt_path.exists():
            print(f"[skip] missing GT: {gt_path}", flush=True)
            continue
        if not npz_path.exists():
            print(f"[skip] missing meta npz: {npz_path}", flush=True)
            continue

        meta_geo = load_npz_meta(npz_path)

        gt_xyz = np.load(str(gt_path)).astype(np.uint8)
        if gt_xyz.shape != (128, 128, 128):
            raise ValueError(f"GT must be 128^3 XYZ; got {gt_xyz.shape} at {gt_path}")

        gt_zyx = xyz_to_zyx(gt_xyz)
        if int((gt_zyx > 0).sum()) == 0:
            print(f"[skip] empty GT: {gt_path}", flush=True)
            continue

        # 256^3 reference volume with centered 128^3 crop
        big_ref = np.zeros((256, 256, 256), dtype=np.uint8)
        big_ref[z0:z0 + 128, y0:y0 + 128, x0:x0 + 128] = (gt_zyx > 0).astype(np.uint8)

        # ROI masks for measuring/gating
        roi_big = binary_dilation(big_ref, iterations=int(args.roi_dilate_big)).astype(np.uint8)
        roi_crop = binary_dilation((gt_zyx > 0).astype(np.uint8), iterations=int(args.roi_dilate_crop)).astype(np.uint8)

        # torch ROIs on device (metrics on GPU; strategy unchanged)
        roi_big_t = torch.from_numpy((roi_big > 0).astype(np.uint8)).to(device=device)
        roi_crop_t = torch.from_numpy((roi_crop > 0).astype(np.uint8)).to(device=device)

        margins_128 = bbox_margins_zyx(gt_zyx > 0)

        # SDF ref (CPU -> GPU once per ID)
        sdf_ref = mask_to_sdf(big_ref).astype(np.float32)
        sdf_ref_t = torch.from_numpy(sdf_ref).to(device=device, dtype=torch.float32)[None, None]

        # Physio bases once per ID
        com = center_of_mass_zyx(big_ref)
        with torch.inference_mode():
            v_resp0, v_card0, win = build_physio_svf_vox_zyx_v2(
                256, 256, 256, com,
                voxel_mm=float(args.voxel_mm),
                device=device,
                resp_mm=float(args.resp_base_mm),
                resp_lateral_mm=float(args.resp_lateral_mm),
                twist_deg=float(args.card_twist_base_deg),
                twist_mod=float(args.card_twist_mod),
                win_sigma_ratio=float(args.win_sigma_ratio),
            )
            # zero-mean within window to remove rigid drift
            w = (win / (win.sum() + 1e-6)).to(torch.float32)
            for vv in (v_resp0, v_card0):
                mean = (vv * w).sum(dim=(2, 3, 4), keepdim=True)
                vv -= mean

        for amp in amps:
            for k in ks:
                case_dir = out_root / f"ID{ID:03d}" / f"amp{int(round(amp))}" / f"k{int(k):02d}"
                meta_path = case_dir / "meta.json"
                if meta_path.exists() and (not int(args.overwrite)):
                    continue
                ensure_dir(case_dir)

                seed_case = int(args.seed) + int(ID) * 100000 + int(round(amp)) * 100 + int(k)
                rng = np.random.RandomState(seed_case)

                rejects = {"nan": 0, "empty": 0, "detJ": 0, "border": 0, "warpback": 0, "outside": 0}
                ok = False

                with torch.inference_mode():
                    for attempt in range(1, int(args.max_tries) + 1):
                        t_resp = float(rng.rand())
                        t_card = float(rng.rand())
                        a_resp = math.sin(2.0 * math.pi * t_resp)
                        a_card = math.sin(2.0 * math.pi * t_card + math.pi / 2.0)

                        v_vox = (a_resp * v_resp0 + a_card * v_card0).to(torch.float32)
                        if int(args.use_divfree) == 1:
                            v_vox = project_divergence_free_vox_zyx(v_vox)

                        # scale so p95(||u||) matches amp (mm) in big ROI
                        scale = 1.0
                        p95_big_mm = None
                        u_norm_xyz = None
                        v_norm_xyz = None
                        disp_vox_t = None

                        # NOTE: same 2-pass scaling as original, but p95 is computed ON GPU to avoid CPU copies.
                        for _ in range(2):
                            v_scaled = v_vox * float(scale)
                            v_norm_xyz = vox_zyx_to_norm_xyz(v_scaled, 256, 256, 256).to(torch.float32)
                            u_norm_xyz = svf_exp_scaling_squaring(v_norm_xyz, base_grid_256, n_steps=int(args.ss_steps)).to(torch.float32)
                            disp_vox_t = norm_xyz_to_vox_zyx(u_norm_xyz, 256, 256, 256)[0]  # (3,256,256,256) vox, zyx
                            if (disp_vox_t is None) or (not torch.isfinite(disp_vox_t).all()):
                                u_norm_xyz = None
                                break
                            p95_big_mm = p95_disp_mm_torch(disp_vox_t, roi_big_t, float(args.voxel_mm))
                            if (not np.isfinite(p95_big_mm)) or (p95_big_mm < 1e-4):
                                u_norm_xyz = None
                                break
                            scale *= float(amp) / float(p95_big_mm)

                        if (u_norm_xyz is None) or (v_norm_xyz is None) or (disp_vox_t is None) or (p95_big_mm is None):
                            rejects["nan"] += 1
                            if attempt % int(args.print_every) == 0:
                                print(f"[hb] ID={ID:03d} amp={amp:g} k={int(k):02d} attempt={attempt}/{int(args.max_tries)} rejects={rejects}", flush=True)
                            continue

# Translation clamp
                        dxn = dyn = dzn = 0.0
                        t_mean = (0.0, 0.0, 0.0)
                        t_clamped = (0.0, 0.0, 0.0)
                        delta = (0.0, 0.0, 0.0)
                        if int(args.clamp_translation) == 1:
                            disp_crop_pre_t = disp_vox_t[:, z0:z0 + 128, y0:y0 + 128, x0:x0 + 128]
                            t_mean = roi_mean_disp_vox_torch(disp_crop_pre_t, roi_crop_t)
                            t_clamped = clamp_translation_vox(t_mean, margins_128, int(args.clamp_safety_vox))
                            delta = (t_clamped[0] - t_mean[0], t_clamped[1] - t_mean[1], t_clamped[2] - t_mean[2])

                            dzn = float(delta[0]) * (2.0 / (256.0 - 1.0))
                            dyn = float(delta[1]) * (2.0 / (256.0 - 1.0))
                            dxn = float(delta[2]) * (2.0 / (256.0 - 1.0))

                            # apply translation delta to forward displacement (same as original)
                            u_norm_xyz[:, 0:1] = u_norm_xyz[:, 0:1] + dxn
                            u_norm_xyz[:, 1:2] = u_norm_xyz[:, 1:2] + dyn
                            u_norm_xyz[:, 2:3] = u_norm_xyz[:, 2:3] + dzn

                            # keep disp_vox_t consistent for downstream metrics
                            disp_vox_t[0] = disp_vox_t[0] + float(delta[0])
                            disp_vox_t[1] = disp_vox_t[1] + float(delta[1])
                            disp_vox_t[2] = disp_vox_t[2] + float(delta[2])

# detJ gate
                        disp_crop_t = disp_vox_t[:, z0:z0 + 128, y0:y0 + 128, x0:x0 + 128]  # (3,128,128,128)
                        p95_crop_mm = p95_disp_mm_torch(disp_crop_t, roi_crop_t, float(args.voxel_mm))
                        detJ_min, detJ_nonpos = jacobian_det_stats_crop_torch(disp_crop_t, roi_crop_t)
                        if int(args.gate_detJ) == 1:
                            if (not np.isfinite(detJ_min)) or (not np.isfinite(detJ_nonpos)):
                                rejects["detJ"] += 1
                                if attempt % int(args.print_every) == 0:
                                    print(f"[hb] ID={ID:03d} amp={amp:g} k={int(k):02d} attempt={attempt}/{int(args.max_tries)} rejects={rejects}", flush=True)
                                continue
                            if (float(detJ_min) < float(args.detJ_min_thr) - 1e-12) or (float(detJ_nonpos) > float(args.detJ_nonpos_max) + 1e-12):
                                rejects["detJ"] += 1
                                if attempt % int(args.print_every) == 0:
                                    print(f"[hb] ID={ID:03d} amp={amp:g} k={int(k):02d} attempt={attempt}/{int(args.max_tries)} rejects={rejects}", flush=True)
                                continue

                        # warp SDF -> blur -> threshold
                        grid_fwd = base_grid_256 + u_norm_xyz.permute(0, 2, 3, 4, 1)
                        sdf_w = grid_sample(sdf_ref_t, grid_fwd, mode="bilinear")
                        sdf_w = gaussian_blur3d(sdf_w, float(args.aa_sigma))
                        # threshold on GPU (same iso rule), then transfer uint8 to CPU (faster than float32 copy)
                        big_def_t = (sdf_w <= float(args.iso_offset)).to(torch.uint8)
                        big_def = big_def_t[0, 0].detach().cpu().numpy()
                        if int(big_def.sum()) == 0:
                            rejects["empty"] += 1
                            if attempt % int(args.print_every) == 0:
                                print(f"[hb] ID={ID:03d} amp={amp:g} k={int(k):02d} attempt={attempt}/{int(args.max_tries)} rejects={rejects}", flush=True)
                            continue

                        # outside gate
                        outside_frac = outside_frac_128_in_256(big_def, z0, y0, x0)
                        if int(args.gate_outside) == 1 and outside_frac > float(args.outside_frac_max) + 1e-12:
                            rejects["outside"] += 1
                            if attempt % int(args.print_every) == 0:
                                print(f"[hb] ID={ID:03d} amp={amp:g} k={int(k):02d} attempt={attempt}/{int(args.max_tries)} rejects={rejects}", flush=True)
                            continue

                        def128 = big_def[z0:z0 + 128, y0:y0 + 128, x0:x0 + 128].astype(np.uint8)

                        # connectivity fix (v9_2)
                        if int(args.connfix) == 1:
                            def128 = binary_closing(def128, iterations=int(args.connfix_close_iters)).astype(np.uint8)
                            def128 = keep_largest_cc(def128)

                        if int(def128.sum()) == 0:
                            rejects["empty"] += 1
                            if attempt % int(args.print_every) == 0:
                                print(f"[hb] ID={ID:03d} amp={amp:g} k={int(k):02d} attempt={attempt}/{int(args.max_tries)} rejects={rejects}", flush=True)
                            continue

                        # border gate
                        bfrac, _ = border_frac_zyx(def128, border_width=int(args.border_width))
                        if int(args.gate_border) == 1 and float(bfrac) > float(args.border_frac_max) + 1e-12:
                            rejects["border"] += 1
                            if attempt % int(args.print_every) == 0:
                                print(f"[hb] ID={ID:03d} amp={amp:g} k={int(k):02d} attempt={attempt}/{int(args.max_tries)} rejects={rejects}", flush=True)
                            continue

                        # warpback gate
                        warpback_p95_mm = float("nan")
                        warpback_dice = float("nan")
                        if int(args.gate_warpback) == 1:
                            if str(args.warpback_inv_mode).lower() == "svf":
                                u0_inv_norm_xyz = svf_exp_scaling_squaring((-v_norm_xyz).to(torch.float32), base_grid_256, n_steps=int(args.ss_steps)).to(torch.float32)
                                if int(args.clamp_translation) == 1:
                                    v_trans_inv = torch.tensor([-dxn, -dyn, -dzn], device=device, dtype=torch.float32).view(1, 3, 1, 1, 1)
                                    u_inv_norm_xyz = compose_disp_norm_xyz(u0_inv_norm_xyz, v_trans_inv, base_grid_256).to(torch.float32)
                                else:
                                    u_inv_norm_xyz = u0_inv_norm_xyz
                            else:
                                u_inv_norm_xyz = invert_disp_norm_xyz(u_norm_xyz, base_grid_256, n_iter=int(args.inv_iters)).to(torch.float32)

                            comp1 = compose_disp_norm_xyz(u_norm_xyz, u_inv_norm_xyz, base_grid_256)
                            comp2 = compose_disp_norm_xyz(u_inv_norm_xyz, u_norm_xyz, base_grid_256)

                            # compute warpback residual p95 on GPU (same definition; norm->vox via FULL 256 scaling)
                            comp1_crop_t = comp1[0, :, z0:z0 + 128, y0:y0 + 128, x0:x0 + 128]  # (3,128,128,128) norm xyz
                            comp2_crop_t = comp2[0, :, z0:z0 + 128, y0:y0 + 128, x0:x0 + 128]
                            wb1 = p95_disp_mm_normxyz_crop_torch(comp1_crop_t, roi_crop_t, float(args.voxel_mm), full_DHW=(256, 256, 256))
                            wb2 = p95_disp_mm_normxyz_crop_torch(comp2_crop_t, roi_crop_t, float(args.voxel_mm), full_DHW=(256, 256, 256))
                            warpback_p95_mm = float(max(wb1, wb2))

                            if (not np.isfinite(warpback_p95_mm)) or (warpback_p95_mm > float(args.warpback_p95_mm_max) + 1e-12):
                                rejects["warpback"] += 1
                                if attempt % int(args.print_every) == 0:
                                    print(f"[hb] ID={ID:03d} amp={amp:g} k={int(k):02d} attempt={attempt}/{int(args.max_tries)} rejects={rejects}", flush=True)
                                continue

                            if int(args.warpback_use_dice_gate) == 1:
                                big_def_t = torch.zeros((256, 256, 256), device=device, dtype=torch.float32)
                                big_def_t[z0:z0 + 128, y0:y0 + 128, x0:x0 + 128] = torch.from_numpy(def128.astype(np.float32)).to(device=device)
                                grid_inv = base_grid_256 + u_inv_norm_xyz.permute(0, 2, 3, 4, 1)
                                back_t = F.grid_sample(big_def_t[None, None], grid_inv, mode="nearest", padding_mode="border", align_corners=True)[0, 0]
                                back = (back_t.detach().cpu().numpy()[z0:z0 + 128, y0:y0 + 128, x0:x0 + 128] > 0.5).astype(np.uint8)
                                warpback_dice = dice_bool(back, gt_zyx)
                                if (not np.isfinite(warpback_dice)) or (warpback_dice < float(args.warpback_dice_min) - 1e-12):
                                    rejects["warpback"] += 1
                                    if attempt % int(args.print_every) == 0:
                                        print(f"[hb] ID={ID:03d} amp={amp:g} k={int(k):02d} attempt={attempt}/{int(args.max_tries)} rejects={rejects}", flush=True)
                                    continue

                        # SUCCESS -> ONE-FIX save
                        if int(args.fixed_view) == 1:
                            view1_zyx = (gt_zyx > 0).astype(np.uint8)
                            view2_zyx = (def128 > 0).astype(np.uint8)
                        else:
                            view2_zyx = (gt_zyx > 0).astype(np.uint8)
                            view1_zyx = (def128 > 0).astype(np.uint8)

                        np.save(str(case_dir / "view1_mask.npy"), zyx_to_xyz(view1_zyx).astype(np.uint8))
                        np.save(str(case_dir / "view2_mask.npy"), zyx_to_xyz(view2_zyx).astype(np.uint8))

                        meta_case: Dict[str, Any] = {}
                        meta_case.update(meta_geo)

                        # required audit fields
                        meta_case.update({
                            "amp_target": float(amp),
                            "p95_crop_mm": float(p95_crop_mm),
                            "detJ_min": float(detJ_min),
                            "border_frac": float(bfrac),
                            "outside_frac": float(outside_frac),
                            "warpback_p95_mm": float(warpback_p95_mm) if np.isfinite(warpback_p95_mm) else float("nan"),
                            "seed_case": int(seed_case),
                            "attempt": int(attempt),
                        })

                        meta_case.update({
                            "id": int(ID),
                            "ID": int(ID),
                            "k": int(k),
                            "fixed_view": int(args.fixed_view),
                            "npz_path": str(npz_path),
                            "gt_path": str(gt_path),
                            "t_resp": float(t_resp),
                            "t_card": float(t_card),
                            "p95_big_mm": float(p95_big_mm) if p95_big_mm is not None else float("nan"),
                            "detJ_nonpos_frac": float(detJ_nonpos),
                            "clamp_t_mean_vox": [float(x) for x in t_mean],
                            "clamp_t_clamped_vox": [float(x) for x in t_clamped],
                            "clamp_delta_vox": [float(x) for x in delta],
                            "ss_steps": int(args.ss_steps),
                            "use_divfree": int(args.use_divfree),
                            "win_sigma_ratio": float(args.win_sigma_ratio),
                            "resp_base_mm": float(args.resp_base_mm),
                            "resp_lateral_mm": float(args.resp_lateral_mm),
                            "card_twist_base_deg": float(args.card_twist_base_deg),
                            "card_twist_mod": float(args.card_twist_mod),
                            "aa_sigma": float(args.aa_sigma),
                            "iso_offset": float(args.iso_offset),
                            "roi_dilate_big": int(args.roi_dilate_big),
                            "roi_dilate_crop": int(args.roi_dilate_crop),
                            "gate_detJ": int(args.gate_detJ),
                            "gate_border": int(args.gate_border),
                            "gate_outside": int(args.gate_outside),
                            "gate_warpback": int(args.gate_warpback),
                            "warpback_inv_mode": str(args.warpback_inv_mode),
                            "warpback_use_dice_gate": int(args.warpback_use_dice_gate),
                            "rejects": rejects,
                        })

                        with open(str(meta_path), "w", encoding="utf-8") as f:
                            json.dump(meta_case, f, indent=2, ensure_ascii=False)

                        print(
                            f"[ok] ID={ID:03d} amp={amp:g} k={int(k):02d} attempt={attempt} "
                            f"p95={p95_crop_mm:.3f}mm detJmin={detJ_min:.4f} bfrac={bfrac:.3e} "
                            f"outside={outside_frac:.3e} wbP95={warpback_p95_mm:.3f}mm",
                            flush=True
                        )
                        ok = True
                        break

                if not ok:
                    print(f"[fail] ID={ID:03d} amp={amp:g} k={int(k):02d} exhausted max_tries={int(args.max_tries)} rejects={rejects}", flush=True)

        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
