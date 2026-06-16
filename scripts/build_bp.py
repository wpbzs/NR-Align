#!/usr/bin/env python3
"""Build backprojection volumes for NR-Align training.

This script reads generated two-view masks and writes backprojected
union/overlap volumes for alignment-network training.

Expected input layout:
  data_root/ID{ID:03d}/ampYY/kZZ/{meta.json, view1_mask.npy, view2_mask.npy}

Outputs (saved INSIDE each case directory; no external cache/symlink):
  bp1_gt.npy, bp2_gt.npy, bp012_gt.npy
  bp1_nr.npy, bp2_nr.npy, bp012_nr.npy
Optionally:
  proj1_*.npy, proj2_*.npy

Key points:
- No persistent GT cache directory is used.
- For speed, we still reuse GT-per-ID results in RAM (NOT on disk).
- We DO NOT pass `gpuids` into TIGRE to avoid version-specific type issues.
  Pin the GPU via CUDA_VISIBLE_DEVICES.

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import imageio.v2 as imageio

from geometry import (
    amp_tag,
    ang_rad,
    ensure_dir,
    load_json,
    load_npz_meta,
    make_geo,
    parse_floats,
    parse_ids,
)


def _get(meta: Dict[str, Any], keys: List[str], *, required: bool = True, default=None):
    for k in keys:
        if k in meta:
            return meta[k]
    if required:
        raise KeyError(f"Missing required meta keys (tried {keys})")
    return default


def _tigre_forward_proj_bin(vol_xyz_u8: np.ndarray, geo, ang: np.ndarray) -> np.ndarray:
    """Binary forward projection using TIGRE Ax.

    ang: np.ndarray shape (nAngles,) or (nAngles,1). In our pipeline nAngles=1.
    Returns uint8 2D proj (nDet, nDet).
    """
    import tigre

    proj = tigre.Ax(vol_xyz_u8.astype(np.float32), geo, ang)
    # TIGRE may return shape (nAngles, nDet, nDet). For nAngles==1, take [0].
    if proj.ndim == 3:
        proj2d = proj[0]
    else:
        proj2d = proj
    return (proj2d > 0).astype(np.uint8)


def _tigre_sirt_recon_bin_pos(proj_u8_2d: np.ndarray, geo, ang: np.ndarray, iters: int) -> np.ndarray:
    """SIRT recon using TIGRE, threshold to positive binary volume."""
    import tigre.algorithms as algs

    # TIGRE expects (nAngles, nDet, nDet). Here nAngles=1.
    proj = proj_u8_2d.astype(np.float32)[None, ...]
    vol = algs.sirt(proj, geo, ang, int(iters))
    return (vol > 0).astype(np.uint8)


def find_case_dirs(data_root: Path, ids: List[int], amps: List[float], ks: List[int], id_pad: int) -> List[Path]:
    out: List[Path] = []
    for ID in ids:
        idname = f"ID{ID:0{id_pad}d}" if id_pad > 0 else f"ID{ID}"
        for amp in amps:
            atag = amp_tag(float(amp))
            for k in ks:
                d = data_root / idname / atag / f"k{k:02d}"
                if (d / "meta.json").exists():
                    out.append(d)
    return out


def group_by_id(case_dirs: List[Path]) -> Dict[int, List[Path]]:
    out: Dict[int, List[Path]] = {}
    for d in case_dirs:
        # .../IDxxx/ampYY/kZZ
        id_str = d.parent.parent.name.replace("ID", "")
        ID = int(id_str)
        out.setdefault(ID, []).append(d)
    for ID in out:
        out[ID] = sorted(out[ID])
    return out


def _load_geo_for_id(case_dir_any: Path, *, npz_dir: Optional[Path], nDetector: int) -> Tuple[Dict[str, Any], Any, Any, np.ndarray, np.ndarray]:
    """Returns (meta_full, geo1, geo2, ang1, ang2)."""
    meta = load_json(case_dir_any / "meta.json")

    need = [
        "det_spacing",
        "v_size",
        "DSD1",
        "DSO1",
        "DSD2",
        "DSO2",
        "off2",
        "ang1_deg",
        "ang2_bp_deg",
    ]
    if not all(k in meta for k in need):
        if npz_dir is None:
            raise RuntimeError(f"meta.json missing geo keys and meta_npz_dir not provided: {case_dir_any}")
        ID = int(_get(meta, ["id", "ID"]))
        geo_meta = load_npz_meta(npz_dir / f"{ID}.npz")
        for k, v in geo_meta.items():
            meta.setdefault(k, v)

    det_spacing = float(_get(meta, ["det_spacing", "d_spacing", "dDetector"]))
    v_size = float(_get(meta, ["v_size", "v_size_mm", "nVoxel_mm"], required=False, default=100.0))

    geo1 = make_geo(
        DSD=float(_get(meta, ["DSD1", "DSD_view1", "DSD"])),
        DSO=float(_get(meta, ["DSO1", "DSO_view1", "DSO"])),
        det_spacing=det_spacing,
        v_size=v_size,
        nDetector=int(nDetector),
    )

    off2 = _get(
        meta,
        ["off2", "offOrigin_dirty", "offOrigin_mm_view2", "offOrigin_mm"],
        required=False,
        default=(0.0, 0.0, 0.0),
    )
    off2 = tuple(float(x) for x in off2)

    geo2 = make_geo(
        DSD=float(_get(meta, ["DSD2", "DSD_view2"])),
        DSO=float(_get(meta, ["DSO2", "DSO_view2"])),
        det_spacing=det_spacing,
        v_size=v_size,
        offOrigin_mm=off2,
        nDetector=int(nDetector),
    )

    ang1_deg = _get(meta, ["ang1_deg", "angles1_deg", "angles_view1_deg"])
    ang2_deg = _get(meta, ["ang2_bp_deg", "angles2_bp_deg", "angles2_deg", "angles_view2_deg"])
    ang1 = ang_rad(ang1_deg)
    ang2 = ang_rad(ang2_deg)

    return meta, geo1, geo2, ang1, ang2



def _save_png2d(arr, out_path: str) -> None:
    """Save a 2D projection array as an 8-bit PNG with robust normalization."""
    try:
        import torch  # optional
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().float().cpu().numpy()
    except Exception:
        pass
    a = np.asarray(arr)
    if a.ndim > 2:
        a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"proj must be 2D after squeeze, got shape={a.shape}")
    if a.dtype == np.bool_:
        u8 = a.astype(np.uint8) * 255
    else:
        a = a.astype(np.float32)
        lo, hi = np.percentile(a, (1.0, 99.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(a)), float(np.max(a))
        if hi <= lo:
            u8 = np.zeros_like(a, dtype=np.uint8)
        else:
            a = np.clip(a, lo, hi)
            u8 = ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)
    imageio.imwrite(out_path, u8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--gt_dir", type=str, required=True)
    ap.add_argument("--meta_npz_dir", type=str, default="", help="Optional geo npz dir (datasets/sim_out/meta)")
    ap.add_argument("--ids", type=str, required=True, help="e.g. 1-879 or 1,2,3")
    ap.add_argument("--amps", type=str, required=True, help="e.g. 6,10,14,18")
    ap.add_argument("--ks", type=str, default="0-1")
    ap.add_argument("--id_pad", type=int, default=3)
    ap.add_argument("--nDetector", type=int, default=512)
    ap.add_argument("--sirt_iters", type=int, default=1)
    ap.add_argument("--save_proj", type=int, default=0)
    ap.add_argument("--overwrite", type=int, default=0)
    ap.add_argument("--strict_pre_masks", type=int, default=1)
    ap.add_argument("--only_nr", type=int, default=0)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    gt_dir = Path(args.gt_dir)
    npz_dir = Path(args.meta_npz_dir) if str(args.meta_npz_dir).strip() else None

    ids = parse_ids(args.ids)
    amps = parse_floats(args.amps)
    ks = parse_ids(args.ks)

    case_dirs = find_case_dirs(data_root, ids, amps, ks, int(args.id_pad))
    print(f"found {len(case_dirs)} cases under {data_root}")
    by_id = group_by_id(case_dirs)
    print(f"grouped into {len(by_id)} IDs")

    # In-memory reuse per ID (not a disk cache)
    gt_bp_mem: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    n_ok = n_skip = n_bad = 0

    for ID, dirs in sorted(by_id.items()):
        try:
            meta_any, geo1, geo2, ang1, ang2 = _load_geo_for_id(dirs[0], npz_dir=npz_dir, nDetector=int(args.nDetector))

            gt_path = gt_dir / f"{ID}.npy"
            if not gt_path.exists():
                print(f"[skip] missing GT {gt_path}")
                n_skip += len(dirs)
                continue
            gt_xyz = np.load(str(gt_path)).astype(np.uint8)

            # compute GT BP once per ID (RAM)
            bp1_gt = bp2_gt = bp012_gt = None
            if int(args.only_nr) == 0:
                if ID in gt_bp_mem:
                    bp1_gt, bp2_gt, bp012_gt = gt_bp_mem[ID]
                else:
                    proj1_gt = _tigre_forward_proj_bin(gt_xyz, geo1, ang1)
                    proj2_gt = _tigre_forward_proj_bin(gt_xyz, geo2, ang2)
                    bp1_gt = _tigre_sirt_recon_bin_pos(proj1_gt, geo1, ang1, int(args.sirt_iters)).astype(np.uint8)
                    bp2_gt = _tigre_sirt_recon_bin_pos(proj2_gt, geo2, ang2, int(args.sirt_iters)).astype(np.uint8)
                    bp012_gt = (bp1_gt.astype(np.int8) + bp2_gt.astype(np.int8)).astype(np.int8)
                    gt_bp_mem[ID] = (bp1_gt, bp2_gt, bp012_gt)

            for d in dirs:
                try:
                    pre1 = d / "view1_mask.npy"
                    pre2 = d / "view2_mask.npy"
                    if int(args.strict_pre_masks) == 1 and ((not pre1.exists()) or (not pre2.exists())):
                        n_skip += 1
                        continue

                    out_bp1_gt = d / "bp1_gt.npy"
                    out_bp2_gt = d / "bp2_gt.npy"
                    out_bp012_gt = d / "bp012_gt.npy"
                    out_bp1_nr = d / "bp1_nr.npy"
                    out_bp2_nr = d / "bp2_nr.npy"
                    out_bp012_nr = d / "bp012_nr.npy"

                    if (out_bp1_nr.exists() and out_bp2_nr.exists() and out_bp012_gt.exists()) and int(args.overwrite) == 0:
                        n_skip += 1
                        continue

                    ensure_dir(d)

                    # Write GT volumes into THIS directory (no symlink/caching)
                    if int(args.only_nr) == 0:
                        np.save(str(out_bp1_gt), bp1_gt)
                        np.save(str(out_bp2_gt), bp2_gt)
                        np.save(str(out_bp012_gt), bp012_gt)

                    # load pre masks
                    if pre1.exists() and pre2.exists():
                        m1_xyz = np.load(str(pre1)).astype(np.uint8)
                        m2_xyz = np.load(str(pre2)).astype(np.uint8)
                    else:
                        m1_xyz = gt_xyz
                        m2_xyz = gt_xyz

                    proj1_nr = _tigre_forward_proj_bin(m1_xyz, geo1, ang1)
                    proj2_nr = _tigre_forward_proj_bin(m2_xyz, geo2, ang2)
                    bp1_nr = _tigre_sirt_recon_bin_pos(proj1_nr, geo1, ang1, int(args.sirt_iters)).astype(np.uint8)
                    bp2_nr = _tigre_sirt_recon_bin_pos(proj2_nr, geo2, ang2, int(args.sirt_iters)).astype(np.uint8)
                    bp012_nr = (bp1_nr.astype(np.int8) + bp2_nr.astype(np.int8)).astype(np.int8)

                    np.save(str(out_bp1_nr), bp1_nr)
                    np.save(str(out_bp2_nr), bp2_nr)
                    np.save(str(out_bp012_nr), bp012_nr)

                    if int(args.save_proj) == 1:
                        # Save projections as PNGs (8-bit, robust normalization)
                        _save_png2d(proj1_nr, str(d / "proj1_nr.png"))
                        _save_png2d(proj2_nr, str(d / "proj2_nr.png"))
                        if int(args.only_nr) == 0:
                            # Save GT projections too
                            proj1_gt = _tigre_forward_proj_bin(gt_xyz, geo1, ang1)
                            proj2_gt = _tigre_forward_proj_bin(gt_xyz, geo2, ang2)
                            _save_png2d(proj1_gt, str(d / "proj1_gt.png"))
                            _save_png2d(proj2_gt, str(d / "proj2_gt.png"))

                    n_ok += 1
                    if n_ok % 200 == 0:
                        print(f"[prog] ok={n_ok} skip={n_skip} bad={n_bad}  last=ID{ID:03d} dir={d}")

                except Exception as e_case:
                    print(f"[bad] {d}: {type(e_case).__name__}: {e_case}")
                    n_bad += 1
                    continue

        except Exception as e_id:
            print(f"[bad-ID] ID{ID:03d}: {type(e_id).__name__}: {e_id}")
            n_bad += len(dirs)
            continue

    print(f"[done] ok={n_ok} skip={n_skip} bad={n_bad}")


if __name__ == "__main__":
    main()
