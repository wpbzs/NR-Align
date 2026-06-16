#!/usr/bin/env python3
"""Small geometry and parsing helpers for the BP construction script.

This file is intentionally lightweight. It does not contain the original
prior-reconstruction model; it only builds TIGRE geometry objects and normalizes
metadata keys used by the synthetic training pipeline.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def ensure_dir(p: Path | str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def parse_ids(s: str) -> List[int]:
    """Parse comma-separated integers and ranges, e.g. '1-3,8'."""
    s = str(s).strip()
    if s == "" or s.lower() == "none":
        return []
    out: List[int] = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
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
    return [float(x.strip()) for x in str(s).split(',') if x.strip()]


def amp_tag(a: float) -> str:
    """Folder tag used by the data generator: 6 -> amp6, 6.5 -> amp6p5."""
    a = float(a)
    if abs(a - round(a)) < 1e-6:
        return f"amp{int(round(a))}"
    return f"amp{a:.1f}".replace('.', 'p')


def _jsonify(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return _jsonify(v.item())
        if v.size == 1:
            return _jsonify(v.reshape(-1)[0].item())
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonify(val) for k, val in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def load_json(path: Path | str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_npz_meta(npz_path: Path | str) -> Dict[str, Any]:
    """Load a geometry .npz file and add aliases used by the public scripts."""
    meta: Dict[str, Any] = {}
    with np.load(str(npz_path), allow_pickle=True) as npz:
        for k in npz.files:
            meta[k] = _jsonify(npz[k])

    alias_map = {
        "DSD1": ["DSD1", "DSD_view1", "DSD_v1", "DSD_A", "DSD_a", "DSD"],
        "DSO1": ["DSO1", "DSO_view1", "DSO_v1", "DSO_A", "DSO_a", "DSO"],
        "ang1_deg": ["ang1_deg", "angles1_deg", "angles1", "angle1_deg", "anglesA_deg", "alpha1_deg"],
        "DSD2": ["DSD2", "DSD_view2", "DSD_v2", "DSD_B", "DSD_b"],
        "DSO2": ["DSO2", "DSO_view2", "DSO_v2", "DSO_B", "DSO_b"],
        "ang2_bp_deg": ["ang2_bp_deg", "angles2_bp_deg", "angles2_bp", "angles2_deg", "angle2_deg", "anglesB_deg"],
        "off2": ["off2", "offOrigin_dirty", "offOrigin", "offOrigin_mm", "off_origin_dirty"],
        "det_spacing": ["det_spacing", "d_spacing", "dDetector", "dDetector_mm"],
        "v_size": ["v_size", "v_size_mm", "nVoxel_mm", "volume_size", "vsize"],
    }
    for std, candidates in alias_map.items():
        if std in meta:
            continue
        for c in candidates:
            if c in meta:
                meta[std] = meta[c]
                meta[f"_alias_{std}"] = c
                break
    return meta


def _as_float_array(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape == ():
        arr = arr.reshape(1)
    return arr.reshape(-1)


def ang_rad(angles_deg: Any) -> np.ndarray:
    """Convert one angle or a list/array of angles in degrees to TIGRE radians."""
    return _as_float_array(angles_deg) * (math.pi / 180.0)


def make_geo(
    DSD: float,
    DSO: float,
    det_spacing: float,
    v_size: float,
    nDetector: int = 512,
    nVoxel: int = 128,
    offOrigin_mm: Sequence[float] | None = None,
):
    """Build a minimal cone-beam TIGRE geometry for 128^3 vessel volumes.

    The public pipeline stores volumes as XYZ arrays. TIGRE also expects its
    geometry vectors in XYZ-like order, so we keep isotropic settings here.
    If your private prior uses a different geometry convention, adapt this
    function rather than the training code.
    """
    try:
        import tigre
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "TIGRE is required for BP construction. Install TIGRE and make sure "
            "it is importable in this Python environment."
        ) from exc

    geo = tigre.geometry(mode='cone')
    geo.DSD = float(DSD)
    geo.DSO = float(DSO)

    geo.nVoxel = np.array([int(nVoxel), int(nVoxel), int(nVoxel)], dtype=np.int32)
    geo.sVoxel = np.array([float(v_size), float(v_size), float(v_size)], dtype=np.float32)
    geo.dVoxel = geo.sVoxel / geo.nVoxel.astype(np.float32)

    geo.nDetector = np.array([int(nDetector), int(nDetector)], dtype=np.int32)
    geo.dDetector = np.array([float(det_spacing), float(det_spacing)], dtype=np.float32)
    geo.sDetector = geo.nDetector.astype(np.float32) * geo.dDetector

    if offOrigin_mm is None:
        offOrigin_mm = (0.0, 0.0, 0.0)
    geo.offOrigin = np.asarray(offOrigin_mm, dtype=np.float32).reshape(3)
    geo.offDetector = np.array([0.0, 0.0], dtype=np.float32)
    geo.accuracy = 0.5
    return geo
