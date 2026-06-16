import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from nr_align_models import build_model


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_ints(s: str) -> List[int]:
    s = str(s).strip()
    if s == "" or s.lower() == "all":
        return []
    out: List[int] = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            a, b = int(a), int(b)
            if a <= b:
                out.extend(range(a, b + 1))
            else:
                out.extend(range(a, b - 1, -1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_amps(s: str) -> List[str]:
    s = str(s).strip()
    if s == '' or s.lower() == 'all':
        return []
    vals = []
    for x in s.split(','):
        x = x.strip()
        if not x:
            continue
        if x.lower().startswith('amp'):
            vals.append(x)
        else:
            vals.append(f'amp{int(float(x))}')
    return sorted(set(vals))


def parse_ks(s: str) -> List[str]:
    s = str(s).strip()
    if s == '' or s.lower() == 'all':
        return []
    return [f'k{k:02d}' for k in parse_ints(s)]


class BPDataset(Dataset):
    """
    root/ID001/amp6/k00/{bp012_nr.npy,bp012_gt.npy}
    also supports ID1 style.
    Returns x_uo=(2,D,H,W), u_gt=(1,D,H,W), o_gt=(1,D,H,W)
    """
    def __init__(self, root: str, ids: str = 'all', amps: str = 'all', ks: str = 'all'):
        self.root = Path(root)
        self.samples: List[Dict] = []

        id_filter = set(parse_ints(ids)) if str(ids).strip().lower() != 'all' else None
        amp_filter = set(parse_amps(amps)) if str(amps).strip().lower() != 'all' else None
        k_filter = set(parse_ks(ks)) if str(ks).strip().lower() != 'all' else None

        for id_dir in sorted(self.root.glob('ID*')):
            if not id_dir.is_dir():
                continue
            m = re.match(r'ID0*([0-9]+)$', id_dir.name)
            if m is None:
                continue
            pid = int(m.group(1))
            if id_filter is not None and pid not in id_filter:
                continue

            for amp_dir in sorted(id_dir.glob('amp*')):
                if not amp_dir.is_dir():
                    continue
                if amp_filter is not None and amp_dir.name not in amp_filter:
                    continue

                for k_dir in sorted(amp_dir.glob('k*')):
                    if not k_dir.is_dir():
                        continue
                    if k_filter is not None and k_dir.name not in k_filter:
                        continue
                    p_nr = k_dir / 'bp012_nr.npy'
                    p_gt = k_dir / 'bp012_gt.npy'
                    if not (p_nr.exists() and p_gt.exists()):
                        continue
                    self.samples.append({
                        'pid': pid,
                        'amp': amp_dir.name,
                        'k': k_dir.name,
                        'p_nr': p_nr,
                        'p_gt': p_gt,
                    })

        if len(self.samples) == 0:
            raise RuntimeError(
                f'No samples found under {root}. Expected ID*/amp*/k*/ with bp012_nr.npy and bp012_gt.npy.'
            )

        self.pid_to_indices: Dict[int, List[int]] = {}
        for i, s in enumerate(self.samples):
            self.pid_to_indices.setdefault(int(s['pid']), []).append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        nr = np.load(str(s['p_nr']))
        gt = np.load(str(s['p_gt']))

        nr = np.array(nr, dtype=np.int16, copy=False)
        gt = np.array(gt, dtype=np.int16, copy=False)

        u_in = (nr > 0).astype(np.float32)
        o_in = (nr == 2).astype(np.float32)
        x_uo = np.stack([u_in, o_in], axis=0)

        u_gt = (gt > 0).astype(np.float32)[None, ...]
        o_gt = (gt == 2).astype(np.float32)[None, ...]

        # stable conversion (avoid worker-specific numpy subclass issue)
        x_uo_t = torch.tensor(np.array(x_uo, copy=True), dtype=torch.float32)
        u_gt_t = torch.tensor(np.array(u_gt, copy=True), dtype=torch.float32)
        o_gt_t = torch.tensor(np.array(o_gt, copy=True), dtype=torch.float32)

        return {
            'x_uo': x_uo_t,
            'u_gt': u_gt_t,
            'o_gt': o_gt_t,
            'pid': int(s['pid']),
            'amp': s['amp'],
            'k': s['k'],
        }


def split_by_patient(ds: BPDataset, seed: int = 0, tr_ratio=0.8, va_ratio=0.1):
    pids = sorted(ds.pid_to_indices.keys())
    rnd = random.Random(seed)
    rnd.shuffle(pids)

    n = len(pids)
    n_tr = max(1, int(round(n * tr_ratio)))
    n_va = max(1, int(round(n * va_ratio)))
    if n_tr + n_va >= n:
        n_va = max(1, n - n_tr - 1)
    n_te = n - n_tr - n_va
    if n_te <= 0:
        n_te = 1
        n_tr = max(1, n_tr - 1)

    tr_p = set(pids[:n_tr])
    va_p = set(pids[n_tr:n_tr + n_va])
    te_p = set(pids[n_tr + n_va:])

    tr_idx, va_idx, te_idx = [], [], []
    for i, s in enumerate(ds.samples):
        pid = int(s['pid'])
        if pid in tr_p:
            tr_idx.append(i)
        elif pid in va_p:
            va_idx.append(i)
        else:
            te_idx.append(i)
    return tr_idx, va_idx, te_idx


def dice_loss(prob: torch.Tensor, tgt: torch.Tensor, eps=1e-6):
    prob = prob.float()
    tgt = tgt.float()
    inter = (prob * tgt).sum(dim=(2, 3, 4))
    den = prob.sum(dim=(2, 3, 4)) + tgt.sum(dim=(2, 3, 4))
    dice = (2 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


def dice_score(logit: torch.Tensor, tgt: torch.Tensor, thr: float = 0.5, eps=1e-6):
    p = (torch.sigmoid(logit) > thr).float()
    t = tgt.float()
    inter = (p * t).sum()
    den = p.sum() + t.sum()
    return float((2 * inter / (den + eps)).item())


def downsample_mask(lbl: torch.Tensor, out_shape: Tuple[int, int, int]):
    """lbl: (B,1,D,H,W) binary {0,1}; use max-pool to preserve thin vessels."""
    td, th, tw = out_shape
    _, _, d, h, w = lbl.shape
    if (d, h, w) == (td, th, tw):
        return lbl
    kd = max(1, d // td)
    kh = max(1, h // th)
    kw = max(1, w // tw)
    # exact / power-of-two here (128->64/32), but keep safe fallback
    if d % td == 0 and h % th == 0 and w % tw == 0:
        return F.max_pool3d(lbl, kernel_size=(kd, kh, kw), stride=(kd, kh, kw))
    # fallback (should rarely hit)
    return (F.interpolate(lbl, size=out_shape, mode='trilinear', align_corners=False) > 0.2).float()


def save_ckpt(path: Path, model, opt, ep: int, best: float, extra: Optional[dict] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model': model.state_dict(),
        'opt': opt.state_dict() if opt is not None else None,
        'epoch': int(ep),
        'best': float(best),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))


class Losses(nn.Module):
    def __init__(self, overlap_w=0.5, ds_w=0.25, ds_overlap_w=0.25, cons_w=0.05):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.overlap_w = float(overlap_w)
        self.ds_w = float(ds_w)
        self.ds_overlap_w = float(ds_overlap_w)
        self.cons_w = float(cons_w)

    def _bce_dice(self, logit, tgt):
        prob = torch.sigmoid(logit)
        return self.bce(logit, tgt) + dice_loss(prob, tgt)

    def forward(self, out: Dict[str, torch.Tensor], u_gt: torch.Tensor, o_gt: torch.Tensor, variant: str):
        stats: Dict[str, float] = {}
        loss = 0.0

        # Main union loss always exists
        L_u = self._bce_dice(out['u_logit'], u_gt)
        loss = loss + L_u
        stats['L_u'] = float(L_u.detach().item())

        if variant in ['dual', 'dual_ds']:
            L_o = self._bce_dice(out['o_logit'], o_gt)
            loss = loss + self.overlap_w * L_o
            stats['L_o'] = float(L_o.detach().item())

            # consistency (overlap should be subset of union)
            u_prob = torch.sigmoid(out['u_logit'])
            o_prob = torch.sigmoid(out['o_logit'])
            L_cons = F.relu(o_prob - u_prob).mean()
            loss = loss + self.cons_w * L_cons
            stats['L_cons'] = float(L_cons.detach().item())

        if variant == 'dual_ds':
            # DS targets at head scales
            for key_u, key_o in [('u_logit_ds64', 'o_logit_ds64'), ('u_logit_ds32', 'o_logit_ds32')]:
                shp = out[key_u].shape[-3:]
                u_gt_ds = downsample_mask(u_gt, shp)
                o_gt_ds = downsample_mask(o_gt, shp)

                L_uds = self._bce_dice(out[key_u], u_gt_ds)
                L_ods = self._bce_dice(out[key_o], o_gt_ds)
                loss = loss + self.ds_w * L_uds + self.ds_overlap_w * L_ods
                stats[f'L_{key_u}'] = float(L_uds.detach().item())
                stats[f'L_{key_o}'] = float(L_ods.detach().item())

        return loss, stats


def train_epoch(model, dl, opt, scaler, device, args, losses):
    model.train()
    t0 = time.time()
    amp_enable = bool(device.type == 'cuda' and int(args.amp) == 1)
    grad_accum = max(1, int(args.grad_accum))

    run = {'loss': 0.0, 'uDice': 0.0, 'oDice': 0.0, 'n': 0}

    opt.zero_grad(set_to_none=True)
    for it, batch in enumerate(dl, 1):
        x = batch['x_uo'].to(device, non_blocking=True)
        u_gt = batch['u_gt'].to(device, non_blocking=True)
        o_gt = batch['o_gt'].to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', enabled=amp_enable):
            out = model(x)
            loss, _ = losses(out, u_gt, o_gt, variant=args.variant)

        scaler.scale(loss / float(grad_accum)).backward()

        if it % grad_accum == 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            run['loss'] += float(loss.item())
            run['uDice'] += dice_score(out['u_logit'], u_gt, thr=float(args.thr))
            if 'o_logit' in out:
                run['oDice'] += dice_score(out['o_logit'], o_gt, thr=float(args.thr))
            run['n'] += 1

    # flush remainder grads
    if len(dl) % grad_accum != 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    for k in ['loss', 'uDice', 'oDice']:
        run[k] /= max(1, run['n'])
    run['time'] = time.time() - t0
    return run


@torch.no_grad()
def validate(model, dl, device, args):
    model.eval()
    u_list, o_list = [], []
    for batch in dl:
        x = batch['x_uo'].to(device, non_blocking=True)
        u_gt = batch['u_gt'].to(device, non_blocking=True)
        o_gt = batch['o_gt'].to(device, non_blocking=True)
        out = model(x)
        u_list.append(dice_score(out['u_logit'], u_gt, thr=float(args.thr)))
        if 'o_logit' in out:
            o_list.append(dice_score(out['o_logit'], o_gt, thr=float(args.thr)))
    return {
        'uDice': float(np.mean(u_list)) if u_list else 0.0,
        'oDice': float(np.mean(o_list)) if o_list else 0.0,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--work_dir', type=str, required=True)

    ap.add_argument('--ids', type=str, default='all')
    ap.add_argument('--amps', type=str, default='all')
    ap.add_argument('--ks', type=str, default='all')

    ap.add_argument('--variant', type=str, required=True, choices=['single', 'dual', 'dual_ds'])

    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--base', type=int, default=24)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--amp', type=int, default=1)
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save_every', type=int, default=10)

    # loss weights
    ap.add_argument('--overlap_w', type=float, default=0.5)
    ap.add_argument('--ds_w', type=float, default=0.25)
    ap.add_argument('--ds_overlap_w', type=float, default=0.25)
    ap.add_argument('--cons_w', type=float, default=0.05)

    args = ap.parse_args()

    seed_all(int(args.seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    ds = BPDataset(
        root=args.data_root,
        ids=args.ids,
        amps=args.amps,
        ks=args.ks,
    )
    tr_idx, va_idx, te_idx = split_by_patient(ds, seed=int(args.seed))
    tr_ds, va_ds = Subset(ds, tr_idx), Subset(ds, va_idx)

    dl_tr = DataLoader(
        tr_ds,
        batch_size=int(args.batch),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )
    dl_va = DataLoader(
        va_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    model = build_model(args.variant, base=int(args.base)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.wd))
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and int(args.amp) == 1))

    losses = Losses(
        overlap_w=float(args.overlap_w),
        ds_w=float(args.ds_w),
        ds_overlap_w=float(args.ds_overlap_w),
        cons_w=float(args.cons_w),
    )

    run_meta = {
        'variant': args.variant,
        'data_root': args.data_root,
        'filters': {'ids': args.ids, 'amps': args.amps, 'ks': args.ks},
        'n_samples': len(ds),
        'n_train': len(tr_idx),
        'n_val': len(va_idx),
        'n_test': len(te_idx),
    }
    with open(work / 'nr_align_run_meta.json', 'w', encoding='utf-8') as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    print(f"[nr_align_data] n_samples={len(ds)} device={device} fixed_sample={ds.samples[0]['p_nr'].parent}", flush=True)
    print(f"[nr_align_split] train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)} (patient-level)", flush=True)
    print(f"[nr_align_variant] {args.variant}", flush=True)

    best_va_u = -1.0
    history = []

    for ep in range(1, int(args.epochs) + 1):
        tr = train_epoch(model, dl_tr, opt, scaler, device, args, losses)
        va = validate(model, dl_va, device, args)

        row = {
            'ep': ep,
            'train_loss': tr['loss'],
            'train_uDice': tr['uDice'],
            'train_oDice': tr['oDice'],
            'val_uDice': va['uDice'],
            'val_oDice': va['oDice'],
            'time_sec': tr['time'],
        }
        history.append(row)

        print(
            f"[ep {ep:03d}] train loss={tr['loss']:.4f} uDice={tr['uDice']:.4f} oDice={tr['oDice']:.4f} | "
            f"val uDice={va['uDice']:.4f} oDice={va['oDice']:.4f} | t={tr['time']:.1f}s",
            flush=True,
        )

        if va['uDice'] > best_va_u:
            best_va_u = va['uDice']
            save_ckpt(
                work / 'nr_align_best.pt',
                model=model,
                opt=opt,
                ep=ep,
                best=best_va_u,
                extra={'variant': args.variant, 'base': int(args.base)},
            )
            print(f"[nr_align_save] best -> {work/'nr_align_best.pt'} (val_uDice={best_va_u:.4f})", flush=True)

        if ep % int(args.save_every) == 0:
            save_ckpt(
                work / f'nr_align_ckpt_ep{ep:03d}.pt',
                model=model,
                opt=opt,
                ep=ep,
                best=best_va_u,
                extra={'variant': args.variant, 'base': int(args.base)},
            )
            print(f"[nr_align_save] epoch ckpt -> {work / f'nr_align_ckpt_ep{ep:03d}.pt'}", flush=True)

        with open(work / 'nr_align_train_history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"[nr_align_done] best val_uDice={best_va_u:.4f}", flush=True)


if __name__ == '__main__':
    main()
