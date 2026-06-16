# Recommended public parameters

This file records the default parameters I recommend exposing in the public repository.

## Main generation setting

```bash
--amp_list_mm 6,10,14,18
--ks 0-2
--variant_map sym_small,sym_large,same_side_small
--resp_period_ms 4000
--card_period_ms 850
--dt_ms_sym_small 40,90
--dt_ms_sym_large 140,240
--dt_ms_same_side_small 50,110
--same_side_center_shift_ms 80,220
```

## Why these settings are useful

- `amp6`: relatively mild non-rigid mismatch.
- `amp10`: common moderate mismatch.
- `amp14` / `amp18`: stronger mismatch, useful for stress testing and avoiding an over-easy training set.
- `sym_small`: two views close to the same physiological state.
- `sym_large`: two views on opposite sides of the anchor with a larger time gap.
- `same_side_small`: both views deviate in the same direction from the anchor phase, which avoids a dataset that is always perfectly symmetric around GT.

## Quality gates

Keep these gates enabled unless you are doing ablation:

```bash
--gate_pair_gap 1
--pair_gap_p95_mm_min 3.0
--pair_gap_p95_mm_max 20.0
--gate_pair_center 1
--pair_center_p95_mm_max 12.0
--gate_pair_outside_sum 1
--pair_outside_sum_max 0.04
--gate_detJ 1
--detJ_min_thr 0.0
--detJ_nonpos_max 0.0
--gate_warpback 1
--warpback_p95_mm_max 2.0
```

The goal is not to generate the most dramatic deformation possible. The goal is to generate difficult but still anatomically/plausibly valid paired masks.
