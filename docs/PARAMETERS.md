# Parameter notes

## `amp_list_mm`

`amp_list_mm` is the most important public-facing difficulty parameter. In the dual-async generator, each requested amplitude is used as the target pair difficulty after the SVF deformation is built. With the default `pair_scale_target_mode=gap`, the generator scales the deformation so that the pair gap is close to the requested amplitude.

Recommended public values:

```bash
--amp_list_mm 6,10,14,18
```

This is better than only using one amplitude because it prevents the model from seeing only a narrow deformation distribution.

## Temporal variants

```bash
--variant_map sym_small,sym_large,same_side_small
```

- `sym_small`: two views are close to the anchor phase.
- `sym_large`: two views are farther apart and placed on opposite sides of the anchor.
- `same_side_small`: two views are on the same side of the anchor phase, avoiding an unrealistically symmetric-only setting.

## Physiological periods

```bash
--resp_period_ms 4000
--card_period_ms 850
```

These convert the sampled acquisition time gap into respiratory/cardiac phase offsets. They are not meant to claim exact patient-specific physiology; they provide a controlled, interpretable simulation rule.

## Geometry and deformation safety

The generator uses SVF exponential integration, SDF warping, determinant-Jacobian checks, outside-crop checks, border checks, and warp-back consistency checks. These gates should stay enabled for the main experiments.
