# Data Format

## Input Masks

The public pipeline starts after an upstream prior reconstruction stage.

```text
data/gt128/{ID}.npy
```

Requirements:

- shape: `(128, 128, 128)`
- dtype: `uint8` or convertible to `uint8`
- values: `{0,1}`
- axis order: XYZ

## Geometry Metadata

```text
data/meta/{ID}.npz
```

Preferred fields:

```text
DSD1, DSO1, ang1_deg
DSD2, DSO2, ang2_bp_deg
det_spacing, v_size, off2
```

`off2` is optional and defaults to `(0,0,0)`.

## Generated Cases

```text
out_root/ID001/amp6/k00/
|-- view1_mask.npy
|-- view2_mask.npy
`-- meta.json
```

Both view masks are XYZ, `128^3`, and binary.

## BP Training Files

```text
out_root/ID001/amp6/k00/
|-- bp1_gt.npy
|-- bp2_gt.npy
|-- bp012_gt.npy
|-- bp1_nr.npy
|-- bp2_nr.npy
`-- bp012_nr.npy
```

`bp012_*` stores two-view union/overlap counts:

- `0`: background
- `1`: appears in one view
- `2`: overlap from both views
