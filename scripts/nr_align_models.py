import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3(in_ch, out_ch, bias=False):
    return nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = conv3(ch, ch, bias=False)
        self.n1 = nn.InstanceNorm3d(ch, affine=True)
        self.c2 = conv3(ch, ch, bias=False)
        self.n2 = nn.InstanceNorm3d(ch, affine=True)

    def forward(self, x):
        r = x
        x = F.relu(self.n1(self.c1(x)), inplace=True)
        x = self.n2(self.c2(x))
        return F.relu(x + r, inplace=True)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.n = nn.InstanceNorm3d(out_ch, affine=True)
        self.rb = ResBlock(out_ch)

    def forward(self, x):
        x = F.relu(self.n(self.c(x)), inplace=True)
        return self.rb(x)


class ResUNet(nn.Module):
    """
    Input x_uo: (B,2,D,H,W)
      ch0 = union_raw = (bp012_nr > 0)
      ch1 = ov2_raw   = (bp012_nr == 2)

    Return dict feats:
      feat32  : (B, 8b, D/4,H/4,W/4)
      feat64  : (B, 4b, D/2,H/2,W/2)
      feat128 : (B, 2b, D,H,W)
    """
    def __init__(self, in_ch: int = 2, base: int = 24):
        super().__init__()
        b = int(base)
        self.base = b

        self.stem = nn.Sequential(
            conv3(in_ch, b, bias=False),
            nn.InstanceNorm3d(b, affine=True),
            nn.ReLU(inplace=True),
            ResBlock(b),
        )
        self.d1 = DownBlock(b, b * 2)      # 128 -> 64
        self.d2 = DownBlock(b * 2, b * 4)  # 64 -> 32
        self.d3 = DownBlock(b * 4, b * 8)  # 32 -> 16

        self.mid = nn.Sequential(ResBlock(b * 8), ResBlock(b * 8))

        # decoder blocks (concat doubles channel)
        self.u3 = nn.ConvTranspose3d(b * 8, b * 4, kernel_size=2, stride=2, bias=True)
        self.n3 = nn.InstanceNorm3d(b * 4, affine=True)
        self.rb3 = ResBlock(b * 8)  # cat with s2(4b) => 8b, spatial 32

        self.u2 = nn.ConvTranspose3d(b * 8, b * 2, kernel_size=2, stride=2, bias=True)
        self.n2 = nn.InstanceNorm3d(b * 2, affine=True)
        self.rb2 = ResBlock(b * 4)  # cat with s1(2b) => 4b, spatial 64

        self.u1 = nn.ConvTranspose3d(b * 4, b, kernel_size=2, stride=2, bias=True)
        self.n1 = nn.InstanceNorm3d(b, affine=True)
        self.rb1 = ResBlock(b * 2)  # cat with s0(b) => 2b, spatial 128

        self.out_ch_main = b * 2
        self.out_ch_ds64 = b * 4
        self.out_ch_ds32 = b * 8

    @staticmethod
    def _cat_skip(x, skip):
        dz = skip.shape[-3] - x.shape[-3]
        dy = skip.shape[-2] - x.shape[-2]
        dx = skip.shape[-1] - x.shape[-1]
        if dz != 0 or dy != 0 or dx != 0:
            x = F.pad(x, [0, max(dx, 0), 0, max(dy, 0), 0, max(dz, 0)])
            x = x[:, :, :skip.shape[-3], :skip.shape[-2], :skip.shape[-1]]
        return torch.cat([x, skip], dim=1)

    def forward(self, x_uo):
        s0 = self.stem(x_uo)   # b, 128
        s1 = self.d1(s0)       # 2b, 64
        s2 = self.d2(s1)       # 4b, 32
        s3 = self.d3(s2)       # 8b, 16

        x = self.mid(s3)       # 8b, 16

        x = F.relu(self.n3(self.u3(x)), inplace=True)  # 4b, 32
        x = self._cat_skip(x, s2)                # 8b, 32
        feat32 = self.rb3(x)

        x = F.relu(self.n2(self.u2(feat32)), inplace=True)  # 2b, 64
        x = self._cat_skip(x, s1)                     # 4b, 64
        feat64 = self.rb2(x)

        x = F.relu(self.n1(self.u1(feat64)), inplace=True)  # b, 128
        x = self._cat_skip(x, s0)                     # 2b, 128
        feat128 = self.rb1(x)

        return {"feat32": feat32, "feat64": feat64, "feat128": feat128}


class SingleHead(nn.Module):
    def __init__(self, base: int = 24):
        super().__init__()
        self.backbone = ResUNet(in_ch=2, base=base)
        self.head_u = nn.Conv3d(self.backbone.out_ch_main, 1, kernel_size=1)

    def forward(self, x_uo):
        feats = self.backbone(x_uo)
        return {"u_logit": self.head_u(feats["feat128"])}


class DualHead(nn.Module):
    def __init__(self, base: int = 24):
        super().__init__()
        self.backbone = ResUNet(in_ch=2, base=base)
        self.head_u = nn.Conv3d(self.backbone.out_ch_main, 1, kernel_size=1)
        self.head_o = nn.Conv3d(self.backbone.out_ch_main, 1, kernel_size=1)

    def forward(self, x_uo):
        feats = self.backbone(x_uo)
        return {
            "u_logit": self.head_u(feats["feat128"]),
            "o_logit": self.head_o(feats["feat128"]),
        }


class DualHeadDS(nn.Module):
    """Main dual heads + deep supervision heads at 64/32 for both union and overlap."""
    def __init__(self, base: int = 24):
        super().__init__()
        self.backbone = ResUNet(in_ch=2, base=base)

        # main heads (128)
        self.head_u = nn.Conv3d(self.backbone.out_ch_main, 1, kernel_size=1)
        self.head_o = nn.Conv3d(self.backbone.out_ch_main, 1, kernel_size=1)

        # DS heads (64 / 32)
        self.head_u_ds64 = nn.Conv3d(self.backbone.out_ch_ds64, 1, kernel_size=1)
        self.head_u_ds32 = nn.Conv3d(self.backbone.out_ch_ds32, 1, kernel_size=1)
        self.head_o_ds64 = nn.Conv3d(self.backbone.out_ch_ds64, 1, kernel_size=1)
        self.head_o_ds32 = nn.Conv3d(self.backbone.out_ch_ds32, 1, kernel_size=1)

    def forward(self, x_uo):
        feats = self.backbone(x_uo)
        out = {
            "u_logit": self.head_u(feats["feat128"]),
            "o_logit": self.head_o(feats["feat128"]),
            "u_logit_ds64": self.head_u_ds64(feats["feat64"]),
            "u_logit_ds32": self.head_u_ds32(feats["feat32"]),
            "o_logit_ds64": self.head_o_ds64(feats["feat64"]),
            "o_logit_ds32": self.head_o_ds32(feats["feat32"]),
        }
        return out


def build_model(variant: str, base: int = 24):
    v = str(variant).strip().lower()
    if v == "single":
        return SingleHead(base=base)
    if v == "dual":
        return DualHead(base=base)
    if v == "dual_ds":
        return DualHeadDS(base=base)
    raise ValueError(f"Unknown NR-Align variant: {variant}")
