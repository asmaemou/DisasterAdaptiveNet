#!/usr/bin/env python3
from pathlib import Path

PROJECT_ROOT = Path("/homes/j244s673/documents/wsu/phd/DisasterAdaptiveNet")

SRC = PROJECT_ROOT / "transformer/scripts/train_xbd_hrtbda_v5_multilabel_crop_cascade.py"
DST = PROJECT_ROOT / "cross_DS/scripts/pmkf/train_xbd_hrtbda_v5_pmkf_multilabel_crop_cascade.py"

if not SRC.exists():
    raise FileNotFoundError(f"Source script not found: {SRC}")

text = SRC.read_text(encoding="utf-8")

if "class PMKFBlock(nn.Module):" in text:
    print("PMKFBlock already exists in source/copy. Writing without duplicate insertion.")
else:
    pmkf_block = r'''

class PMKFBlock(nn.Module):
    """
    Parallel Multi-Kernel Fusion block.

    Added after CSF fusion in Phase II:
        CSF fused features -> PMKF -> decoder

    Input/output:
        [B, C, H, W] -> [B, C, H, W]

    3x3 captures local damage texture.
    5x5 captures medium-scale partial damage.
    7x7 captures wider collapse/debris context.

    Depthwise-separable convolutions are used to keep memory lower.
    """

    def __init__(self, channels: int, branch_channels: Optional[int] = None, dropout: float = 0.10):
        super().__init__()

        if branch_channels is None:
            branch_channels = max(channels // 4, 32)

        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )

        self.branch7 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.GELU(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(p=dropout),
        )

        # Starts small so PMKF does not disturb the original CSF features too much.
        self.alpha = nn.Parameter(torch.tensor(0.10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)

        fused = torch.cat([x3, x5, x7], dim=1)
        fused = self.fuse(fused)

        return x + self.alpha * fused
'''

    marker = "class LayerNorm2d(nn.Module):"
    if marker not in text:
        raise RuntimeError(f"Could not find insertion marker: {marker}")

    text = text.replace(marker, pmkf_block + "\n\n" + marker)


old_init = "        self.csf = nn.ModuleList([CSFModule(c) for c in self.backbone.channels])"
new_init = """        self.csf = nn.ModuleList([CSFModule(c) for c in self.backbone.channels])
        self.pmkf = nn.ModuleList([PMKFBlock(c, dropout=0.10) for c in self.backbone.channels])"""

if new_init not in text:
    if old_init not in text:
        raise RuntimeError("Could not find HRTBDAPhase2 CSF line for PMKF insertion.")
    text = text.replace(old_init, new_init)


old_forward = """        fused = [module(a, b) for module, a, b in zip(self.csf, fpre, fpost)]
        damage_logits = self.decoder(fused, output_size=pre.shape[-2:])"""

new_forward = """        fused = [module(a, b) for module, a, b in zip(self.csf, fpre, fpost)]
        fused = [pmkf(feat) for pmkf, feat in zip(self.pmkf, fused)]
        damage_logits = self.decoder(fused, output_size=pre.shape[-2:])"""

if new_forward not in text:
    if old_forward not in text:
        raise RuntimeError("Could not find HRTBDAPhase2 forward CSF/decoder lines.")
    text = text.replace(old_forward, new_forward)


text = text.replace(
    'print("Architecture: HRTBDA v5 4-branch HRNet-style + DCSwin + CSF fusion", flush=True)',
    'print("Architecture: HRTBDA v5 4-branch HRNet-style + DCSwin + CSF fusion + PMKF", flush=True)',
)

text = text.replace(
    'print("===== HRTBDA V5 MULTI-LABEL RARE-CROP CASCADED TRAINING =====", flush=True)',
    'print("===== HRTBDA V5 + PMKF MULTI-LABEL RARE-CROP CASCADED TRAINING =====", flush=True)',
)

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(text, encoding="utf-8")

print(f"Wrote PMKF training script:")
print(DST)
print()
print("Check PMKF lines with:")
print(f"grep -n \"PMKF\\|self.pmkf\\|fused = \\[pmkf\" {DST}")