#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HRTBDA v7-MSDF + PMKF

PMKF = Parallel Multi-Kernel Fusion

This file does NOT modify your original HRTBDA v7-MSDF model.
It imports the original model, wraps the MSDF module, and applies PMKF
immediately after MSDF.

You only need to change ONE import line below if your original model
file/class name is different.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# CHANGE THIS IMPORT LINE ONLY IF YOUR ORIGINAL FILE/CLASS NAME DIFFERS
# ---------------------------------------------------------------------
#
# Example expected original file:
#   models/hrtbda_v7_msdf.py
#
# Example expected original class:
#   HRTBDAv7MSDF
#
# If your original file/class name is different, change this line only.
#
from models.hrtbda_v7_msdf import HRTBDAv7MSDF as BaseHRTBDAv7MSDF


class PMKFBlock(nn.Module):
    """
    Parallel Multi-Kernel Fusion block.

    Input:
        MSDF feature map: [B, C, H, W]

    Output:
        Enhanced feature map: [B, C, H, W]

    Kernels:
        3x3 -> local damage texture
        5x5 -> medium building damage pattern
        7x7 -> larger contextual damage/debris pattern
    """

    def __init__(self, in_channels, branch_channels=None, dropout=0.10):
        super().__init__()

        if branch_channels is None:
            branch_channels = max(in_channels // 4, 32)

        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        self.branch_7x7 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                branch_channels * 3,
                in_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

        # Residual strength.
        # Starts small so PMKF does not destroy the original MSDF features.
        self.alpha = nn.Parameter(torch.tensor(0.10))

    def forward(self, x):
        x3 = self.branch_3x3(x)
        x5 = self.branch_5x5(x)
        x7 = self.branch_7x7(x)

        fused = torch.cat([x3, x5, x7], dim=1)
        fused = self.fusion(fused)

        return x + self.alpha * fused


class MSDFWithPMKF(nn.Module):
    """
    Wrapper around the original MSDF module.

    It runs:
        original MSDF
        -> PMKF
        -> returns enhanced MSDF features

    This avoids editing the old model file.
    """

    def __init__(self, original_msdf, msdf_channels, dropout=0.10):
        super().__init__()
        self.original_msdf = original_msdf
        self.pmkf = PMKFBlock(
            in_channels=msdf_channels,
            dropout=dropout,
        )

    def forward(self, *args, **kwargs):
        out = self.original_msdf(*args, **kwargs)

        # Most MSDF modules return one tensor: [B, C, H, W]
        if torch.is_tensor(out):
            return self.pmkf(out)

        # Some modules return tuple/list, for example:
        #   (features, auxiliary_output)
        # In that case, apply PMKF only to the first tensor.
        if isinstance(out, tuple):
            out_list = list(out)
            if torch.is_tensor(out_list[0]):
                out_list[0] = self.pmkf(out_list[0])
            return tuple(out_list)

        if isinstance(out, list):
            if len(out) > 0 and torch.is_tensor(out[0]):
                out[0] = self.pmkf(out[0])
            return out

        raise TypeError(
            "MSDF output type is not supported. "
            f"Got type: {type(out)}. Expected Tensor, tuple, or list."
        )


class HRTBDAv7MSDFPMKF(BaseHRTBDAv7MSDF):
    """
    New model:
        HRTBDA v7-MSDF + PMKF

    This class inherits your original HRTBDA v7-MSDF model and replaces
    only the MSDF module with MSDF + PMKF.

    Important:
        msdf_channels must match the output channels of your MSDF feature map.

    Common values:
        64, 128, 256, 512

    If you are not sure, first try 256.
    """

    def __init__(
        self,
        *args,
        msdf_channels=256,
        pmkf_dropout=0.10,
        msdf_module_name="msdf",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not hasattr(self, msdf_module_name):
            available_modules = [name for name, _ in self.named_children()]
            raise AttributeError(
                f"Could not find MSDF module named '{msdf_module_name}'.\n"
                f"Available top-level modules are: {available_modules}\n\n"
                "Fix this by setting msdf_module_name to the correct name.\n"
                "For example:\n"
                "    model = HRTBDAv7MSDFPMKF(msdf_module_name='MSDF')"
            )

        original_msdf = getattr(self, msdf_module_name)

        wrapped_msdf = MSDFWithPMKF(
            original_msdf=original_msdf,
            msdf_channels=msdf_channels,
            dropout=pmkf_dropout,
        )

        setattr(self, msdf_module_name, wrapped_msdf)

        print("=" * 70)
        print("Created HRTBDA v7-MSDF + PMKF model")
        print(f"Wrapped MSDF module name : {msdf_module_name}")
        print(f"MSDF channels            : {msdf_channels}")
        print(f"PMKF dropout             : {pmkf_dropout}")
        print("=" * 70)


# Optional aliases, useful if your training script imports by name.
HRTBDAv7_MSDF_PMKF = HRTBDAv7MSDFPMKF
HRTBDA_V7_MSDF_PMKF = HRTBDAv7MSDFPMKF


def build_model(*args, **kwargs):
    """
    Optional builder function.
    Some training scripts use build_model().
    """
    return HRTBDAv7MSDFPMKF(*args, **kwargs)


if __name__ == "__main__":
    print("This file defines the HRTBDA v7-MSDF + PMKF model.")
    print("Import it from your training script like this:")
    print("from models.hrtbda_v7_msdf_pmkf import HRTBDAv7MSDFPMKF")