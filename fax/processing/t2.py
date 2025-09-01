# fax/processing/transformations.py
# largely copied from https://github.com/ericyuegu/hal
# Pure-Torch preprocessing where possible.

from __future__ import annotations
from typing import Callable, List

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from fax.stats import FeatureStats
from fax.constants import (
    INCLUDED_BUTTONS,
    STICK_XY_CLUSTER_CENTERS_V0,
    SHOULDER_CLUSTER_CENTERS_V0,
)

# for type hinting
Transformation = Callable[..., torch.Tensor]

# ---------------------------------------------------------------------
# Torch copies of cluster centers (moved to the input's device on use)
_STICK_CENTERS = torch.as_tensor(STICK_XY_CLUSTER_CENTERS_V0, dtype=torch.float32)  # [C, 2]
_SHOULDER_CENTERS = torch.as_tensor(SHOULDER_CLUSTER_CENTERS_V0, dtype=torch.float32)  # [C]

_EPS = 1e-6


# SIMPLE PUBLIC TRANSFORMATIONS
def cast_int32(array: torch.Tensor, stats: FeatureStats) -> torch.Tensor:
    """Identity function; cast to int32."""
    _ = stats  # unused
    return array.to(torch.int32)


def normalize(array: torch.Tensor, stats: FeatureStats) -> torch.Tensor:
    """Normalize feature to [-1, 1] with epsilon for stability."""
    denom = max(float(stats.max - stats.min), _EPS)
    return (2.0 * (array - stats.min) / denom - 1.0).to(torch.float32)


def invert_and_normalize(array: torch.Tensor, stats: FeatureStats) -> torch.Tensor:
    """Invert and normalize feature to [-1, 1] with epsilon for stability."""
    denom = max(float(stats.max - stats.min), _EPS)
    return (2.0 * (stats.max - array) / denom - 1.0).to(torch.float32)


def standardize(array: torch.Tensor, stats: FeatureStats) -> torch.Tensor:
    """Standardize feature to mean 0 and std 1 with epsilon for stability."""
    denom = max(float(stats.std), _EPS)
    return ((array - stats.mean) / denom).to(torch.float32)


def union(array_1: torch.Tensor, array_2: torch.Tensor) -> torch.Tensor:
    """Perform logical OR of two features."""
    return array_1 | array_2


# ---------------------------------------------------------------------
# Torch helpers replacing NumPy flows


def _nearest_2d_cluster(xy_L2: torch.Tensor, centers_C2: torch.Tensor) -> torch.Tensor:
    """Return argmin center index for each (x,y). xy:[L,2], centers:[C,2] -> idx:[L]."""
    centers = centers_C2.to(device=xy_L2.device, dtype=xy_L2.dtype)
    d = torch.cdist(xy_L2, centers)  # [L, C] Euclidean distances
    return torch.argmin(d, dim=-1)  # [L]


def _nearest_1d_cluster(x_L: torch.Tensor, centers_C: torch.Tensor) -> torch.Tensor:
    """Return argmin center index for each x. x:[L], centers:[C] -> idx:[L]."""
    centers = centers_C.to(device=x_L.device, dtype=x_L.dtype)
    d2 = (x_L.unsqueeze(-1) - centers.unsqueeze(0)) ** 2  # [L, C]
    return torch.argmin(d2, dim=-1)


def _one_hot(idx_L: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot to float32. idx:[L] -> [L, num_classes]."""
    return F.one_hot(idx_L.to(torch.long), num_classes=num_classes).to(torch.float32)


def _convert_multi_hot_to_one_hot_torch(buttons_LD: torch.Tensor) -> torch.Tensor:
    """
    Stateless per-frame reduction:
      - If multiple pressed: choose LEFT-MOST (lowest index) pressed bit.
      - If none pressed: set the last column ("no_button") to 1.
    buttons_LD: bool/int tensor of shape [L, D]
    returns one-hot float32 [L, D]
    """
    if buttons_LD.dtype not in (torch.int32, torch.int64):
        buttons_LD = buttons_LD.to(torch.int64)

    row_sum = buttons_LD.sum(dim=-1)  # [L]
    leftmost_idx = torch.argmax(buttons_LD, dim=-1)  # [L] first 1 wins for ties
    no_press_mask = row_sum == 0
    leftmost_idx = torch.where(
        no_press_mask,
        torch.full_like(leftmost_idx, buttons_LD.shape[-1] - 1),
        leftmost_idx,
    )
    return _one_hot(leftmost_idx, num_classes=buttons_LD.shape[-1])


# ---------------------------------------------------------------------
# PREPROCESSING (pure Torch versions)


def encode_main_stick_one_hot_coarse(sample: TensorDict, player: str) -> torch.Tensor:
    """
    Quantize (x,y) to nearest 2D cluster center and one-hot.
    Returns [L, C] float32 on the same device as inputs.
    """
    x = sample[f'{player}_main_stick_x'].to(torch.float32)
    y = sample[f'{player}_main_stick_y'].to(torch.float32)
    xy = torch.stack([x, y], dim=-1)  # [L, 2]
    idx = _nearest_2d_cluster(xy, _STICK_CENTERS)  # [L]
    return _one_hot(idx, num_classes=_STICK_CENTERS.shape[0])


def encode_c_stick_one_hot_coarse(sample: TensorDict, player: str) -> torch.Tensor:
    x = sample[f'{player}_c_stick_x'].to(torch.float32)
    y = sample[f'{player}_c_stick_y'].to(torch.float32)
    xy = torch.stack([x, y], dim=-1)  # [L, 2]
    idx = _nearest_2d_cluster(xy, _STICK_CENTERS)  # [L]
    return _one_hot(idx, num_classes=_STICK_CENTERS.shape[0])


def encode_shoulder_one_hot_coarse(sample: TensorDict, player: str) -> torch.Tensor:
    l = sample[f'{player}_l_shoulder'].to(torch.float32)
    r = sample[f'{player}_r_shoulder'].to(torch.float32)
    s = torch.maximum(l, r)  # [L]
    idx = _nearest_1d_cluster(s, _SHOULDER_CENTERS)  # [L]
    return _one_hot(idx, num_classes=_SHOULDER_CENTERS.shape[0])


def encode_buttons_one_hot(sample: TensorDict, player: str) -> torch.Tensor:
    """
    Combine X/Y -> jump, L/R -> shoulder; reduce multi-press per frame to a single class:
      - left-most pressed wins; if none, "no_button" (last class).
    """
    a = sample[f'{player}_button_a'].bool()
    b = sample[f'{player}_button_b'].bool()
    x = sample[f'{player}_button_x'].bool()
    y = sample[f'{player}_button_y'].bool()
    z = sample[f'{player}_button_z'].bool()
    l = sample[f'{player}_button_l'].bool()
    r = sample[f'{player}_button_r'].bool()

    jump = x | y
    shoulder = l | r
    no_button = ~(a | b | jump | z | shoulder)

    buttons = torch.stack([a, b, jump, z, shoulder, no_button], dim=-1)  # [L, 6]
    return _convert_multi_hot_to_one_hot_torch(buttons)


# ---------------------------------------------------------------------
# POSTPROCESSING (unchanged API; stays Torch)


def sample_main_stick_coarse(pred_C: TensorDict, temperature: float = 1.0) -> tuple[float, float]:
    probs = torch.softmax(pred_C['main_stick'] / temperature, dim=-1)
    idx = int(torch.multinomial(probs, num_samples=1).item())
    x, y = _STICK_CENTERS[idx]
    return float(x), float(y)


def sample_c_stick_coarse(pred_C: TensorDict, temperature: float = 1.0) -> tuple[float, float]:
    probs = torch.softmax(pred_C['c_stick'] / temperature, dim=-1)
    idx = int(torch.multinomial(probs, num_samples=1).item())
    x, y = _STICK_CENTERS[idx]
    return float(x), float(y)


def sample_single_button(pred_C: TensorDict, temperature: float = 1.0) -> List[str]:
    probs = torch.softmax(pred_C['buttons'] / temperature, dim=-1)
    idx = int(torch.multinomial(probs, num_samples=1).item())
    button = INCLUDED_BUTTONS[idx]
    return [button]


def sample_analog_shoulder_coarse(pred_C: TensorDict, temperature: float = 1.0) -> float:
    probs = torch.softmax(pred_C['shoulder'] / temperature, dim=-1)
    idx = int(torch.multinomial(probs, num_samples=1).item())
    return float(_SHOULDER_CENTERS[idx].item())
