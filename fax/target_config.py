# largely copied from https://github.com/ericyuegu/hal

from typing import Dict, Optional, Tuple

import attr
import numpy as np

from fax.constants import (
    INCLUDED_BUTTONS,
    STICK_XY_CLUSTER_CENTERS_V0,
    SHOULDER_CLUSTER_CENTERS_V0,
)
from fax.transformations import (
    Transformation,
    encode_main_stick_one_hot_coarse,
    encode_c_stick_one_hot_coarse,
    encode_shoulder_one_hot_coarse,
    encode_buttons_one_hot,
)


@attr.s(auto_attribs=True)
class TargetConfig:
    """Configuration for how we structure input features, offsets, and grouping into heads."""

    # Controller inputs
    transformation_by_target: Dict[str, Transformation]

    # Mapping from feature name to frame offset relative to sampled index
    # e.g. to predict controller inputs from 5 frames in the future, set buttons_5 = 5, etc.
    # +1 HAS ALREADY BEEN APPLIED TO CONTROLLER INPUTS AT DATASET CREATION,
    # meaning next frame's controller ("targets") are matched with current frame's gamestate ("inputs")
    frame_offsets_by_target: Dict[str, int]

    # Input dimensions (D,) of concatenated features after preprocessing
    # TensorDict does not support differentiated sizes across keys for the same dimension
    target_shapes_by_head: Dict[str, Tuple[int, ...]]

    # Parameters for Gaussian loss
    reference_points: Optional[np.ndarray] = None
    sigma: float = 0.08

    # If specified, we will predict multiple heads at once
    multi_token_heads: Optional[Tuple[int, ...]] = None

    @property
    def target_size(self) -> int:
        """Total dimension of target features."""
        return sum(shape[0] for shape in self.target_shapes_by_head.values())


def coarse_shoulder() -> TargetConfig:
    return TargetConfig(
        transformation_by_target={
            'main_stick': encode_main_stick_one_hot_coarse,
            'c_stick': encode_c_stick_one_hot_coarse,
            'shoulder': encode_shoulder_one_hot_coarse,
            'buttons': encode_buttons_one_hot,
        },
        frame_offsets_by_target={
            'main_stick': 0,
            'c_stick': 0,
            'shoulder': 0,
            'buttons': 0,
        },
        target_shapes_by_head={
            'main_stick': (len(STICK_XY_CLUSTER_CENTERS_V0),),
            'c_stick': (len(STICK_XY_CLUSTER_CENTERS_V0),),
            'shoulder': (len(SHOULDER_CLUSTER_CENTERS_V0),),
            'buttons': (len(INCLUDED_BUTTONS),),
        },
    )
