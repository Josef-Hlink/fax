# largely copied from https://github.com/ericyuegu/hal

from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, cast

import attr
import numpy as np
import torch
from tensordict import TensorDict

from fax.constants import (
    ACTION_EMBEDDING_DIM,
    CHARACTER_EMBEDDING_DIM,
    INCLUDED_BUTTONS,
    SHOULDER_CENTERS,
    STAGE_EMBEDDING_DIM,
    STICK_CENTERS,
    Player,
)

from .transformations import (
    Transformation,
    cast_int32,
    encode_buttons_one_hot,
    encode_c_stick_one_hot_coarse,
    encode_main_stick_one_hot_coarse,
    encode_shoulder_one_hot_coarse,
    invert_and_normalize,
    normalize,
    sample_analog_shoulder_coarse,
    sample_c_stick_coarse,
    sample_main_stick_coarse,
    sample_single_button,
    standardize,
)


@attr.s(auto_attribs=True)
class InputConfig:
    """Configuration for how we structure input features, offsets, and grouping into heads."""

    # Features to preprocess twice, specific to player state
    player_features: Tuple[str, ...]

    # Mapping from feature name to transformation function
    # Must include embedded features such as stage, character, action, but embedding happens at model arch
    # Feature names that do not exist in raw sample are assumed to preprocess using multiple features
    transformation_by_feature_name: Dict[str, Transformation]

    # Mapping from transformed/preprocessed input to frame offset relative to sample index
    # e.g. to include controller inputs from prev frame with current frame gamestate, set p1_button_a = -1, etc.
    # +1 HAS ALREADY BEEN APPLIED TO CONTROLLER INPUTS AT DATASET CREATION,
    # meaning next frame's controller ("targets") are matched with current frame's gamestate ("inputs")
    frame_offsets_by_input: Dict[str, int]

    # Mapping from head name to features to be fed to that head
    # Usually for int categorical features
    # All unlisted features are concatenated to the default "gamestate" head
    grouped_feature_names_by_head: Dict[str, Tuple[str, ...]]

    # Input dimensions (D,) of concatenated features after preprocessing
    # TensorDict does not support differentiated sizes across keys for the same dimension
    # *May be dynamically updated with data config*
    input_shapes_by_head: Dict[str, Tuple[int, ...]]

    @property
    def input_size(self) -> int:
        """Total dimension of input features."""
        return sum(shape[0] for shape in self.input_shapes_by_head.values())


def get_input_config() -> InputConfig:
    """
    Baseline input features.

    Separate embedding heads for stage, character, & action.
    No controller, no platforms, no projectiles.
    """

    return InputConfig(
        player_features=(
            'character',
            'action',
            'percent',
            'stock',
            'facing',
            'invulnerable',
            'jumps_left',
            'on_ground',
            'shield_strength',
            'position_x',
            'position_y',
        ),
        transformation_by_feature_name={
            # Shared/embedded features are passed unchanged, to be embedded by model
            'stage': cast_int32,
            'character': cast_int32,
            'action': cast_int32,
            # Normalized player features
            'percent': normalize,
            'stock': normalize,
            'facing': normalize,
            'invulnerable': normalize,
            'jumps_left': normalize,
            'on_ground': normalize,
            'shield_strength': invert_and_normalize,
            'position_x': standardize,
            'position_y': standardize,
            'controller': partial(concat_controller_inputs, target_config=get_target_config()),
        },
        frame_offsets_by_input={
            'controller': -1,
        },
        grouped_feature_names_by_head={
            'stage': ('stage',),
            # TODO handle Nana
            'ego_character': ('ego_character',),
            'opponent_character': ('opponent_character',),
            'ego_action': ('ego_action',),
            'opponent_action': ('opponent_action',),
            'controller': ('controller',),
        },
        input_shapes_by_head={
            'gamestate': (2 * 9,),  # 2x for ego and opponent
            'controller': (get_target_config().target_size,),
            'stage': (STAGE_EMBEDDING_DIM,),
            'ego_character': (CHARACTER_EMBEDDING_DIM,),
            'opponent_character': (CHARACTER_EMBEDDING_DIM,),
            'ego_action': (ACTION_EMBEDDING_DIM,),
            'opponent_action': (ACTION_EMBEDDING_DIM,),
        },
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


def get_target_config() -> TargetConfig:
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
            'main_stick': (len(STICK_CENTERS),),
            'c_stick': (len(STICK_CENTERS),),
            'shoulder': (len(SHOULDER_CENTERS),),
            'buttons': (len(INCLUDED_BUTTONS),),
        },
    )


@attr.s(auto_attribs=True)
class PostprocessConfig:
    """Configuration for how we convert model predictions to controller inputs."""

    transformation_by_controller_input: Dict[str, Callable[[TensorDict], Any]]


def get_postprocess_config() -> PostprocessConfig:
    return PostprocessConfig(
        transformation_by_controller_input={
            'main_stick': sample_main_stick_coarse,
            'c_stick': sample_c_stick_coarse,
            'buttons': sample_single_button,
            'shoulder': sample_analog_shoulder_coarse,
        }
    )


def preprocess_target_features(
    sample_T: TensorDict, ego: Player, target_config: TargetConfig
) -> TensorDict:
    processed_features: Dict[str, torch.Tensor] = {}

    for feature_name, transformation in target_config.transformation_by_target.items():
        processed_features[feature_name] = transformation(sample_T, ego)

    return TensorDict(processed_features, batch_size=sample_T.batch_size)


def concat_controller_inputs(
    sample_T: TensorDict, ego: Player, target_config: TargetConfig
) -> torch.Tensor:
    controller_feats = cast(
        Mapping[str, torch.Tensor], preprocess_target_features(sample_T, ego, target_config)
    )
    return torch.cat(tuple(controller_feats.values()), dim=-1)
