# largely copied from https://github.com/ericyuegu/hal

from typing import Dict, Final, Literal, Tuple

import numpy as np
from melee import Action, Character, Stage

NP_MASK_VALUE: Final[int] = (1 << 31) - 1

VALID_PLAYERS: Final[Tuple[str, str]] = ('p1', 'p2')
Player = Literal['p1', 'p2']
PLAYER_1_PORT: Final[int] = 1
PLAYER_2_PORT: Final[int] = 2


def get_opponent(player: Player) -> Player:
    return 'p2' if player == 'p1' else 'p1'


###################
# Gamestate       #
###################

STAGES: Tuple[str, ...] = (
    'FINAL_DESTINATION',
    'BATTLEFIELD',
    'POKEMON_STADIUM',
    'DREAMLAND',
    'FOUNTAIN_OF_DREAMS',
    'YOSHIS_STORY',
)
IDX_BY_STAGE: Dict[Stage, int] = {
    stage: i for i, stage in enumerate(stage for stage in Stage if stage.name in STAGES)
}
IDX_BY_STAGE_STR: Dict[str, int] = {stage.name: i for stage, i in IDX_BY_STAGE.items()}
STAGE_BY_IDX: Dict[int, str] = {i: stage.name for stage, i in IDX_BY_STAGE.items()}

CHARACTERS: Tuple[str, ...] = (
    'MARIO',
    'FOX',
    'CPTFALCON',
    'DK',
    'KIRBY',
    'BOWSER',
    'LINK',
    'SHEIK',
    'NESS',
    'PEACH',
    'POPO',
    'NANA',
    'PIKACHU',
    'SAMUS',
    'YOSHI',
    'JIGGLYPUFF',
    'MEWTWO',
    'LUIGI',
    'MARTH',
    'ZELDA',
    'YLINK',
    'DOC',
    'FALCO',
    'PICHU',
    'GAMEANDWATCH',
    'GANONDORF',
    'ROY',
)
IDX_BY_CHARACTER: Dict[Character, int] = {
    char: i for i, char in enumerate(char for char in Character if char.name in CHARACTERS)
}
IDX_BY_CHARACTER_STR: Dict[str, int] = {char.name: i for char, i in IDX_BY_CHARACTER.items()}
CHARACTER_BY_IDX: Dict[int, str] = {i: char.name for char, i in IDX_BY_CHARACTER.items()}

IDX_BY_ACTION: Dict[Action, int] = {action: i for i, action in enumerate(Action)}
ACTION_BY_IDX: Dict[int, str] = {i: action.name for action, i in IDX_BY_ACTION.items()}

ORIGINAL_BUTTONS: Tuple[str, ...] = (
    'BUTTON_A',
    'BUTTON_B',
    'BUTTON_X',
    'BUTTON_Y',
    'BUTTON_Z',
    'BUTTON_L',
    'BUTTON_R',
)
ORIGINAL_BUTTONS_NO_SHOULDER: Tuple[str, ...] = (
    'BUTTON_A',
    'BUTTON_B',
    'BUTTON_X',
    'BUTTON_Y',
    'BUTTON_Z',
    'NO_BUTTON',
)
INCLUDED_BUTTONS: Tuple[str, ...] = (
    'BUTTON_A',
    'BUTTON_B',
    'BUTTON_X',
    'BUTTON_Z',
    'BUTTON_L',
    'NO_BUTTON',
)
INCLUDED_BUTTONS_NO_SHOULDER: Tuple[str, ...] = (
    'BUTTON_A',
    'BUTTON_B',
    'BUTTON_X',
    'BUTTON_Z',
    'NO_BUTTON',
)


###################
# Embeddings      #
###################

REPLAY_UUID: Tuple[str] = ('replay_uuid',)
FRAME: Tuple[str] = ('frame',)
STAGE: Tuple[str, ...] = ('stage',)
PLAYER_INPUT_FEATURES_TO_EMBED: Tuple[str, ...] = ('character', 'action')
PLAYER_INPUT_FEATURES_TO_NORMALIZE: Tuple[str, ...] = (
    'percent',
    'stock',
    'facing',
    'invulnerable',
    'jumps_left',
    'on_ground',
)
PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE: Tuple[str, ...] = ('shield_strength',)
PLAYER_POSITION: Tuple[str, ...] = (
    'position_x',
    'position_y',
)
# Optional input features
PLAYER_ACTION_FRAME_FEATURES: Tuple[str, ...] = (
    'action_frame',
    'hitlag_left',
    'hitstun_left',
)
PLAYER_SPEED_FEATURES: Tuple[str, ...] = (
    'speed_air_x_self',
    'speed_y_self',
    'speed_x_attack',
    'speed_y_attack',
    'speed_ground_x_self',
)
PLAYER_ECB_FEATURES: Tuple[str, ...] = (
    'ecb_bottom_x',
    'ecb_bottom_y',
    'ecb_top_x',
    'ecb_top_y',
    'ecb_left_x',
    'ecb_left_y',
    'ecb_right_x',
    'ecb_right_y',
)
# Target features
TARGET_FEATURES_TO_ONE_HOT_ENCODE: Tuple[str, ...] = ('a', 'b', 'x', 'z', 'l', 'no_button')

SHOULDER_CENTERS: np.ndarray = np.array([0.0, 0.4, 1.0])
SHOULDER_CENTERS.flags.writeable = False

STICK_CENTERS: np.ndarray = np.array(
    [
        [0.5, 0.5],
        [1.0, 0.5],
        [0.0, 0.5],
        [0.50, 0.0],
        [0.50, 1.0],
        [0.50, 0.25],
        [0.50, 0.75],
        [0.75, 0.5],
        [0.25, 0.5],
        [0.15, 0.15],
        [0.85, 0.15],
        [0.85, 0.85],
        [0.15, 0.85],
        [0.28, 0.93],
        [0.28, 0.07],
        [0.72, 0.07],
        [0.72, 0.93],
        [0.07, 0.28],
        [0.07, 0.72],
        [0.93, 0.72],
        [0.93, 0.28],
    ]
)
STICK_CENTERS.flags.writeable = False

# --- Below determines model input/output head shape ---
# Categorical input embedding sizes
NUM_STAGES: int = len(IDX_BY_STAGE)
NUM_CHARACTERS: int = len(IDX_BY_CHARACTER)
NUM_ACTIONS: int = len(IDX_BY_ACTION)
STAGE_EMBEDDING_DIM: int = 4
CHARACTER_EMBEDDING_DIM: int = 12
ACTION_EMBEDDING_DIM: int = 32
