# -*- coding: utf-8 -*-

"""
Index:
 - STAGES
 - CHARACTERS
 - PLAYERS
 - ACTIONS
 - CONTROLLER
 - EMBEDDINGS
 - MISC
"""

from typing import Dict, Tuple, Final, Literal
from melee import Stage, Character, Action, to_internal_stage

import numpy as np


##########
# STAGES #
##########

# subset of melee Stage enum: FoD, PS, YS, DL, BF, FD
PEPPI_STAGE_IDS = [0x02, 0x03, 0x08, 0x1C, 0x1F, 0x20]

# peppi and slippi agree on stage ids
peppi_stage_to_internal = to_internal_stage

# id -> name
STAGE_ID_TO_NAME: Dict[int, str] = {
    sid: peppi_stage_to_internal(sid).name
    for sid in PEPPI_STAGE_IDS
    if peppi_stage_to_internal(sid) not in (Stage.NO_STAGE, Stage.RANDOM_STAGE)
}

# name -> id
STAGE_NAME_TO_ID: Dict[str, int] = {
    peppi_stage_to_internal(sid).name: sid
    for sid in PEPPI_STAGE_IDS
    if peppi_stage_to_internal(sid) not in (Stage.NO_STAGE, Stage.RANDOM_STAGE)
}

NUM_STAGES = len(STAGE_ID_TO_NAME)


##############
# CHARACTERS #
##############

# subset of melee Character enum
PEPPI_CHARACTER_IDS = list(range(0x00, 0x1A))  # 0x1A is exclusive


# peppi thinks it's funny to use a wildly different enum from slippi for characters
def peppi_character_to_internal(char_id: int) -> Character:
    """Convert peppi character id to internal melee Character enum."""
    if char_id == 0x00:
        return Character.CPTFALCON
    if char_id == 0x01:
        return Character.DK
    if char_id == 0x02:
        return Character.FOX
    if char_id == 0x03:
        return Character.GAMEANDWATCH
    if char_id == 0x04:
        return Character.KIRBY
    if char_id == 0x05:
        return Character.BOWSER
    if char_id == 0x06:
        return Character.LINK
    if char_id == 0x07:
        return Character.LUIGI
    if char_id == 0x08:
        return Character.MARIO
    if char_id == 0x09:
        return Character.MARTH
    if char_id == 0x0A:
        return Character.MEWTWO
    if char_id == 0x0B:
        return Character.NESS
    if char_id == 0x0C:
        return Character.PEACH
    if char_id == 0x0D:
        return Character.PIKACHU
    if char_id == 0x0E:
        return Character.POPO
    if char_id == 0x0F:
        return Character.JIGGLYPUFF
    if char_id == 0x10:
        return Character.SAMUS
    if char_id == 0x11:
        return Character.YOSHI
    if char_id == 0x12:
        return Character.ZELDA
    if char_id == 0x13:
        return Character.SHEIK
    if char_id == 0x14:
        return Character.FALCO
    if char_id == 0x15:
        return Character.YLINK
    if char_id == 0x16:
        return Character.DOC
    if char_id == 0x17:
        return Character.ROY
    if char_id == 0x18:
        return Character.PICHU
    if char_id == 0x19:
        return Character.GANONDORF
    return Character.UNKNOWN_CHARACTER


# id -> name
CHARACTER_ID_TO_NAME: Dict[int, str] = {
    cid: peppi_character_to_internal(cid).name
    for cid in PEPPI_CHARACTER_IDS
    if peppi_character_to_internal(cid) != Character.UNKNOWN_CHARACTER
}

# name -> id
CHARACTER_NAME_TO_ID: Dict[str, int] = {
    peppi_character_to_internal(cid).name: cid
    for cid in PEPPI_CHARACTER_IDS
    if peppi_character_to_internal(cid) != Character.UNKNOWN_CHARACTER
}

NUM_CHARACTERS = len(CHARACTER_ID_TO_NAME)


###########
# PLAYERS #
###########

Player = Literal['p1', 'p2']
VALID_PLAYERS: Final[tuple[Player, Player]] = ('p1', 'p2')
PLAYER_1_PORT: Final[int] = 1
PLAYER_2_PORT: Final[int] = 2


def get_opponent(player: Player) -> Player:
    return 'p2' if player == 'p1' else 'p1'


###########
# ACTIONS #
###########

ACTION_ID_TO_NAME: Dict[Action, str] = {action: action.name for action in Action}

ACTION_NAME_TO_ID: Dict[str, Action] = {action.name: action for action in Action}

NUM_ACTIONS = len(ACTION_ID_TO_NAME)


##############
# CONTROLLER #
##############

ORIGINAL_BUTTONS: Tuple[str, ...] = (
    'BUTTON_A',
    'BUTTON_B',
    'BUTTON_X',
    'BUTTON_Y',
    'BUTTON_Z',
    'BUTTON_L',
    'BUTTON_R',
)

INCLUDED_BUTTONS: Tuple[str, ...] = (
    'BUTTON_A',
    'BUTTON_B',
    'BUTTON_X',
    'BUTTON_Z',
    'BUTTON_L',
    'NO_BUTTON',
)

TARGET_FEATURES_TO_ONE_HOT_ENCODE: Tuple[str, ...] = ('a', 'b', 'x', 'z', 'l', 'no_button')

SHOULDER_CENTERS: np.ndarray = np.array([0.0, 0.4, 1.0])

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


##############
# EMBEDDINGS #
##############

STAGE_EMBEDDING_DIM: int = 4
CHARACTER_EMBEDDING_DIM: int = 12
ACTION_EMBEDDING_DIM: int = 32


########
# MISC #
########

NP_MASK_VALUE: Final[int] = (1 << 31) - 1
