# -*- coding: utf-8 -*-

"""
Schema definitions for the MDS dataset, including data types for each column.

Largely copied from https://github.com/ericyuegu/hal
"""

import numpy as np
from numpy.typing import DTypeLike

# Define a mapping of column names to NumPy data types.
NP_TYPE_BY_COLUMN: dict[str, DTypeLike] = {
    'replay_uuid': np.int32,
    'frame': np.int32,
    'stage': np.int32,
    # Player 1
    'p1_port': np.int32,
    'p1_character': np.int32,
    'p1_position_x': np.float32,
    'p1_position_y': np.float32,
    'p1_percent': np.float32,
    'p1_shield_strength': np.float32,
    'p1_stock': np.int32,
    'p1_facing': np.int32,
    'p1_action': np.int32,
    'p1_action_frame': np.int32,
    'p1_invulnerable': np.int32,
    'p1_invulnerability_left': np.int32,
    'p1_hitlag_left': np.int32,
    'p1_hitstun_left': np.int32,
    'p1_jumps_left': np.int32,
    'p1_on_ground': np.int32,
    'p1_speed_air_x_self': np.float32,
    'p1_speed_y_self': np.float32,
    'p1_speed_x_attack': np.float32,
    'p1_speed_y_attack': np.float32,
    'p1_speed_ground_x_self': np.float32,
    'p1_ecb_bottom_x': np.float32,
    'p1_ecb_bottom_y': np.float32,
    'p1_ecb_top_x': np.float32,
    'p1_ecb_top_y': np.float32,
    'p1_ecb_left_x': np.float32,
    'p1_ecb_left_y': np.float32,
    'p1_ecb_right_x': np.float32,
    'p1_ecb_right_y': np.float32,
    # Player 2
    'p2_port': np.int32,
    'p2_character': np.int32,
    'p2_position_x': np.float32,
    'p2_position_y': np.float32,
    'p2_percent': np.float32,
    'p2_shield_strength': np.float32,
    'p2_stock': np.int32,
    'p2_facing': np.int32,
    'p2_action': np.int32,
    'p2_action_frame': np.int32,
    'p2_invulnerable': np.int32,
    'p2_invulnerability_left': np.int32,
    'p2_hitlag_left': np.int32,
    'p2_hitstun_left': np.int32,
    'p2_jumps_left': np.int32,
    'p2_on_ground': np.int32,
    'p2_speed_air_x_self': np.float32,
    'p2_speed_y_self': np.float32,
    'p2_speed_x_attack': np.float32,
    'p2_speed_y_attack': np.float32,
    'p2_speed_ground_x_self': np.float32,
    'p2_ecb_bottom_x': np.float32,
    'p2_ecb_bottom_y': np.float32,
    'p2_ecb_top_x': np.float32,
    'p2_ecb_top_y': np.float32,
    'p2_ecb_left_x': np.float32,
    'p2_ecb_left_y': np.float32,
    'p2_ecb_right_x': np.float32,
    'p2_ecb_right_y': np.float32,
    # Player 1 controller
    'p1_button_a': np.int32,
    'p1_button_b': np.int32,
    'p1_button_x': np.int32,
    'p1_button_y': np.int32,
    'p1_button_z': np.int32,
    'p1_button_start': np.int32,
    'p1_button_d_up': np.int32,
    'p1_button_l': np.int32,
    'p1_button_r': np.int32,
    'p1_main_stick_x': np.float32,
    'p1_main_stick_y': np.float32,
    'p1_c_stick_x': np.float32,
    'p1_c_stick_y': np.float32,
    'p1_l_shoulder': np.float32,
    'p1_r_shoulder': np.float32,
    # Player 2 controller
    'p2_button_a': np.int32,
    'p2_button_b': np.int32,
    'p2_button_x': np.int32,
    'p2_button_y': np.int32,
    'p2_button_z': np.int32,
    'p2_button_start': np.int32,
    'p2_button_d_up': np.int32,
    'p2_button_l': np.int32,
    'p2_button_r': np.int32,
    'p2_main_stick_x': np.float32,
    'p2_main_stick_y': np.float32,
    'p2_c_stick_x': np.float32,
    'p2_c_stick_y': np.float32,
    'p2_l_shoulder': np.float32,
    'p2_r_shoulder': np.float32,
}

# The first projectiles on screen, in the order the game lists them. An empty slot has type -1.
N_PROJECTILE_SLOTS = 8
PROJECTILE_FIELDS: dict[str, DTypeLike] = {
    'type': np.int32,  # value of melee.ProjectileType, 255 for one libmelee does not know
    'subtype': np.int32,
    'owner': np.int32,  # 1 or 2 for the player it belongs to, 0 for neither
    'frame': np.int32,  # how long it has been out
    'position_x': np.float32,
    'position_y': np.float32,
    'speed_x': np.float32,
    'speed_y': np.float32,
}
for _slot in range(N_PROJECTILE_SLOTS):
    for _field, _dtype in PROJECTILE_FIELDS.items():
        NP_TYPE_BY_COLUMN[f'projectile{_slot}_{_field}'] = _dtype

# Netplay rank of each player, constant over a replay; 0 when unknown (as in evaluation games)
RANKS = ('Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond', 'Master', 'Grandmaster')
RANK_INDEX = {rank: i for i, rank in enumerate(RANKS, start=1)}
NP_TYPE_BY_COLUMN['p1_rank'] = np.int32
NP_TYPE_BY_COLUMN['p2_rank'] = np.int32

# What the pools keep and the agents don't use: 42% of a replay's bytes
UNUSED_COLUMNS = frozenset(
    [
        f'projectile{slot}_{field}'
        for slot in range(N_PROJECTILE_SLOTS)
        for field in PROJECTILE_FIELDS
    ]
    + ['p1_rank', 'p2_rank']
)

# Create a mapping of column names to human-readable dtype strings.
MDS_DTYPE_STR_BY_COLUMN = {
    name: f'ndarray:{np.dtype(dtype).name}' for name, dtype in NP_TYPE_BY_COLUMN.items()
}
