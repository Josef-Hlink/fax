#!/usr/bin/env python3

import random

import melee
from loguru import logger

from fax.emulator_helper import EmulatorManager, find_open_udp_ports, send_controller_inputs
from fax.gamestate_utils import extract_eval_gamestate_as_tensordict
from fax.paths import PATHS


def generate_random_inputs():
    return {
        'main_stick': (random.random(), random.random()),
        'c_stick': (random.random(), random.random()),
        'shoulder': 0,
        'buttons': {
            'BUTTON_A': random.randint(0, 1),
            'BUTTON_B': random.randint(0, 1),
            'BUTTON_X': random.randint(0, 1),
            'BUTTON_Y': random.randint(0, 1),
            'BUTTON_Z': random.randint(0, 1),
            'BUTTON_L': random.randint(0, 1),
            'BUTTON_R': random.randint(0, 1),
        },
    }


def main():
    udp_port = find_open_udp_ports(1)[0]
    emulator_manager = EmulatorManager(
        udp_port=udp_port,
        player='p1',
        replay_dir=PATHS.replays,
        debug=True,
    )

    gs_generator = emulator_manager.run_game()
    gs = next(gs_generator)

    i = 0
    while gs is not None:
        td = extract_eval_gamestate_as_tensordict(gs)
        inputs = generate_random_inputs()
        send_controller_inputs(emulator_manager.ego_controller, inputs)
        i += 1
        logger.debug(
            f'Iteration {i}, {td["p1_position_x"]}, {td["p1_position_y"]}'
        )  # seems to stand still


if __name__ == '__main__':
    main()
