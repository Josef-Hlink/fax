#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from loguru import logger

from fax.paths import REPLAY_OUTPUT
from fax.emulator_helper import EmulatorManager, find_open_udp_ports
from fax.gamestate_utils import extract_eval_gamestate_as_tensordict


def main():
    udp_port = find_open_udp_ports(1)[0]
    emulator_manager = EmulatorManager(
        udp_port=udp_port,
        player='p1',
        replay_dir=REPLAY_OUTPUT,
        debug=True,
        opponent_cpu_level=0,
    )

    gs_generator = emulator_manager.run_game()
    gs = next(gs_generator)

    i = 0
    while gs is not None:
        td = extract_eval_gamestate_as_tensordict(gs)
        i += 1
        logger.debug(
            f'>> {i}, p1: ({float(td["p1_position_x"]):.2f}, {float(td["p1_position_y"]):.2f})'
            + f' p2: ({float(td["p2_position_x"]):.2f}, {float(td["p2_position_y"]):.2f})'
        )
        gs = gs_generator.send((generate_random_inputs(), generate_random_inputs()))


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


if __name__ == '__main__':
    main()
