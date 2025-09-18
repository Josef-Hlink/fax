#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pathlib import Path

from loguru import logger

from fax.config import create_parser, parse_args
from fax.utils.emulator_helper import EmulatorManager, Matchup, find_open_udp_ports
from fax.utils.gamestate_utils import extract_eval_gamestate_as_tensordict


def main(emulator_path: Path, replay_dir: Path, iso_path: Path):
    udp_port = find_open_udp_ports(1)[0]
    emulator_manager = EmulatorManager(
        udp_port=udp_port,
        player='p1',
        emulator_path=emulator_path,
        replay_dir=replay_dir,
        opponent_cpu_level=0,
        matchup=Matchup('BATTLEFIELD', 'FOX', 'FOX'),
        debug=True,
    )

    gs_generator = emulator_manager.run_game(iso_path=iso_path)
    gs = next(gs_generator)

    i = 0
    while gs is not None:
        td = extract_eval_gamestate_as_tensordict(gs)
        i += 1
        logger.debug(
            f'>> {i}, p1: ({float(td["p1_position_x"]):.2f}, {float(td["p1_position_y"]):.2f})'
            + f' p2: ({float(td["p2_position_x"]):.2f}, {float(td["p2_position_y"]):.2f})'
        )
        logger.info(f'   p1 stocks: {td["p1_stock"].item()}, p2 stocks: {td["p2_stock"].item()}')
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
    exposed_args = {'PATHS': 'exe replays iso', 'BASE': 'debug'}
    parser = create_parser(exposed_args)
    cfg = parse_args(parser.parse_args(), __file__)
    main(emulator_path=cfg.paths.exe, replay_dir=cfg.paths.replays, iso_path=cfg.paths.iso)
