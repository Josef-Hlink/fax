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
        opponent_cpu_level=9,
        matchup=Matchup('BATTLEFIELD', 'FOX', 'FOX'),
        debug=True,
    )

    gs_generator = emulator_manager.run_game(iso_path=iso_path)
    gs = next(gs_generator)

    i = 0
    game_ended = False
    while True:
        try:
            if game_ended:
                # hacky way to tell the emulator to close when the game ends
                raise Exception('Game ended, exiting loop')
            td = extract_eval_gamestate_as_tensordict(gs)
            i += 1
            if i % 600 == 0:  # log stocks every 10 in-game seconds (600 frames)
                logger.debug(f'f{i}: {(td["p1_stock"].item()), (td["p2_stock"].item())}')
            if td['p1_stock'].item() == 0 or td['p2_stock'].item() == 0:
                game_ended = True
                logger.info(f'Game ended at frame {i}')
            gs = gs_generator.send((generate_random_inputs(), None))
        except Exception:
            logger.error(f'Emulator closed successfully')
            break


empty_inputs = {
    'main_stick': (0.5, 0.5),
    'c_stick': (0.5, 0.5),
    'shoulder': 0.0,
    'buttons': {
        'BUTTON_A': 0,
        'BUTTON_B': 0,
        'BUTTON_X': 0,
        'BUTTON_Y': 0,
        'BUTTON_Z': 0,
        'BUTTON_L': 0,
        'BUTTON_R': 0,
    },
}


def generate_random_inputs():
    return {
        'main_stick': (random.random(), random.random()),
        'c_stick': (random.random(), random.random()),
        'shoulder': 0,
        'buttons': {
            'BUTTON_A': 0.0,
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
