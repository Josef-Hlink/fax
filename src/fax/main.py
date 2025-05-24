#!/usr/bin/env python3

from loguru import logger

from fax.utils import generate_random_inputs
from fax.emulator_helper import EmulatorManager, find_open_udp_ports
from fax.gamestate_utils import extract_eval_gamestate_as_tensordict
from fax.paths import PATHS


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
        i += 1
        logger.debug(
            f'>> {i}, p1: ({float(td["p1_position_x"]):.2f}, {float(td["p1_position_y"]):.2f})'
            + f' p2: ({float(td["p2_position_x"]):.2f}, {float(td["p2_position_y"]):.2f})'
        )
        gs = gs_generator.send((generate_random_inputs(), generate_random_inputs()))


if __name__ == '__main__':
    main()
