#!/usr/bin/env python3

import argparse

from loguru import logger

from fax.paths import PATHS
from fax.emulator_helper import EmulatorManager, find_open_udp_ports
from fax.gamestate_utils import extract_eval_gamestate_as_tensordict
from fax.utils import (
    controller_output_from_model,
    generate_random_inputs,
    flatten_tensordict,
    compute_reward,
)
from fax.model import SimpleControllerNet


def main(debug: bool = False):
    udp_port = find_open_udp_ports(1)[0]

    model = SimpleControllerNet()

    emulator_manager = EmulatorManager(
        udp_port=udp_port,
        player='p1',
        replay_dir=PATHS.replays,
        debug=True,
        opponent_cpu_level=0,
    )

    gs_generator = emulator_manager.run_game()
    gs = next(gs_generator)

    i = 0
    while gs is not None:
        td = extract_eval_gamestate_as_tensordict(gs)
        i += 1
        features = flatten_tensordict(td)
        model_out = model.forward(features)
        ego_inputs = controller_output_from_model(model_out)
        gs = gs_generator.send((ego_inputs, generate_random_inputs()))
        reward = compute_reward(extract_eval_gamestate_as_tensordict(gs))
        if debug:
            logger.debug(
                f'>> {i}, p1: ({float(td["p1_position_x"]):.2f}, {float(td["p1_position_y"]):.2f})'
                + f' p2: ({float(td["p2_position_x"]):.2f}, {float(td["p2_position_y"]):.2f})'
            )
            logger.debug(f'Features: {features.tolist()}')
            logger.debug(f'Model output: {model_out}')
            logger.debug(f'Ego inputs: {ego_inputs}')
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the FAX model.')
    parser.add_argument('-D', '--debug', action='store_true')
    args = parser.parse_args()

    main(args.debug)
