#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pathlib import Path
from typing import Optional

import attr
from loguru import logger

from fax.config import create_parser, parse_args
from fax.dataprep.slp_reader import parse_eval_replay, EvalReplayRecord
from fax.eval.eval import run_closed_loop_evaluation


def main(cfg):
    p1_weights = cfg.paths.runs / 'coincident-tincan-228'
    # p2_weights = replay_dir / 'dry-bog-213'cfg.paths.runs

    # let the evaluation run for a while to generate replays
    run_closed_loop_evaluation(cfg, p1_weights, 'p1')
    valid_games = 0
    p1wins = 0
    for replay_path in sorted((cfg.paths.replays / 'cle').glob('*.slp')):
        try:
            replay: Optional[EvalReplayRecord] = parse_eval_replay(replay_path)
        except Exception as e:
            logger.error(f'Failed to parse replay {replay_path}: {e}')
            continue
        if replay is not None:
            valid_games += 1
            if replay.p1stocks > replay.p2stocks:
                p1wins += 1
            # for k, v in attr.asdict(replay).items():
            #     logger.info(f'{k}: {v}')
        else:
            logger.warning(f'Invalid replay: {replay_path}, removing')
            replay_path.unlink(missing_ok=True)

    logger.info(f'valid games: {valid_games}, P1 wins: {p1wins}')
    logger.info(f'p1 win rate: {100 * p1wins / valid_games:.1f}%')


if __name__ == '__main__':
    exposed_args = {'PATHS': 'exe runs replays iso', 'BASE': 'debug'}
    parser = create_parser(exposed_args)
    cfg = parse_args(parser.parse_args(), __file__)
    main(cfg)
