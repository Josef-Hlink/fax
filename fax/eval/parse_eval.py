#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from typing import Optional

import attr
from loguru import logger

from fax.config import create_parser, parse_args
from fax.dataprep.slp_reader import parse_eval_replay, EvalReplayRecord


def parse_eval_replays(cfg):
    valid_games = 0
    p1wins = 0
    p1stocks_taken = 0
    p2stocks_taken = 0
    replays_path = cfg.paths.replays / 'cle' / f'{cfg.eval.p1_type}_vs_{cfg.eval.p2_type}'
    # for replay_path in random.sample(list(replays_path.glob('*.slp')), k=100):
    for replay_path in replays_path.glob('*.slp'):
        try:
            replay: Optional[EvalReplayRecord] = parse_eval_replay(replay_path)
        except Exception as e:
            logger.error(f'Failed to parse replay {replay_path}: {e}')
            continue
        if replay is not None:
            valid_games += 1
            if replay.p1stocks > replay.p2stocks:
                p1wins += 1
            p1stocks_taken += 4 - replay.p2stocks
            p2stocks_taken += 4 - replay.p1stocks
            # for k, v in attr.asdict(replay).items():
            #     logger.info(f'{k}: {v}')
        else:
            logger.warning(f'Invalid replay: {replay_path}, removing')
            replay_path.unlink(missing_ok=True)

    logger.info(f'valid games: {valid_games}, P1 wins: {p1wins}')
    logger.info(f'p1 win rate: {100 * p1wins / valid_games:.1f}%')
    logger.info(f'stocks taken: P1 {p1stocks_taken}, P2 {p2stocks_taken}')


if __name__ == '__main__':
    exposed_args = {'PATHS': 'replays', 'BASE': 'debug', 'EVAL': 'p1-type p2-type'}
    parser = create_parser(exposed_args)
    cfg = parse_args(parser.parse_args(), __file__)
    parse_eval_replays(cfg)
