#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from typing import Optional

from loguru import logger

from fax.config import create_parser, parse_args
from fax.dataprep.slp_reader import parse_eval_replay, EvalReplayRecord


def parse_eval_replays(cfg):
    random.seed(cfg.base.seed)

    # first pass: possibly clean up invalid replays
    replays_path = cfg.paths.replays / 'cle' / f'{cfg.eval.p1_type}_vs_{cfg.eval.p2_type}'
    replay_paths = list(replays_path.glob('*.slp'))
    for replay_path in replay_paths:
        try:
            replay: Optional[EvalReplayRecord] = parse_eval_replay(replay_path)
        except Exception as e:
            logger.error(f'Failed to parse replay {replay_path}: {e}')
            replay_path.unlink(missing_ok=True)
            continue
        if replay is None:
            logger.warning(f'Invalid replay: {replay_path}, removing')
            replay_path.unlink(missing_ok=True)
        else:  # valid replay
            continue

    # second pass: sample up to 100 valid replays
    p1wins = 0
    p1stocks_taken = 0
    p2stocks_taken = 0
    replay_paths = list(replays_path.glob('*.slp'))
    n_games = len(replay_paths)
    if n_games > 100:
        logger.info(f'Sampling 100 replays from {len(replay_paths)} available')
    replay_paths = random.sample(replay_paths, k=min(100, n_games))
    n_games = len(replay_paths)
    for replay_path in replay_paths:
        replay: Optional[EvalReplayRecord] = parse_eval_replay(replay_path)
        assert replay is not None, f'Replay {replay_path} should be valid after first pass'
        if replay.p1stocks > replay.p2stocks:
            p1wins += 1
        p1stocks_taken += 4 - replay.p2stocks
        p2stocks_taken += 4 - replay.p1stocks

    # log stats
    logger.info(f'valid games: {n_games}, P1 wins: {p1wins}')
    logger.info(f'p1 win rate: {100 * p1wins / n_games:.1f}%')
    logger.info(f'avg. stocks taken per game:')
    logger.info(f'  p1: {p1stocks_taken / n_games:.2f}')
    logger.info(f'  p2: {p2stocks_taken / n_games:.2f}')
    try:
        logger.info(f'p1 "k/d" ratio: {p1stocks_taken / p2stocks_taken:.2f}')
    except ZeroDivisionError:
        logger.info(f'p1 "k/d" ratio: inf (p2 took 0 stocks)')


if __name__ == '__main__':
    exposed_args = {'PATHS': 'replays', 'BASE': 'debug seed', 'EVAL': 'p1-type p2-type'}
    parser = create_parser(exposed_args)
    cfg = parse_args(parser.parse_args(), __file__)
    parse_eval_replays(cfg)
