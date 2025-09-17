# -*- coding: utf-8 -*-

"""
This module houses functions for reading and parsing .slp files into
ReplayRecord instances for easy access to relevant fields using peppi_py.
"""

from pathlib import Path
from typing import Any, List, Optional

import attr
import numpy as np
import pyarrow as pa
from loguru import logger
from peppi_py import Game, read_slippi

FOX = 2  # character ID for Fox


@attr.s(auto_attribs=True, slots=True)
class TrainReplayRecord:
    archive: str
    bucket: int
    file_name: str
    stage: int
    p1char: int
    p2char: int
    winner: int
    p1stocks: int
    p2stocks: int
    n_frames: int
    p1rank: Optional[str]
    p2rank: Optional[str]


@attr.s(auto_attribs=True, slots=True)
class EvalReplayRecord:
    file_name: str
    stage: int
    p1char: int
    p2char: int
    winner: int
    p1stocks: int
    p2stocks: int
    n_frames: int
    cpu_lvl: Optional[int] = None


def parse_train_replay(
    file: Path, arch: str, bucket_counts: List[int], bucket_limit: int
) -> Optional[TrainReplayRecord]:
    # first parse minimally to check some cheap early exit conditions
    try:
        game: Game = read_slippi(file.as_posix(), skip_frames=True)
    except Exception as e:
        logger.debug(f'Failed to read {file.name}: {e}')
        return

    # determine potential destination directory based on number of foxes
    # stop early if already enough samples of this category
    # we handle this first, as it's the most one most likely to hit
    # (we fill up nofox and onefox rather quickly)
    try:
        p1char = game.start.players[0].character
        p2char = game.start.players[1].character
    except IndexError:
        logger.debug(f'Game in {file.name} does not have two players')
        return
    bucket = (p1char == FOX) + (p2char == FOX)  # 0, 1, or 2 foxes
    if bucket_counts[bucket] >= bucket_limit:
        return

    # stop early if not both players started with 4 stocks
    if not (game.start.players[0].stocks == 4 and game.start.players[1].stocks == 4):
        return

    # check if game was ended properly (no try guard here, we know the file is readable)
    game = read_slippi(file.as_posix(), skip_frames=False)
    assert game.frames is not None and len(game.frames.ports) == 2, 'not a valid 1v1 game'
    p1stocks = as_int(game.frames.ports[0].leader.post.stocks[-1])
    p2stocks = as_int(game.frames.ports[1].leader.post.stocks[-1])
    if not (p1stocks == 0 or p2stocks == 0):
        logger.debug(f'Game in {file.name} did not end properly')
        return

    # also find netplay ranks if available
    ranks = [
        p.netplay.name.replace(' Player', '') if p.netplay else None for p in game.start.players
    ]

    # create and return record
    return TrainReplayRecord(
        archive=arch,
        bucket=bucket,
        file_name=file.name,
        stage=game.start.stage,
        p1char=p1char,
        p2char=p2char,
        winner=1 if p2stocks == 0 else 2,
        p1stocks=p1stocks,
        p2stocks=p2stocks,
        n_frames=len(game.frames.ports[0].leader.post.stocks),
        p1rank=ranks[0],
        p2rank=ranks[1],
    )


def parse_eval_replay(file: Path) -> Optional[EvalReplayRecord]:
    """Read contents of a .slp file and return an EvalReplayRecord."""

    try:
        game: Game = read_slippi(file.as_posix(), skip_frames=True)
    except Exception as e:
        logger.warning(f'Failed to read {file.name}: {e}')
        return

    # check if game was ended properly
    assert game.frames is not None and len(game.frames.ports) == 2, 'not a valid 1v1 game'
    p1stocks = as_int(game.frames.ports[0].leader.post.stocks[-1])
    p2stocks = as_int(game.frames.ports[1].leader.post.stocks[-1])
    if not (p1stocks == 0 or p2stocks == 0):
        logger.debug(f'Game in {file.name} did not end properly')

    # check if we have a CPU player
    cpu_lvl = None
    if game.start.players[1].type.name == 'CPU':
        cpu_lvl = game.start.players[1].type.value

    # create and return record
    return EvalReplayRecord(
        file_name=file.name,
        stage=game.start.stage,
        p1char=game.start.players[0].character,
        p2char=game.start.players[1].character,
        winner=1 if p2stocks == 0 else 2,
        p1stocks=p1stocks,
        p2stocks=p2stocks,
        n_frames=len(game.frames.ports[0].leader.post.stocks),
        cpu_lvl=cpu_lvl,
    )


def as_int(x: Any) -> int:
    """Convert Arrow/NumPy scalars or plain numbers to a Python int."""
    if pa is not None and isinstance(x, pa.Scalar):
        return int(x.as_py())
    if isinstance(x, np.generic):
        return int(x.item())
    return int(x)
