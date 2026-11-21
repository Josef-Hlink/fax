# -*- coding: utf-8 -*-

"""
This module houses functions for reading and parsing .slp files into
ReplayRecord instances for easy access to relevant fields using peppi_py.
"""

import gzip
from pathlib import Path
from typing import Any, Optional, Tuple
from zipfile import ZipFile

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
    member: str  # path inside the archive
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


def extract_member(zip_arch: ZipFile, member: str, dest: Path) -> None:
    """Write one replay from a .zip archive to `dest`, gunzipped if it is stored as .slp.gz."""
    data = zip_arch.read(member)
    if member.endswith('.gz'):
        data = gzip.decompress(data)
    with open(dest, 'wb') as f:
        f.write(data)


def parse_train_replay(
    file: Path, arch: str, member: str
) -> Tuple[Optional[TrainReplayRecord], Optional[str]]:
    """
    Read a replay and return its record, or None and the reason it is not usable for training
    (anything but a 1v1 game that started on 4 stocks and was played out).
    """
    try:
        game: Game = read_slippi(file.as_posix(), skip_frames=False)
    except Exception as e:
        return None, f'unreadable: {e}'

    if len(game.start.players) != 2 or game.frames is None or len(game.frames.ports) != 2:
        return None, 'not a 1v1 game'
    p1char = game.start.players[0].character
    p2char = game.start.players[1].character
    if not (game.start.players[0].stocks == 4 and game.start.players[1].stocks == 4):
        return None, 'did not start on 4 stocks'
    if len(game.frames.id) == 0:
        return None, 'no frames'
    p1stocks = as_int(game.frames.ports[0].leader.post.stocks[-1])
    p2stocks = as_int(game.frames.ports[1].leader.post.stocks[-1])
    if not (p1stocks == 0 or p2stocks == 0):
        return None, 'not played out'

    # netplay ranks, if available
    ranks = [
        p.netplay.name.replace(' Player', '') if p.netplay else None for p in game.start.players
    ]

    record = TrainReplayRecord(
        archive=arch,
        bucket=(p1char == FOX) + (p2char == FOX),  # 0, 1, or 2 foxes
        file_name=Path(member).name.removesuffix('.gz'),
        member=member,
        stage=game.start.stage,
        p1char=p1char,
        p2char=p2char,
        winner=1 if p2stocks == 0 else 2,
        p1stocks=p1stocks,
        p2stocks=p2stocks,
        n_frames=count_frames(game),
        p1rank=ranks[0],
        p2rank=ranks[1],
    )
    return record, None


def parse_eval_replay(file: Path) -> Optional[EvalReplayRecord]:
    """Read contents of a .slp file and return an EvalReplayRecord."""

    try:
        game: Game = read_slippi(file.as_posix(), skip_frames=False)
    except Exception as e:
        logger.warning(f'Failed to read {file.name}: {e}')
        return

    assert game.frames is not None and len(game.frames.ports) == 2, 'not a valid 1v1 game'
    p1stocks = as_int(game.frames.ports[0].leader.post.stocks[-1])
    p2stocks = as_int(game.frames.ports[1].leader.post.stocks[-1])

    # determine the winner; a game is over when someone is out of stocks or the timer ran out,
    # in the latter case nobody is at 0 stocks and only the game's own placements tell who won
    placements = None
    if game.end is not None and game.end.players is not None and len(game.end.players) == 2:
        placements = [p.placement for p in game.end.players]
    if p1stocks == 0 or p2stocks == 0:
        winner = 1 if p2stocks == 0 else 2
    elif placements is not None and placements[0] != placements[1]:
        winner = 1 if placements[0] < placements[1] else 2
    else:
        logger.warning(f'Game in {file.name} did not end properly')
        return

    # check if we have a CPU player
    cpu_lvl = None
    if game.start.players[1].type.name == 'CPU':
        cpu_lvl = game.start.players[1].cpu_level

    # create and return record
    return EvalReplayRecord(
        file_name=file.name,
        stage=game.start.stage,
        p1char=game.start.players[0].character,
        p2char=game.start.players[1].character,
        winner=winner,
        p1stocks=p1stocks,
        p2stocks=p2stocks,
        n_frames=count_frames(game),
        cpu_lvl=cpu_lvl,
    )


def count_frames(game: Game) -> int:
    """Number of distinct frames in a game; netplay replays repeat the frames that were rolled back."""
    assert game.frames is not None
    return as_int(game.frames.id[-1]) - as_int(game.frames.id[0]) + 1


def as_int(x: Any) -> int:
    """Convert Arrow/NumPy scalars or plain numbers to a Python int."""
    if pa is not None and isinstance(x, pa.Scalar):
        return int(x.as_py())
    if isinstance(x, np.generic):
        return int(x.item())
    return int(x)
