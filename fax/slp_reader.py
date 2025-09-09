#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module houses functions for reading and parsing .slp files into
ReplayRecord instances for easy access to relevant fields using peppi_py.
Example usage in __main__ at the bottom.
"""

from pathlib import Path
from typing import Any, Optional

import attr
import numpy as np
import pyarrow as pa
from loguru import logger
from peppi_py import Game, read_slippi

from fax.constants import CHARACTER_ID_TO_NAME, STAGE_ID_TO_NAME
from fax.utils import debug_enabled, timed


@attr.s(auto_attribs=True, slots=True)
class ReplayRecord:
    file_name: str
    stage: int
    p1char: int
    p2char: int
    # possible to derive from stocks left, but inexpensive to store
    winner: int
    # cpu level is p2 is cpu, None if human vs human
    cpu_lvl: Optional[int] = None
    # stocks left at end of game, None if not parsed
    p1stocks: Optional[int] = None
    p2stocks: Optional[int] = None
    # netplay ranks
    p1rank: Optional[str] = None
    p2rank: Optional[str] = None


@timed
def parse_replay(
    slp_path: Path,
    parse_stocks: bool = False,
    parse_ranks: bool = False,
) -> ReplayRecord:
    """Read contents of a .slp file and return a ReplayRecord."""

    skip_frames = not parse_stocks  # only need frames if parsing stocks
    game: Game = read_slippi(slp_path.as_posix(), skip_frames=skip_frames)

    assert game.end.players is not None, 'game not ended properly'
    record = ReplayRecord(
        file_name=str(slp_path.name),
        stage=game.start.stage,
        p1char=game.start.players[0].character,
        p2char=game.start.players[1].character,
        winner=1 if game.end.players[0].placement < game.end.players[1].placement else 2,
    )

    if game.start.players[1].type.name == 'CPU':
        record.cpu_lvl = game.start.players[1].type.value

    if parse_stocks:
        assert game.frames is not None, 'frames not parsed'
        assert len(game.frames.ports) == 2, 'not a 1v1 game'
        record.p1stocks = as_int(game.frames.ports[0].leader.post.stocks[-1])
        record.p2stocks = as_int(game.frames.ports[1].leader.post.stocks[-1])

    if parse_ranks:
        assert game.start.players[0].netplay is not None, 'not a netplay replay'
        assert game.start.players[1].netplay is not None, 'not a netplay replay'
        record.p1rank = game.start.players[0].netplay.name.replace(' Player', '')
        record.p2rank = game.start.players[1].netplay.name.replace(' Player', '')

    if not debug_enabled():
        return record

    # log all fields if in debug mode
    for field in attr.fields(ReplayRecord):
        key = field.name
        value = getattr(record, key)
        if key == 'stage':
            logger.debug(f'{key}: {STAGE_ID_TO_NAME.get(value, "UNKNOWN")} ({value})')
        elif key in ('p1char', 'p2char'):
            logger.debug(f'{key}: {CHARACTER_ID_TO_NAME.get(value, "UNKNOWN")} ({value})')
        else:
            logger.debug(f'{key}: {value}')

    return record


def as_int(x: Any) -> int:
    """Convert Arrow/NumPy scalars or plain numbers to a Python int."""
    if pa is not None and isinstance(x, pa.Scalar):
        return int(x.as_py())
    if isinstance(x, np.generic):
        return int(x.item())
    return int(x)


if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Parse a .slp replay file.')
    parser.add_argument('slp_path', type=Path, help='Path to the .slp file to parse.')
    parser.add_argument('-s', '--stocks', action='store_true')
    parser.add_argument('-r', '--ranks', action='store_true')
    parser.add_argument('-D', '--debug', action='store_true')
    args = parser.parse_args()

    # set up logging
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if args.debug else 'INFO')
    logger.debug('Debug mode enabled')

    _ = parse_replay(args.slp_path, args.stocks, args.ranks)
    logger.info('completed parsing replay with no errors')
