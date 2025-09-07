# -*- coding: utf-8 -*-

"""
This script will index a directory of .slp files into an SQLite database.
After indexing, it will allow for querying the files for building the MDS dataset
for training our different agents, as well as for data analysis for the paper.
"""

import copy
import os
import json
from itertools import islice
from typing import Optional
from pathlib import Path
from multiprocessing import Pool, cpu_count

import attr
import sqlite3
from tqdm import tqdm
from peppi_py import Game, read_slippi
from melee import Stage, to_internal_stage, Character

from .peppi_schema import peppi_character_to_internal


def parse_ranked_replays(slp_dir: Path, n: int = -1, debug: bool = False) -> None:
    print(f'Indexing .slp files in {slp_dir}')
    print(debug)
    misses = 0
    iterator = slp_dir.rglob('*.slp')
    if n > 0:
        total = min(n, sum(1 for _ in slp_dir.rglob('*.slp')))
        iterator = islice(iterator, n)
    elif n == -1:
        total = sum(1 for _ in slp_dir.rglob('*.slp'))
        iterator = slp_dir.rglob('*.slp')
    else:
        raise ValueError('n must be -1 (process all files) or a positive integer')
    if not debug:  # wrap iterator in tqdm for sleeker UI
        iterator = tqdm(iterator, desc='poop', total=total, disable=debug)
    for slp_path in iterator:
        try:
            record = parse_ranked_replay(slp_path)

            if debug:
                for field in attr.fields(ReplayRecord):
                    key = field.name
                    value = getattr(record, key)
                    if key == 'stage':
                        print(f'  {key}: {get_stage_name(value)} ({value})')
                    elif key in ('p1c', 'p2c'):
                        print(f'  {key}: {get_character_name(value)} ({value})')
                    else:
                        print(f'  {key}: {value}')
        # do something with record
        except Exception as e:
            # print(f'Error parsing {slp_path}: {e}')
            misses += 1
            # keep track of raising files in database too
            continue
    print(f'Finished indexing. Missed {misses} files.')


@attr.s(auto_attribs=True, slots=True)
class ReplayRecord:
    file_name: str
    stage: int
    p1c: int
    p2c: int
    winner: int


@attr.s(auto_attribs=True, slots=True)
class RankedReplayRecord(ReplayRecord):
    p1r: str
    p2r: str


def parse_replay(slp_path: Path) -> ReplayRecord:
    game: Game = read_slippi(str(slp_path), skip_frames=True)
    # assert that game ended correctly
    if game.end.players is None:
        raise ValueError('game not ended properly')
    record = ReplayRecord(
        file_name=str(slp_path.name),
        stage=game.start.stage,
        p1c=game.start.players[0].character,
        p2c=game.start.players[1].character,
        winner=1 if game.end.players[0].placement < game.end.players[1].placement else 2,
    )
    return record


def parse_ranked_replay(slp_path: Path) -> RankedReplayRecord:
    game: Game = read_slippi(str(slp_path), skip_frames=True)
    # assert that both players are playing on netplay
    if game.start.players[0].netplay is None or game.start.players[1].netplay is None:
        raise ValueError('not a netplay replay')
    base_record = parse_replay(slp_path)
    return RankedReplayRecord(
        **attr.asdict(base_record),
        p1r=game.start.players[0].netplay.name.replace(' Player', ''),
        p2r=game.start.players[1].netplay.name.replace(' Player', ''),
    )


def get_stage_name(slp_id: int) -> str:
    try:
        return Stage(to_internal_stage(slp_id)).name
    except ValueError:
        return f'UNKNOWN_STAGE_{slp_id}'


def get_character_name(slp_id: int) -> str:
    try:
        return Character(peppi_character_to_internal(slp_id)).name
    except ValueError:
        return f'UNKNOWN_CHARACTER_{slp_id}'
