# -*- coding: utf-8 -*-

"""
This script will index a directory of .slp files into an SQLite database.
After indexing, it will allow for querying the files for building the MDS dataset
for training our different agents, as well as for data analysis for the paper.
"""

from itertools import islice
from pathlib import Path

import attr
import sqlite3
from tqdm import tqdm
from peppi_py import Game, read_slippi

from fax.constants import STAGE_ID_TO_NAME, CHARACTER_ID_TO_NAME


def parse_ranked_replays(slp_dir: Path, db_path: Path, n: int = -1, debug: bool = False) -> None:
    """Parse .slp files in the given directory and index them into an SQLite database.
    Args:
        slp_dir: Directory containing .slp files to index.
        db_path: Path to the SQLite database file.
        n: Number of files to process. Default is -1 (process all files).
        debug: If True, print debug information.
    """
    slp_dir = slp_dir.expanduser().resolve()
    assert slp_dir.is_dir(), f'{slp_dir} is not a directory'
    assert any(slp_dir.rglob('*.slp')), f'No .slp files found in {slp_dir}'

    db_path = db_path.expanduser().resolve()
    init_database(db_path)

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
        iterator = tqdm(
            iterator, desc=f'Indexing .slp files in {slp_dir}', total=total, disable=debug
        )
    else:
        print(f'Indexing .slp files in {slp_dir}...')

    for slp_path in iterator:
        try:
            record = parse_ranked_replay(slp_path)

            if debug:
                for field in attr.fields(ReplayRecord):
                    key = field.name
                    value = getattr(record, key)
                    if key == 'stage':
                        print(f'  {key}: {STAGE_ID_TO_NAME[value]} ({value})')
                    elif key in ('p1c', 'p2c'):
                        print(f'  {key}: {CHARACTER_ID_TO_NAME[value]} ({value})')
                    else:
                        print(f'  {key}: {value}')
            insert_ranked_replay(db_path, record)
        except Exception as e:
            if debug:
                print(f'Failed to parse {slp_path}: {e}')
            insert_parse_error(db_path, str(slp_path.name), str(e))
    return


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


def init_database(db_path: Path) -> None:
    """Initialize the SQLite database with the required schema."""
    if db_path.exists():
        raise FileExistsError(
            f'{db_path} already exists. Please delete it first if you want to overwrite it.'
        )
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # table for ranked replays
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ranked_replays (
            id INTEGER PRIMARY KEY,
            file_name TEXT UNIQUE NOT NULL,
            stage INTEGER NOT NULL,
            p1c INTEGER NOT NULL,
            p2c INTEGER NOT NULL,
            winner INTEGER NOT NULL,
            p1r TEXT NOT NULL,
            p2r TEXT NOT NULL
        )
    """)
    # table for files that failed to parse
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parse_errors (
            id INTEGER PRIMARY KEY,
            file_name TEXT UNIQUE NOT NULL,
            error_message TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    return


def insert_ranked_replay(db_path: Path, record: RankedReplayRecord) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO ranked_replays (file_name, stage, p1c, p2c, winner, p1r, p2r)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
        (
            record.file_name,
            record.stage,
            record.p1c,
            record.p2c,
            record.winner,
            record.p1r,
            record.p2r,
        ),
    )
    conn.commit()
    conn.close()
    return


def insert_parse_error(db_path: Path, file_name: str, error_message: str) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO parse_errors (file_name, error_message)
        VALUES (?, ?)
    """,
        (file_name, error_message),
    )
    conn.commit()
    conn.close()
    return
