#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module houses the DataBase class for indexing and querying .slp files.
Example usage of some of the queries for an existing db in __main__ at the bottom.
"""

from typing import List
from pathlib import Path

import sqlite3
from loguru import logger

from fax.slp_reader import ReplayRecord
from fax.constants import CHARACTER_NAME_TO_ID


class DataBase:
    """Class for indexing and querying .slp files in an SQLite database."""

    def __init__(self, db_path: Path) -> None:
        """Initialize the DataBase class.
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        logger.debug(f'Connected to database at {self.db_path}')

    def create_replays_table(self) -> None:
        """Create the replays table in the database."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS ranked_replays (
                id INTEGER PRIMARY KEY,
                file_name TEXT UNIQUE NOT NULL,
                stage INTEGER NOT NULL,
                p1char INTEGER NOT NULL,
                p2char INTEGER NOT NULL,
                winner INTEGER NOT NULL,
                p1rank TEXT NOT NULL,
                p2rank TEXT NOT NULL,
                p1stocks INTEGER NOT NULL,
                p2stocks INTEGER NOT NULL
            )
        """)
        self.conn.commit()
        logger.debug('Created ranked_replays table')
        return

    def create_errors_table(self) -> None:
        """Create the parse_errors table in the database."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS parse_errors (
                id INTEGER PRIMARY KEY,
                file_name TEXT UNIQUE NOT NULL,
                error_message TEXT NOT NULL
            )
        """)
        self.conn.commit()
        logger.debug('Created parse_errors table')
        return

    def insert_replay(self, record: ReplayRecord) -> None:
        """Insert a ReplayRecord into the ranked_replays table."""
        assert record.p1rank is not None and record.p2rank is not None, (
            'Ranks must be provided to insert replay'
        )
        fields = (
            'file_name',
            'stage',
            'p1char',
            'p2char',
            'winner',
            'p1rank',
            'p2rank',
            'p1stocks',
            'p2stocks',
        )
        self.cursor.execute(
            f"""
            INSERT OR IGNORE INTO ranked_replays ({', '.join(fields)})
            VALUES ({', '.join(['?' for _ in fields])})
        """,
            tuple(map(lambda f: getattr(record, f), fields)),
        )
        self.conn.commit()
        logger.debug(f'Inserted replay {record.file_name} into database')
        return

    def insert_error(self, file_name: str, error_message: str) -> None:
        """Insert a parse error into the parse_errors table."""
        self.cursor.execute(
            """
            INSERT OR IGNORE INTO parse_errors (file_name, error_message)
            VALUES (?, ?)
        """,
            (file_name, error_message),
        )
        self.conn.commit()
        logger.debug(f'Inserted parse error for {file_name} into database')
        return

    @property
    def n_replays(self) -> int:
        """Return the number of replays in the database."""
        self.cursor.execute('SELECT COUNT(*) FROM ranked_replays')
        return self.cursor.fetchone()[0]

    @property
    def n_errors(self) -> int:
        """Return the number of parse errors in the database."""
        self.cursor.execute('SELECT COUNT(*) FROM parse_errors')
        return self.cursor.fetchone()[0]

    def get_corrupted_replays(self) -> List[str]:
        """Files that failed to parse, mostly due to unfilled buffers."""
        self.cursor.execute('SELECT file_name FROM parse_errors')
        return [row[0] for row in self.cursor.fetchall()]

    def get_unfinished_replays(self) -> List[str]:
        """Files that parsed but were not finished games (e.g. DCs, timeouts)."""
        self.cursor.execute("""
            SELECT file_name FROM ranked_replays
            WHERE p1stocks > 0 AND p2stocks > 0
        """)
        return [row[0] for row in self.cursor.fetchall()]

    def get_faulty_replays(self) -> List[str]:
        """Return a list of all .slp unfit for training."""
        return self.get_corrupted_replays() + self.get_unfinished_replays()

    def query_character(self, char: int | str) -> List[str]:
        """Query the database for replays involving a specific character (as either player).
        Args:
            char: Character ID (name also supported).
        Returns:
            List of file names involving the character.
        """
        if isinstance(char, str):
            char = CHARACTER_NAME_TO_ID[char.upper()]
        self.cursor.execute(
            """
            SELECT file_name FROM ranked_replays
            WHERE p1char = ? OR p2char = ?
        """,
            (char, char),
        )
        return [row[0] for row in self.cursor.fetchall()]

    def query_matchup(self, p1char: int | str, p2char: int | str) -> List[str]:
        """Query the database for replays of a specific character matchup (order-agnostic).
        Args:
            p1char: Character ID for player 1 (name also supported).
            p2char: Character ID for player 2 (name also supported).
        Returns:
            List of file names matching the character matchup.
        """
        if isinstance(p1char, str):
            p1char = CHARACTER_NAME_TO_ID[p1char.upper()]
        if isinstance(p2char, str):
            p2char = CHARACTER_NAME_TO_ID[p2char.upper()]
        self.cursor.execute(
            """
            SELECT file_name FROM ranked_replays
            WHERE (p1char = ? AND p2char = ?)
               OR (p1char = ? AND p2char = ?)
        """,
            (p1char, p2char, p2char, p1char),
        )
        return [row[0] for row in self.cursor.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()


if __name__ == '__main__':
    import sys
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from fax.paths import PROJECT_ROOT

    parser = ArgumentParser(
        description='Example usage of the DataBase class linked to an existing SQLite database.',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--db_path',
        type=Path,
        default=PROJECT_ROOT / 'db.sqlite',
        help='Path to the SQLite database file.',
    )
    parser.add_argument('-D', '--debug', action='store_true', help='Enable debug mode.')
    args = parser.parse_args()

    # set up logging
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if args.debug else 'INFO')
    logger.debug('Debug mode enabled')

    db = DataBase(args.db_path)
    logger.info(
        f'Database has {db.n_replays} replays,'
        + f' {len(db.get_unfinished_replays())} of which are unfinished'
    )
    logger.info(f'Database has info on {db.n_errors} parse errors in the database')
    fox_games = db.query_character('fox')
    logger.info(f'Found {len(fox_games)} games with fox in the database')
    spacies = db.query_matchup('fox', 'falco')
    logger.info(f'Found {len(spacies)} spacies games in the database')
    fox_dittos = db.query_matchup('fox', 'fox')
    logger.info(f'Found {len(fox_dittos)} fox dittos in the database')
    db.close()
