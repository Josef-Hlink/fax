# -*- coding: utf-8 -*-

"""
This module houses the DataBase class for indexing and querying .slp files.
Example usage of some of the queries for an existing db in __main__ at the bottom.
"""

import sqlite3
from pathlib import Path
from typing import List

from loguru import logger

from fax.constants import CHARACTER_NAME_TO_ID
from fax.dataprep.slp_reader import TrainReplayRecord


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
            CREATE TABLE IF NOT EXISTS replays (
                id INTEGER PRIMARY KEY,
                archive TEXT NOT NULL,
                bucket INTEGER NOT NULL,
                file_name TEXT UNIQUE NOT NULL,
                stage INTEGER NOT NULL,
                p1char INTEGER NOT NULL,
                p2char INTEGER NOT NULL,
                winner INTEGER NOT NULL,
                p1stocks INTEGER NOT NULL,
                p2stocks INTEGER NOT NULL,
                n_frames INTEGER NOT NULL,
                p1rank TEXT,
                p2rank TEXT
            )
        """)
        self.conn.commit()
        logger.debug('Created ranked_replays table')
        return

    def insert_replay(self, record: TrainReplayRecord) -> None:
        """Insert a TrainReplayRecord into the replays table."""
        fields = (
            'archive',
            'bucket',
            'file_name',
            'stage',
            'p1char',
            'p2char',
            'winner',
            'p1stocks',
            'p2stocks',
            'n_frames',
            'p1rank',
            'p2rank',
        )
        self.cursor.execute(
            f"""
            INSERT OR IGNORE INTO replays ({', '.join(fields)})
            VALUES ({', '.join(['?' for _ in fields])})
        """,
            tuple(map(lambda f: getattr(record, f), fields)),
        )
        self.conn.commit()
        logger.debug(f'Inserted replay {record.file_name} into database')
        return

    @property
    def n_replays(self) -> int:
        """Return the number of replays in the database."""
        self.cursor.execute('SELECT COUNT(*) FROM replays')
        return self.cursor.fetchone()[0]

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
            SELECT file_name FROM replays
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
