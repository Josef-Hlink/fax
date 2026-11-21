# -*- coding: utf-8 -*-

"""
This module houses the DataBase class for indexing and querying .slp files.
Example usage of some of the queries for an existing db in __main__ at the bottom.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

from loguru import logger

from fax.dataprep.slp_reader import TrainReplayRecord
from fax.utils.constants import CHARACTER_NAME_TO_ID


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

    FIELDS = (
        'archive',
        'bucket',
        'file_name',
        'member',
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

    def create_replays_table(self) -> None:
        """Create the tables, or bring those of an older index up to date."""
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
        columns = {row[1] for row in self.cursor.execute('PRAGMA table_info(replays)')}
        if 'member' not in columns:
            # where the replay is inside its archive; older indexes only kept the file name
            self.cursor.execute('ALTER TABLE replays ADD COLUMN member TEXT')
        if 'split' not in columns:
            # 'train' or 'val' once a conversion has picked the replay, so it stays there
            self.cursor.execute('ALTER TABLE replays ADD COLUMN split TEXT')
        # members that are not usable for training, so a later run does not parse them again
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS skipped (
                archive TEXT NOT NULL,
                member TEXT NOT NULL,
                reason TEXT NOT NULL,
                PRIMARY KEY (archive, member)
            )
        """)
        self.cursor.execute('CREATE INDEX IF NOT EXISTS replays_bucket ON replays (bucket)')
        self.conn.commit()

    def insert_replays(self, records: Sequence[TrainReplayRecord]) -> None:
        """
        Insert TrainReplayRecords into the replays table. A replay that an older index already
        has only gets its place in the archive filled in.
        """
        self.cursor.executemany(
            f"""
            INSERT INTO replays ({', '.join(self.FIELDS)})
            VALUES ({', '.join(['?' for _ in self.FIELDS])})
            ON CONFLICT (file_name) DO UPDATE SET member = excluded.member
        """,
            [tuple(getattr(record, f) for f in self.FIELDS) for record in records],
        )
        self.conn.commit()

    def insert_skipped(self, rows: Sequence[Tuple[str, str, str]]) -> None:
        """Remember (archive, member, reason) of members that are not usable."""
        self.cursor.executemany('INSERT OR IGNORE INTO skipped VALUES (?, ?, ?)', rows)
        self.conn.commit()

    def known_members(self, archive: str) -> Set[str]:
        """Members of an archive that are indexed or skipped already."""
        known = set()
        for table in ('replays', 'skipped'):
            self.cursor.execute(f'SELECT member FROM {table} WHERE archive = ?', (archive,))
            known.update(row[0] for row in self.cursor.fetchall())
        return known

    def replays_in_bucket(self, bucket: int) -> List[Dict[str, Any]]:
        """All replays of a bucket, as dicts with the fields a conversion needs."""
        fields = ('file_name', 'archive', 'member', 'p1rank', 'p2rank', 'split')
        self.cursor.execute(
            f'SELECT {", ".join(fields)} FROM replays WHERE bucket = ? ORDER BY file_name',
            (bucket,),
        )
        return [dict(zip(fields, row)) for row in self.cursor.fetchall()]

    def set_split(self, file_names: Sequence[str], split: str) -> None:
        """Record that these replays belong to the train or the val split."""
        self.cursor.executemany(
            'UPDATE replays SET split = ? WHERE file_name = ?', [(split, f) for f in file_names]
        )
        self.conn.commit()

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
            SELECT file_name FROM replays
            WHERE (p1char = ? AND p2char = ?)
               OR (p1char = ? AND p2char = ?)
        """,
            (p1char, p2char, p2char, p1char),
        )
        return [row[0] for row in self.cursor.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
