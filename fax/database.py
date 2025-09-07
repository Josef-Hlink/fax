# -*- coding: utf-8 -*-

"""
This module houses the DataBase class for indexing and querying .slp files.
"""

from pathlib import Path

import sqlite3

from fax.slp_reader import ReplayRecord


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
                p2rank TEXT NOT NULL
            )
        """)
        self.conn.commit()
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
        return

    def insert_replay(self, record: ReplayRecord) -> None:
        """Insert a ReplayRecord into the ranked_replays table."""
        assert record.p1rank is not None and record.p2rank is not None, (
            'Ranks must be provided to insert replay'
        )
        fields = ('file_name', 'stage', 'p1char', 'p2char', 'winner', 'p1rank', 'p2rank')
        self.cursor.execute(
            f"""
            INSERT OR IGNORE INTO ranked_replays ({', '.join(fields)})
            VALUES ({', '.join(['?' for _ in fields])})
        """,
            tuple(map(lambda f: getattr(record, f), fields)),
        )
        self.conn.commit()
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

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
