from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from ...Indexor.Chunker.Chunk import Chunk

ChunkKey = tuple[str, int]


class DataBaseHandler:
    """
    DataBaseHandler Class Definition

    The StoringManager is the class used to check the database
        and will perform every interaction with it.
    """

    def __init__(self, database: Path) -> None:
        """
            StoringManager Init Method

            Parameters:
                database: path to the database it will interact with
        """
        self.__db = sqlite3.connect(str(database))
        self.__db.execute("PRAGMA foreign_keys = ON")
        self.__pending_hashes: dict[str, str] = {}
        self.__create_tables()

    def __create_tables(self) -> None:
        """Create the database schema when it does not exist."""
        self.__db.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                hash_id TEXT PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                content_hash TEXT NOT NULL,
                modified_timestamp REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                hash_id TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                start INTEGER NOT NULL,
                end INTEGER NOT NULL,
                chunk_type TEXT NOT NULL,
                parent_id INTEGER,
                PRIMARY KEY (hash_id, chunk_id),
                FOREIGN KEY (hash_id) REFERENCES files(hash_id)
                    ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS reverse_key (
                hash_id TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                token TEXT NOT NULL,
                frequency INTEGER NOT NULL,
                PRIMARY KEY (hash_id, chunk_id, token),
                FOREIGN KEY (hash_id, chunk_id)
                    REFERENCES chunks(hash_id, chunk_id)
                    ON DELETE CASCADE
            );
            """
        )
        self.__db.commit()

    def __path_hash(self, filename: Path) -> str:
        """Return the stable identifier for a file path."""
        path = str(filename.resolve()).encode("utf-8")
        return hashlib.sha256(path).hexdigest()

    def __content_hash(self, filename: Path) -> str:
        """Return the SHA256 digest of a file's raw contents."""
        digest = hashlib.sha256()
        with filename.open("rb") as file_object:
            for block in iter(lambda: file_object.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def get_file_identity(self, filename: Path) -> tuple[str, str]:
        """Return the path hash and pending content hash for a file."""
        resolved = str(filename.resolve())
        content_hash = self.__pending_hashes.get(resolved)
        if content_hash is None:
            content_hash = self.__content_hash(filename)
        return self.__path_hash(filename), content_hash

    def get_file_metadata(self, filename: Path) -> dict[str, object] | None:
        """
            Return the file metadata stored in the database.
        """
        hash_id = self.__path_hash(filename)
        row = self.__db.execute(
            "SELECT hash_id, path, content_hash, modified_timestamp "
            "FROM files WHERE hash_id = ?", (hash_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "hash_id": row[0],
            "path": row[1],
            "content_hash": row[2],
            "modified_timestamp": row[3],
        }

    def check_file_modified(self, filename: Path) -> bool:
        """
            Check if the file has been modified since the last corpus indexing.
        """
        metadata = self.get_file_metadata(filename)
        if metadata is None:
            self.__pending_hashes[str(filename.resolve())] = (
                self.__content_hash(filename)
            )
            return True
        timestamp = filename.stat().st_mtime
        if timestamp == metadata["modified_timestamp"]:
            return False
        content_hash = self.__content_hash(filename)
        self.__pending_hashes[str(filename.resolve())] = content_hash
        return content_hash != metadata["content_hash"]

    def update_file_metadata(self, filename: Path) -> None:
        """
            Store or Update the current file metadata into the database.
        """
        resolved = str(filename.resolve())
        content_hash = self.__pending_hashes.pop(resolved, None)
        if content_hash is None:
            content_hash = self.__content_hash(filename)
        self.__db.execute(
            "INSERT INTO files(hash_id, path, content_hash, "
            "modified_timestamp) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(hash_id) DO UPDATE SET "
            "path=excluded.path, content_hash=excluded.content_hash, "
            "modified_timestamp=excluded.modified_timestamp",
            (self.__path_hash(filename), resolved, content_hash,
             filename.stat().st_mtime),
        )
        self.__db.commit()

    def store_chunks(self, chunks: Iterable[Chunk]) -> None:
        """
            Store the given chunk into the database.
        """
        rows = [
            (chunk.file_path_hash, chunk.id, chunk.start, chunk.end,
             chunk.chunk_type.value, chunk.parent_id)
            for chunk in chunks
        ]
        self.__db.executemany(
            "INSERT OR REPLACE INTO chunks "
            "(hash_id, chunk_id, start, end, chunk_type, parent_id) "
            "VALUES (?, ?, ?, ?, ?, ?)", rows,
        )
        self.__db.commit()

    def get_chunk(self, hash_id: str, chunk_id: int) -> Chunk:
        """
            Return the corresponding chunk for a given id
        """
        from ...Indexor.Chunker.Chunk import Chunk, ChunkType

        row = self.__db.execute(
            "SELECT c.chunk_id, f.path, f.content_hash, c.start, c.end, "
            "c.chunk_type, c.parent_id FROM chunks c JOIN files f "
            "ON c.hash_id = f.hash_id WHERE c.hash_id = ? AND c.chunk_id = ?",
            (hash_id, chunk_id),
        ).fetchone()
        if row is None:
            raise KeyError((hash_id, chunk_id))
        tokens = self.__db.execute(
            "SELECT token, frequency FROM reverse_key "
            "WHERE hash_id = ? AND chunk_id = ? "
            "ORDER BY token", (hash_id, chunk_id)
        ).fetchall()
        token_values: list[str] = []
        for token, frequency in tokens:
            token_values.extend([token] * frequency)
        return Chunk(
            id=row[0], file_path=Path(row[1]), start=row[3], end=row[4],
            chunk_type=ChunkType(row[5]), parent_id=row[6],
            tokens=token_values,
            file_path_hash=hash_id, file_content_hash=row[2],
        )

    def store_bm25_index(
            self, index: dict[str, dict[ChunkKey, int]]) -> None:
        """
            Store the Index created by the BM25 algorithm
        """
        self.__db.execute("DELETE FROM reverse_key")
        rows = []
        for token, postings in index.items():
            for chunk_key, frequency in postings.items():
                if not isinstance(chunk_key, tuple):
                    raise ValueError("Database indexes require hashed chunks")
                hash_id, chunk_id = chunk_key
                rows.append((hash_id, chunk_id, token, frequency))
        self.__db.executemany(
            "INSERT INTO reverse_key(hash_id, chunk_id, token, frequency) "
            "VALUES (?, ?, ?, ?)", rows
        )
        self.__db.commit()

    def load_bm25_index(self) -> dict[str, dict[tuple[str, int], int]]:
        """
            Load the BM25 index from the databasa
        """
        rows = self.__db.execute(
            "SELECT token, hash_id, chunk_id, frequency FROM reverse_key"
        ).fetchall()
        index: dict[str, dict[tuple[str, int], int]] = {}
        for token, hash_id, chunk_id, frequency in rows:
            index.setdefault(token, {})[(hash_id, chunk_id)] = frequency
        return index

    def get_all_chunks(self) -> list[Chunk]:
        """Return every currently stored chunk."""
        rows = self.__db.execute(
            "SELECT hash_id, chunk_id FROM chunks ORDER BY hash_id, chunk_id"
        ).fetchall()
        return [self.get_chunk(row[0], row[1]) for row in rows]

    def replace_file_chunks(
            self, filename: Path, chunks: Iterable[Chunk]) -> None:
        """Replace all stored chunks for a file in one transaction."""
        hash_id = self.__path_hash(filename)
        self.__db.execute("DELETE FROM chunks WHERE hash_id = ?", (hash_id,))
        self.__db.executemany(
            "INSERT INTO chunks(hash_id, chunk_id, start, end, "
            "chunk_type, parent_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(chunk.file_path_hash, chunk.id, chunk.start, chunk.end,
              chunk.chunk_type.value, chunk.parent_id) for chunk in chunks],
        )
        self.__db.commit()

    def close(self) -> None:
        """Close the SQLite connection."""
        self.__db.close()
