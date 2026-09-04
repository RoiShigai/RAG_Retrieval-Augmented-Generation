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
            CREATE TABLE IF NOT EXISTS chunk_tokens (
                hash_id TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                token TEXT NOT NULL,
                frequency INTEGER NOT NULL CHECK (frequency > 0),
                PRIMARY KEY (hash_id, chunk_id, token),
                FOREIGN KEY (hash_id, chunk_id)
                    REFERENCES chunks(hash_id, chunk_id)
                    ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS token_stats (
                token TEXT PRIMARY KEY,
                chunk_frequency INTEGER NOT NULL CHECK (chunk_frequency > 0)
            );
            CREATE TABLE IF NOT EXISTS index_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_chunks INTEGER NOT NULL CHECK (total_chunks >= 0)
            );
            """
        )
        self.__db.commit()
        self.__migrate_bm25_tables()

    def __migrate_bm25_tables(self) -> None:
        """Copy legacy reverse-key data into the persistent BM25 tables."""
        with self.__db:
            self.__db.execute(
                "INSERT OR IGNORE INTO chunk_tokens "
                "(hash_id, chunk_id, token, frequency) "
                "SELECT hash_id, chunk_id, token, frequency FROM reverse_key"
            )
            self.__refresh_bm25_metadata()

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
            "SELECT token, frequency FROM chunk_tokens "
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
        rows: list[tuple[str, int, str, int]] = []
        for token, postings in index.items():
            for chunk_key, frequency in postings.items():
                if not isinstance(chunk_key, tuple):
                    raise ValueError("Database indexes require hashed chunks")
                hash_id, chunk_id = chunk_key
                rows.append((hash_id, chunk_id, token, frequency))
        chunk_lengths: dict[tuple[str, int], int] = {}
        token_stats: dict[str, int] = {}
        for hash_id, chunk_id, token, frequency in rows:
            key = (hash_id, chunk_id)
            chunk_lengths[key] = chunk_lengths.get(key, 0) + frequency
            token_stats[token] = token_stats.get(token, 0) + 1

        with self.__db:
            self.__db.execute("DELETE FROM reverse_key")
            self.__db.execute("DELETE FROM chunk_tokens")
            self.__db.execute("DELETE FROM token_stats")
            self.__db.execute("DELETE FROM index_metadata")
            self.__db.executemany(
                "INSERT INTO reverse_key(hash_id, chunk_id, token, frequency) "
                "VALUES (?, ?, ?, ?)", rows
            )
            self.__db.executemany(
                "INSERT INTO chunk_tokens(hash_id, chunk_id, token, "
                "frequency) "
                "VALUES (?, ?, ?, ?)", rows
            )
            self.__db.executemany(
                "INSERT INTO token_stats(token, chunk_frequency) "
                "VALUES (?, ?)",
                token_stats.items(),
            )
            total_chunks = self.__db.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]
            self.__db.execute(
                "INSERT INTO index_metadata(id, total_chunks) VALUES (1, ?)",
                (total_chunks,),
            )

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

    def load_bm25_data(
            self,
            ) -> tuple[
                dict[str, dict[tuple[str, int], int]],
                dict[str, int],
                dict[tuple[str, int], int],
                int,
            ]:
        """Load all persisted data required to calculate BM25 scores."""
        index = self.load_bm25_index()
        stats = {
            token: frequency for token, frequency in self.__db.execute(
                "SELECT token, chunk_frequency FROM token_stats"
            ).fetchall()
        }
        lengths = {
            (hash_id, chunk_id): length
            for hash_id, chunk_id, length in self.__db.execute(
                "SELECT hash_id, chunk_id, SUM(frequency) "
                "FROM chunk_tokens GROUP BY hash_id, chunk_id"
            ).fetchall()
        }
        row = self.__db.execute(
            "SELECT total_chunks FROM index_metadata WHERE id = 1"
        ).fetchone()
        return index, stats, lengths, 0 if row is None else row[0]

    def synchronize_corpus(self, corpus: Path) -> bool:
        """Remove deleted files and report whether indexing is required."""
        actual_paths = {
            path.resolve() for path in corpus.rglob("*")
            if path.is_file() and path.suffix in {".py", ".md"}
        }
        stored_rows = self.__db.execute(
            "SELECT hash_id, path, modified_timestamp FROM files"
        ).fetchall()
        stored_paths = {Path(row[1]).resolve(): row for row in stored_rows}
        deleted = [row[0] for path, row in stored_paths.items()
                   if path not in actual_paths]
        changed = bool(deleted)
        with self.__db:
            for hash_id in deleted:
                self.__db.execute(
                    "DELETE FROM files WHERE hash_id = ?", (hash_id,)
                )
            if deleted:
                self.__refresh_bm25_metadata()

        for path in actual_paths:
            if self.check_file_modified(path):
                changed = True
            elif path in stored_paths:
                metadata = stored_paths[path]
                if path.stat().st_mtime != metadata[2]:
                    self.update_file_metadata(path)
        return changed

    def __refresh_bm25_metadata(self) -> None:
        """Rebuild aggregate BM25 statistics from persisted token rows."""
        self.__db.execute("DELETE FROM token_stats")
        self.__db.execute(
            "INSERT INTO token_stats(token, chunk_frequency) "
            "SELECT token, COUNT(*) FROM chunk_tokens GROUP BY token"
        )
        total_chunks = self.__db.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]
        self.__db.execute("DELETE FROM index_metadata")
        self.__db.execute(
            "INSERT INTO index_metadata(id, total_chunks) VALUES (1, ?)",
            (total_chunks,),
        )

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
