from __future__ import annotations
import hashlib
import sqlite3
from Indexor import Chunk, ChunkType
from pathlib import Path
from typing import Iterable, cast


ChunkKey = tuple[str, int]
SCHEMA_VERSION = 2


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
        self.__database_path = database.resolve()
        self.__db = sqlite3.connect(str(self.__database_path))
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
                token_lengths INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (hash_id, chunk_id),
                FOREIGN KEY (hash_id) REFERENCES files(hash_id)
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
                total_chunks INTEGER NOT NULL CHECK (total_chunks >= 0),
                total_token_lengths INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS database_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL CHECK (version >= 0)
            );
            CREATE INDEX IF NOT EXISTS idx_chunk_tokens_token
                ON chunk_tokens(token);
            """
        )
        self.__db.execute(
            "INSERT OR IGNORE INTO database_metadata(id, version) "
            "VALUES (1, 0)"
        )
        self.__db.commit()
        self.__migrate_schema()

    def __migrate_schema(self) -> None:
        """Migrate legacy BM25 storage once and set the schema version."""
        version = self.__db.execute("PRAGMA user_version").fetchone()[0]
        if version >= SCHEMA_VERSION:
            return

        if not self.__has_column("chunks", "token_lengths"):
            self.__db.execute(
                "ALTER TABLE chunks ADD COLUMN token_lengths "
                "INTEGER NOT NULL DEFAULT 0"
            )
        if not self.__has_column("index_metadata", "total_token_lengths"):
            self.__db.execute(
                "ALTER TABLE index_metadata ADD COLUMN total_token_lengths "
                "INTEGER NOT NULL DEFAULT 0"
            )

        has_reverse_key = self.__has_table("reverse_key")
        with self.__db:
            if has_reverse_key:
                self.__db.execute(
                    "INSERT OR IGNORE INTO chunk_tokens "
                    "(hash_id, chunk_id, token, frequency) "
                    "SELECT hash_id, chunk_id, token, frequency "
                    "FROM reverse_key"
                )
                self.__db.execute("DROP TABLE reverse_key")
            self.__db.execute(
                "UPDATE chunks SET token_lengths = COALESCE(("
                "SELECT SUM(frequency) FROM chunk_tokens "
                "WHERE chunk_tokens.hash_id = chunks.hash_id "
                "AND chunk_tokens.chunk_id = chunks.chunk_id), 0)"
            )
            self.__refresh_bm25_metadata()
            self.__db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

        if has_reverse_key:
            self.__db.execute("VACUUM")

    def __has_table(self, table: str) -> bool:
        """Return whether a table exists in the database."""
        row = self.__db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        return row is not None

    def __has_column(self, table: str, column: str) -> bool:
        """Return whether a table contains a column."""
        return column in {
            row[1] for row in self.__db.execute(f"PRAGMA table_info({table})")
        }

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
             chunk.chunk_type.value, chunk.parent_id, len(chunk.tokens))
            for chunk in chunks
        ]
        self.__db.executemany(
            "INSERT OR REPLACE INTO chunks "
            "(hash_id, chunk_id, start, end, chunk_type, parent_id, "
            "token_lengths) VALUES (?, ?, ?, ?, ?, ?, ?)", rows,
        )
        self.__db.commit()

    def get_chunk(self, hash_id: str, chunk_id: int) -> Chunk:
        """
            Return the corresponding chunk for a given id
        """

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
        token_stats: dict[str, int] = {}
        for hash_id, chunk_id, token, frequency in rows:
            token_stats[token] = token_stats.get(token, 0) + 1

        with self.__db:
            self.__db.execute("DELETE FROM chunk_tokens")
            self.__db.execute("DELETE FROM token_stats")
            self.__db.execute("DELETE FROM index_metadata")
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
                "INSERT INTO index_metadata(id, total_chunks, "
                "total_token_lengths) VALUES (1, ?, ?)",
                (total_chunks, self.__total_token_lengths()),
            )
            self.__increment_database_version()

    def get_database_version(self) -> str:
        """Return the current version of the indexed database."""
        row = self.__db.execute(
            "SELECT version FROM database_metadata WHERE id = 1"
        ).fetchone()
        if row is None:
            raise RuntimeError("Database version metadata is missing")
        database_id = hashlib.sha256(
            str(self.__database_path).encode("utf-8")
        ).hexdigest()
        return f"{SCHEMA_VERSION}:{database_id}:{row[0]}"

    def get_bm25_postings(
            self, tokens: Iterable[str]
            ) -> tuple[list[tuple[str, str, int, int, int, int]], int, int]:
        """Return postings, statistics, and lengths for query tokens only."""
        token_list = list(dict.fromkeys(tokens))
        if not token_list:
            return [], 0, 0
        placeholders = ", ".join("?" for _ in token_list)
        rows = self.__db.execute(
            "SELECT t.token, t.hash_id, t.chunk_id, t.frequency, "
            "s.chunk_frequency, c.token_lengths "
            "FROM chunk_tokens t "
            "JOIN token_stats s ON s.token = t.token "
            "JOIN chunks c ON c.hash_id = t.hash_id "
            "AND c.chunk_id = t.chunk_id "
            f"WHERE t.token IN ({placeholders})",
            token_list,
        ).fetchall()
        metadata = self.__db.execute(
            "SELECT total_chunks, total_token_lengths "
            "FROM index_metadata WHERE id = 1"
        ).fetchone()
        if metadata is None:
            return rows, 0, 0
        return rows, metadata[0], metadata[1]

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
                self.__increment_database_version()

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
            "INSERT INTO index_metadata(id, total_chunks, "
            "total_token_lengths) VALUES (1, ?, ?)",
            (total_chunks, self.__total_token_lengths()),
        )

    def __increment_database_version(self) -> None:
        """Increment the persisted indexed-data generation."""
        self.__db.execute(
            "UPDATE database_metadata SET version = version + 1 WHERE id = 1"
        )

    def refresh_bm25_metadata(self) -> None:
        """Refresh BM25 aggregates without rebuilding persisted postings."""
        with self.__db:
            self.__refresh_bm25_metadata()

    def __total_token_lengths(self) -> int:
        """Return the persisted total number of chunk tokens."""
        row = self.__db.execute(
            "SELECT COALESCE(SUM(token_lengths), 0) FROM chunks"
        ).fetchone()
        return 0 if row is None else row[0]

    def get_all_chunks(self) -> list[Chunk]:
        """Return every currently stored chunk."""
        rows = self.__db.execute(
            "SELECT c.hash_id, c.chunk_id, f.path, f.content_hash, "
            "c.start, c.end, c.chunk_type, c.parent_id, "
            "t.token, t.frequency "
            "FROM chunks c JOIN files f ON c.hash_id = f.hash_id "
            "LEFT JOIN chunk_tokens t "
            "ON c.hash_id = t.hash_id AND c.chunk_id = t.chunk_id "
            "ORDER BY c.hash_id, c.chunk_id, t.token"
        ).fetchall()
        chunks: list[Chunk] = []
        current_key: tuple[str, int] | None = None
        current_tokens: list[str] = []
        current_data: tuple[object, ...] | None = None

        for row in rows:
            key = (row[0], row[1])
            if current_key is not None and key != current_key:
                chunks.append(self.__build_chunk(current_data, current_tokens))
                current_tokens = []
            if key != current_key:
                current_key = key
                current_data = row[:8]
            if row[8] is not None:
                current_tokens.extend([row[8]] * row[9])

        if current_key is not None:
            chunks.append(self.__build_chunk(current_data, current_tokens))
        return chunks

    def __build_chunk(
            self,
            data: tuple[object, ...] | None,
            tokens: list[str]) -> Chunk:
        """Build a chunk from one bulk-query row and its token frequencies."""
        if data is None:
            raise ValueError("Chunk data is required")
        return Chunk(
            id=cast(int, data[1]),
            file_path=Path(cast(str, data[2])),
            start=cast(int, data[4]),
            end=cast(int, data[5]),
            chunk_type=ChunkType(cast(str, data[6])),
            parent_id=cast(int | None, data[7]),
            tokens=tokens,
            file_path_hash=cast(str, data[0]),
            file_content_hash=cast(str, data[3]),
        )

    def replace_file_chunks(
            self, filename: Path, chunks: Iterable[Chunk]) -> None:
        """Replace all stored chunks for a file in one transaction."""
        hash_id = self.__path_hash(filename)
        chunk_list = list(chunks)
        chunk_rows = [
            (chunk.file_path_hash, chunk.id, chunk.start, chunk.end,
             chunk.chunk_type.value, chunk.parent_id, len(chunk.tokens))
            for chunk in chunk_list
        ]
        token_rows: list[tuple[str, int, str, int]] = []
        for chunk in chunk_list:
            frequencies: dict[str, int] = {}
            for token in chunk.tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            token_rows.extend(
                (chunk.file_path_hash, chunk.id, token, frequency)
                for token, frequency in frequencies.items()
            )

        with self.__db:
            self.__db.execute(
                "DELETE FROM chunks WHERE hash_id = ?", (hash_id,)
            )
            self.__db.executemany(
                "INSERT INTO chunks(hash_id, chunk_id, start, end, "
                "chunk_type, parent_id, token_lengths) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                chunk_rows,
            )
            self.__db.executemany(
                "INSERT INTO chunk_tokens(hash_id, chunk_id, token, "
                "frequency) VALUES (?, ?, ?, ?)",
                token_rows,
            )
            self.__increment_database_version()

    def close(self) -> None:
        """Close the SQLite connection."""
        self.__db.close()
