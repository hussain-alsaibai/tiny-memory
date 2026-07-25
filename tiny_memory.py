"""
tiny_memory.py
~~~~~~~~~~~~~~

Zero-dependency AI agent memory system.
Single file. No embeddings API needed.

Features
--------
- SemanticMemory:  BM25 + TF-IDF n-gram pseudo-embeddings, hybrid search
- EpisodicMemory:  LRU sliding-window event log
- DeclarativeMemory: key/value fact store with TTL
- SlidingWindowMemory: token-budgeted conversation window w/ compression
- SummaryCompressor: extractive (default) or LLM-backed summarizer
- TokenCounter: pluggable token counter (heuristic or tiktoken-style)
- DotDict: dot-notation access over declarative memory
- SQLiteBackend: optional SQLite persistence (default keeps JSON/JSONL)

Author: Hussain Alsaibai (hussain-alsaibai)
Version: 0.1.0
License: MIT
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import uuid
from collections import Counter, OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# BM25 — pure-Python, no dependencies
# ---------------------------------------------------------------------------

class BM25:
    """
    Pure-Python BM25 implementation.

    Parameters
    ----------
    k1 : float
        Term frequency saturation parameter. Default 1.5.
    b : float
        Document length normalisation parameter. Default 0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_lengths: list[int] = []
        self._avgdl: float = 0.0
        self._doc_freqs: dict[str, int] = defaultdict(int)  # term -> num docs containing term
        self._N: int = 0  # total documents
        self._docs: list[list[str]] = []  # tokenised documents

    def index(self, documents: list[list[str]]) -> None:
        """
        Build the BM25 index from a list of pre-tokenised documents.
        Each document is a list of string tokens.
        """
        self._docs = documents
        self._N = len(documents)
        self._doc_lengths = [len(doc) for doc in documents]
        self._avgdl = sum(self._doc_lengths) / self._N if self._N > 0 else 0.0

        # Count document frequencies
        self._doc_freqs.clear()
        for doc in documents:
            unique_terms = set(doc)
            for term in unique_terms:
                self._doc_freqs[term] += 1

    def score(self, query: list[str]) -> list[float]:
        """
        Return BM25 scores for all indexed documents given a query
        (list of string tokens). Returns a list of scores aligned with
        the input document order.
        """
        scores: list[float] = []
        idfs = self._compute_idf(query)

        for i, doc in enumerate(self._docs):
            score = 0.0
            doc_tf = Counter(doc)
            doc_len = self._doc_lengths[i]

            for term in query:
                if term not in doc_tf:
                    continue
                tf = doc_tf[term]
                idf = idfs.get(term, 0.0)
                # BM25 scoring formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avgdl, 1))
                score += idf * (numerator / max(denominator, 1e-9))
            scores.append(score)

        return scores

    def _compute_idf(self, terms: list[str]) -> dict[str, float]:
        """Compute IDF for each unique term in the query."""
        idf: dict[str, float] = {}
        N = self._N
        for term in set(terms):
            df = self._doc_freqs.get(term, 0)
            # Smoothed IDF to avoid zero for unseen terms
            idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return idf


# ---------------------------------------------------------------------------
# TF-IDF N-gram pseudo-embeddings
# ---------------------------------------------------------------------------

class TFIDFEmbedding:
    """
    Deterministic character n-gram TF-IDF pseudo-embeddings.

    Uses a fixed vocabulary of the most-common character 3-grams
    to produce fixed-dimension vectors without any external API.
    The vocabulary is determined once from a reference corpus
    (passed at construction) and is otherwise deterministic.

    Parameters
    ----------
    dimension : int
        Embedding vector dimension. Default 128.
    ngram_size : int
        Character n-gram size. Default 3.
    """

    def __init__(
        self,
        dimension: int = 128,
        ngram_size: int = 3,
    ) -> None:
        self.dimension = dimension
        self.ngram_size = ngram_size
        self._vocab: list[str] = []
        self._idf: dict[str, float] = {}
        self._doc_count: int = 0
        self._built: bool = False
        self._lock = threading.Lock()

    def build_vocab(self, corpus: list[str]) -> None:
        """
        Build the fixed n-gram vocabulary and IDF weights from a corpus.
        Must be called before embedding any text.
        """
        with self._lock:
            ngram_counts: dict[str, int] = Counter()
            doc_ngram_sets: dict[str, int] = defaultdict(int)

            for doc in corpus:
                tokens = self._tokenise(doc)
                ngrams = self._extract_ngrams(tokens)
                ngram_counts.update(ngrams)
                for ng in set(ngrams):
                    doc_ngram_sets[ng] += 1

            self._doc_count = len(corpus)
            # Select top `dimension` n-grams by document frequency
            sorted_ngrams = sorted(doc_ngram_sets.items(), key=lambda x: x[1], reverse=True)
            self._vocab = [ng for ng, _ in sorted_ngrams[: self.dimension]]

            # Compute IDF for each vocab n-gram
            N = max(self._doc_count, 1)
            for ng in self._vocab:
                df = doc_ngram_sets.get(ng, 0)
                self._idf[ng] = math.log(N / (df + 1)) + 1

            self._built = True

    def embed(self, text: str) -> list[float]:
        """
        Convert text to a normalised TF-IDF n-gram vector.

        Returns
        -------
        list[float]
            Dense vector of length `dimension`.
        """
        if not self._built:
            # Build vocab lazily from the text itself + empty seed
            self.build_vocab([text])

        tokens = self._tokenise(text)
        ngrams = self._extract_ngrams(tokens)
        tf = Counter(ngrams)

        # Build sparse TF-IDF vector for vocab
        vec: dict[int, float] = {}
        for i, ng in enumerate(self._vocab):
            if ng in tf:
                tf_val = tf[ng]
                idf_val = self._idf.get(ng, math.log(self._doc_count + 1) + 1)
                vec[i] = tf_val * idf_val

        # Normalise (L2)
        magnitude = math.sqrt(sum(v * v for v in vec.values()))
        if magnitude > 0:
            vec = {i: v / magnitude for i, v in vec.items()}

        # Dense output
        dense = [0.0] * self.dimension
        for i, v in vec.items():
            dense[i] = v

        return dense

    def _tokenise(self, text: str) -> list[str]:
        """Lowercase and split on non-alphanumeric characters."""
        text = text.lower()
        tokens = re.split(r'[^a-z0-9]+', text)
        return [t for t in tokens if t]

    def _extract_ngrams(self, tokens: list[str]) -> list[str]:
        """Extract character n-grams from token list."""
        ngrams: list[str] = []
        for token in tokens:
            if len(token) < self.ngram_size:
                continue
            padded = '##' + token + '##'
            for i in range(len(padded) - self.ngram_size + 1):
                ngrams.append(padded[i : i + self.ngram_size])
        return ngrams


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------

class SemanticMemory:
    """
    Semantic memory store with BM25 keyword search and TF-IDF cosine similarity.

    Supports hybrid search (combining BM25 + cosine scores), TTL expiration,
    thread-safe operations, and JSONL persistence.

    Parameters
    ----------
    persist_path : str | Path | None
        Path to JSONL file for persistence. If None, persistence is disabled.
    embedding_dim : int
        Dimension of TF-IDF embedding vectors. Default 128.
    """

    def __init__(
        self,
        persist_path: str | Path | None = None,
        embedding_dim: int = 128,
    ) -> None:
        self.persist_path = Path(persist_path) if persist_path else None
        self.embedding_dim = embedding_dim

        self._entries: dict[str, dict[str, Any]] = {}  # id -> entry dict
        self._lock = threading.RLock()

        # Lazy-initialise the TF-IDF embedder (built on first add)
        self._embedder: TFIDFEmbedding = TFIDFEmbedding(dimension=embedding_dim)
        self._embedder_built: bool = False

        self._bm25: BM25 = BM25()
        self._bm25_dirty: bool = True

        if self.persist_path and self.persist_path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, text: str, metadata: Optional[dict[str, Any]] = None, ttl: Optional[float] = None) -> str:
        """
        Add a text entry to semantic memory.

        Parameters
        ----------
        text : str
            The text to store and make searchable.
        metadata : dict | None
            Arbitrary metadata to attach to the entry.
        ttl : float | None
            Time-to-live in seconds. Entry expires after this many seconds.

        Returns
        -------
        str
            The UUID4 hex ID of the created entry.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        memory_id = uuid.uuid4().hex
        now = time.time()

        entry: dict[str, Any] = {
            "id": memory_id,
            "text": text,
            "metadata": metadata or {},
            "timestamp": now,
            "ttl": ttl,
            "expires_at": (now + ttl) if ttl is not None else None,
        }

        with self._lock:
            # Build embedder vocab lazily from existing texts + new text
            if not self._embedder_built:
                corpus = [e["text"] for e in self._entries.values()]
                if text not in corpus:
                    corpus.append(text)
                self._embedder.build_vocab(corpus)
                self._embedder_built = True

            entry["embedding_vector"] = self._embedder.embed(text)
            self._entries[memory_id] = entry
            self._bm25_dirty = True

        return memory_id

    def search(
        self,
        query: str,
        top_k: int = 5,
        hybrid: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Search semantic memory.

        Parameters
        ----------
        query : str
            The search query string.
        top_k : int
            Maximum number of results to return. Default 5.
        hybrid : bool
            If True, combine BM25 and cosine similarity scores.
            If False, use BM25 keyword search only. Default True.

        Returns
        -------
        list[dict]
            List of matching entries, each containing
            ``id``, ``text``, ``metadata``, ``score``, ``timestamp``.
            Sorted by descending score. Expired entries are automatically
            excluded.
        """
        with self._lock:
            # Always prune expired entries first
            self._prune_expired(lock_held=True)
            if self._bm25_dirty:
                self._bm25.index([self._tokenise(e["text"]) for e in self._entries.values()])
                self._bm25_dirty = False

            if not self._entries:
                return []

            query_tokens = self._tokenise(query)
            query_embedding = self._embedder.embed(query)
            bm25_scores = self._bm25.score(query_tokens)

            results: list[dict[str, Any]] = []
            entries_list = list(self._entries.values())

            for i, entry in enumerate(entries_list):
                bm25_raw = bm25_scores[i]

                if hybrid:
                    cosine_raw = self._cosine_sim(query_embedding, entry["embedding_vector"])
                    # Normalise both scores to [0, 1] and blend
                    # BM25: map to [0,1] using a soft normalisation (avoid divide by zero)
                    bm25_norm = bm25_raw / max(sum(1 for _ in query_tokens), 1)
                    # Cosine is already [0, 1]
                    combined = 0.5 * min(bm25_norm, 1.0) + 0.5 * cosine_raw
                else:
                    combined = bm25_raw

                # Skip zero-score results (no relevance)
                if combined <= 0.0:
                    continue

                results.append(
                    {
                        "id": entry["id"],
                        "text": entry["text"],
                        "metadata": entry["metadata"],
                        "timestamp": entry["timestamp"],
                        "score": combined,
                    }
                )

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

    def delete(self, memory_id: str) -> bool:
        """
        Delete an entry by its ID.

        Returns
        -------
        bool
            True if the entry was found and deleted, False otherwise.
        """
        with self._lock:
            if memory_id not in self._entries:
                return False
            del self._entries[memory_id]
            self._bm25_dirty = True
            return True

    def save(self) -> None:
        """Persist all entries to the JSONL file."""
        if not self.persist_path:
            return
        with self._lock:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as fh:
                for entry in self._entries.values():
                    # Omit embedding_vector from disk to save space
                    serialisable = {k: v for k, v in entry.items() if k != "embedding_vector"}
                    fh.write(json.dumps(serialisable, ensure_ascii=False) + "\n")

    def load(self) -> None:
        """Load entries from the JSONL file, rebuilding embeddings."""
        if not self.persist_path or not self.persist_path.exists():
            return
        with self._lock:
            self._entries.clear()
            raw_texts: list[str] = []
            with open(self.persist_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    raw_texts.append(entry["text"])
                    self._entries[entry["id"]] = entry

            # Rebuild embedder vocab
            if raw_texts:
                self._embedder.build_vocab(raw_texts)
                self._embedder_built = True
                # Re-compute embeddings for loaded entries
                for entry in self._entries.values():
                    entry["embedding_vector"] = self._embedder.embed(entry["text"])

            self._bm25_dirty = True

    def count(self) -> int:
        """Return the number of non-expired entries."""
        with self._lock:
            self._prune_expired(lock_held=True)
            return len(self._entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Lowercase tokenisation for BM25."""
        text = text.lower()
        tokens = re.split(r'[^a-z0-9]+', text)
        return [t for t in tokens if t]

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Dot product of two normalised vectors = cosine similarity."""
        return sum(x * y for x, y in zip(a, b))

    def _prune_expired(self, lock_held: bool = False) -> None:
        """Remove entries past their TTL."""
        now = time.time()
        to_delete = [
            eid
            for eid, e in self._entries.items()
            if e.get("expires_at") is not None and e["expires_at"] <= now
        ]
        for eid in to_delete:
            del self._entries[eid]
        if to_delete and not lock_held:
            self._bm25_dirty = True


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    LRU episodic memory store for recent agent events.

    Stores events in a deque with configurable maximum size.
    Persists to JSONL.

    Parameters
    ----------
    max_size : int
        Maximum number of events to retain. Default 1000.
    persist_path : str | Path | None
        Path to JSONL file for persistence.
    """

    def __init__(
        self,
        max_size: int = 1000,
        persist_path: str | Path | None = None,
    ) -> None:
        self.max_size = max_size
        self.persist_path = Path(persist_path) if persist_path else None

        self._events: deque[dict[str, Any]] = deque(maxlen=max_size)
        self._lock = threading.RLock()

        if self.persist_path and self.persist_path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, event_type: str, data: Any) -> str:
        """
        Record a new episodic event.

        Parameters
        ----------
        event_type : str
            Semantic label for the event (e.g. "tool_call", "user_message").
        data : Any
            Event payload. Must be JSON-serializable.

        Returns
        -------
        str
            UUID4 hex event ID.
        """
        event_id = uuid.uuid4().hex
        event: dict[str, Any] = {
            "id": event_id,
            "timestamp": time.time(),
            "type": event_type,
            "data": data,
        }
        with self._lock:
            self._events.append(event)
        return event_id

    def get_recent(self, n: int = 50) -> list[dict[str, Any]]:
        """
        Return the n most recent events.

        Returns
        -------
        list[dict]
            List of event dicts, oldest first.
        """
        with self._lock:
            n = min(n, len(self._events))
            # deque doesn't support slicing; convert to list
            events_list = list(self._events)
            return events_list[-n:]

    def search(self, query: str) -> list[dict[str, Any]]:
        """
        Full-text search across event types and data values.

        Parameters
        ----------
        query : str
            Keyword to search for.

        Returns
        -------
        list[dict]
            Events whose type or serialised data contain the query keyword.
        """
        query_lower = query.lower()
        results: list[dict[str, Any]] = []
        with self._lock:
            for event in self._events:
                # Search in event type
                if query_lower in event["type"].lower():
                    results.append(event)
                    continue
                # Search in serialised data
                try:
                    data_str = json.dumps(event["data"]).lower()
                    if query_lower in data_str:
                        results.append(event)
                except (TypeError, ValueError):
                    pass
        return results

    def clear(self) -> None:
        """Remove all recorded events."""
        with self._lock:
            self._events.clear()

    def save(self) -> None:
        """Persist events to JSONL."""
        if not self.persist_path:
            return
        with self._lock:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as fh:
                for event in self._events:
                    fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def load(self) -> None:
        """Load events from JSONL."""
        if not self.persist_path or not self.persist_path.exists():
            return
        with self._lock:
            self._events.clear()
            with open(self.persist_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    self._events.append(json.loads(line))

    def count(self) -> int:
        """Return the number of stored events."""
        with self._lock:
            return len(self._events)


# ---------------------------------------------------------------------------
# DeclarativeMemory
# ---------------------------------------------------------------------------

class DeclarativeMemory:
    """
    Key-value declarative memory store.

    Stores facts as key-value pairs with optional TTL.
    Persists to a single JSON file.

    Parameters
    ----------
    persist_path : str | Path | None
        Path to JSON file for persistence.
    """

    def __init__(
        self,
        persist_path: str | Path | None = None,
    ) -> None:
        self.persist_path = Path(persist_path) if persist_path else None
        self._store: dict[str, dict[str, Any]] = {}  # key -> {value, timestamp, ttl}
        self._lock = threading.RLock()

        if self.persist_path and self.persist_path.exists():
            self.load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remember(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Store a fact.

        Parameters
        ----------
        key : str
            Fact identifier.
        value : Any
            JSON-serializable value.
        ttl : float | None
            Optional time-to-live in seconds.
        """
        now = time.time()
        with self._lock:
            self._store[key] = {
                "value": value,
                "timestamp": now,
                "ttl": ttl,
                "expires_at": (now + ttl) if ttl is not None else None,
            }

    def recall(self, key: str) -> Any:
        """
        Retrieve a fact.

        Parameters
        ----------
        key : str
            Fact identifier.

        Returns
        -------
        Any
            The stored value, or None if not found or expired.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.get("expires_at") is not None and entry["expires_at"] <= time.time():
                del self._store[key]
                return None
            return entry["value"]

    def update(self, key: str, value: Any) -> bool:
        """
        Update an existing fact. Does nothing if the key does not exist.

        Returns
        -------
        bool
            True if the key existed and was updated.
        """
        with self._lock:
            if key not in self._store:
                return False
            entry = self._store[key]
            entry["value"] = value
            entry["timestamp"] = time.time()
            return True

    def forget(self, key: str) -> bool:
        """
        Delete a fact.

        Returns
        -------
        bool
            True if the key existed.
        """
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def search(self, keyword: str) -> dict[str, Any]:
        """
        Find all key-value pairs where the key or value (str) contains the keyword.

        Parameters
        ----------
        keyword : str
            Keyword to search for (case-insensitive).

        Returns
        -------
        dict
            Mapping of matching keys to their values.
        """
        keyword_lower = keyword.lower()
        results: dict[str, Any] = {}
        with self._lock:
            for key, entry in self._store.items():
                if keyword_lower in key.lower():
                    results[key] = entry["value"]
                elif isinstance(entry["value"], str) and keyword_lower in entry["value"].lower():
                    results[key] = entry["value"]
        return results

    def keys(self) -> list[str]:
        """Return all non-expired keys."""
        with self._lock:
            self._prune_expired(lock_held=True)
            return list(self._store.keys())

    def values(self) -> list[Any]:
        """Return all non-expired values."""
        with self._lock:
            self._prune_expired(lock_held=True)
            return [e["value"] for e in self._store.values()]

    def items(self) -> list[tuple[str, Any]]:
        """Return all non-expired key-value pairs."""
        with self._lock:
            self._prune_expired(lock_held=True)
            return [(k, e["value"]) for k, e in self._store.items()]

    def save(self) -> None:
        """Persist the store to JSON."""
        if not self.persist_path:
            return
        with self._lock:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as fh:
                json.dump(self._store, fh, ensure_ascii=False, indent=2)

    def load(self) -> None:
        """Load the store from JSON."""
        if not self.persist_path or not self.persist_path.exists():
            return
        with self._lock:
            with open(self.persist_path, encoding="utf-8") as fh:
                self._store = json.load(fh)

    def count(self) -> int:
        """Return the number of non-expired entries."""
        with self._lock:
            self._prune_expired(lock_held=True)
            return len(self._store)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_expired(self, lock_held: bool = False) -> None:
        """Remove expired entries."""
        now = time.time()
        expired = [k for k, e in self._store.items() if e.get("expires_at") is not None and e["expires_at"] <= now]
        for k in expired:
            del self._store[k]


# ---------------------------------------------------------------------------
# TokenCounter — pluggable token budget helper
# ---------------------------------------------------------------------------

class TokenCounter:
    """
    Pluggable token counter used for token-budget enforcement.

    The default heuristic uses the common ``~4 characters per token`` rule
    of thumb (OpenAI / Anthropic averages for English text). For exact
    counts, supply any callable with the same signature as
    ``tiktoken.Encoding.encode`` that returns a list of token ids, e.g.::

        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        counter = TokenCounter(lambda text: enc.encode(text))

    Parameters
    ----------
    encoder : callable | None
        ``str -> list[int]``. If ``None``, uses the 4-char heuristic.
    chars_per_token : float
        Fallback characters-per-token ratio. Default 4.0 (English avg).
    """

    def __init__(
        self,
        encoder: Optional[Any] = None,
        chars_per_token: float = 4.0,
    ) -> None:
        self.encoder = encoder
        self.chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        """Return the token count for ``text``."""
        if self.encoder is not None:
            try:
                return len(self.encoder(text))
            except Exception:
                # Bad encoder — fall back to heuristic
                pass
        if not text:
            return 0
        # Heuristic: words / 0.75 ≈ tokens (OpenAI rule of thumb).
        # But we also accept ``chars_per_token`` for language-specific tuning.
        return max(1, int(round(len(text) / max(self.chars_per_token, 0.1))))

    def count_messages(self, messages: list[Any]) -> int:
        """
        Count tokens across a list of messages.

        Each item may be a string OR a dict with a ``"content"`` key
        (ChatML / OpenAI style). Dicts contribute ``content`` + 4 tokens
        overhead per message (matches the OpenAI cookbook guideline).
        """
        total = 0
        for m in messages:
            if isinstance(m, str):
                total += self.count(m)
            elif isinstance(m, dict):
                total += self.count(str(m.get("content", ""))) + 4
            else:
                total += self.count(str(m))
        return total


# ---------------------------------------------------------------------------
# SummaryCompressor — extractive summarizer with pluggable LLM hook
# ---------------------------------------------------------------------------

class SummaryCompressor:
    """
    Compress a list of messages into a shorter form respecting a token budget.

    Two compression modes:

    * **extractive** (default, zero-dependency): rank sentences by TF-IDF
      weight, pick the highest-ranked until the target budget is filled.
      Fast, deterministic, no LLM required.
    * **abstractive** (optional): call a user-supplied ``summarizer``
      callable ``list[str] -> str``. Drop-in for an LLM-based compressor.

    Parameters
    ----------
    summarizer : callable | None
        ``list[str] -> str`` used for abstractive mode. If ``None``,
        extractive mode is used.
    token_counter : TokenCounter | None
        Token counter used to enforce the budget. Defaults to a heuristic
        :class:`TokenCounter`.
    cache_size : int
        LRU cache size for repeated summaries. Default 64.
    """

    _SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        summarizer: Optional[Any] = None,
        token_counter: Optional[TokenCounter] = None,
        cache_size: int = 64,
    ) -> None:
        self.summarizer = summarizer
        self.token_counter = token_counter or TokenCounter()
        # Tiny LRU for repeated compress() calls
        self._cache: "OrderedDict[tuple, str]" = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        messages: list[str],
        target_tokens: int,
    ) -> str:
        """
        Compress ``messages`` into at most ``target_tokens`` tokens.

        Parameters
        ----------
        messages : list[str]
            Source text fragments (one per message / turn).
        target_tokens : int
            Maximum token budget for the output summary.

        Returns
        -------
        str
            Compressed summary text. Empty string if ``messages`` is empty
            or ``target_tokens`` <= 0.
        """
        if not messages or target_tokens <= 0:
            return ""

        cache_key = (tuple(messages), int(target_tokens))
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                # Touch for LRU
                self._cache.move_to_end(cache_key)
                return cached

        if self.summarizer is not None:
            summary = self._call_abstractive(messages, target_tokens)
        else:
            summary = self._call_extractive(messages, target_tokens)

        with self._lock:
            self._cache[cache_key] = summary
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        return summary

    def clear_cache(self) -> None:
        """Drop the LRU cache."""
        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------------
    # Implementations
    # ------------------------------------------------------------------

    def _call_abstractive(
        self,
        messages: list[str],
        target_tokens: int,
    ) -> str:
        """Call user-supplied LLM-based summarizer; trim to budget."""
        joined = "\n".join(messages)
        try:
            summary = self.summarizer(messages)
        except Exception:
            # Fall back to extractive if the LLM fails
            return self._call_extractive(messages, target_tokens)

        if not isinstance(summary, str):
            summary = str(summary)

        # Trim if the LLM produced too many tokens
        return self._trim_to_budget(summary, target_tokens)

    def _call_extractive(
        self,
        messages: list[str],
        target_tokens: int,
    ) -> str:
        """Extractive summarization via TF-IDF sentence ranking."""
        # Split each message into sentences
        sentences: list[str] = []
        for msg in messages:
            for s in self._SENTENCE_SPLIT.split(msg.strip()):
                if s:
                    sentences.append(s)

        if not sentences:
            return ""

        # Score each sentence by the sum of TF-IDF weights of its terms
        # (using log-frequency as a stand-in for IDF, single-corpus).
        scored: list[tuple[float, str]] = []
        for sent in sentences:
            tokens = re.findall(r"\w+", sent.lower())
            if not tokens:
                continue
            tf = Counter(tokens)
            # Score: sum of (1 + log(tf)) for unique terms — favours
            # sentence length lightly but rewards vocabulary density.
            score = sum(1.0 + math.log(c) for c in tf.values())
            # Length penalty to avoid one-word sentences dominating
            score = score / math.sqrt(max(len(tokens), 1))
            scored.append((score, sent))

        # Sort by descending score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Greedily fill the budget, preserving original order on output
        chosen_indices: set[int] = []
        budget = target_tokens
        # Use original ordering for the final assembly
        chosen_pairs: list[tuple[int, str]] = []  # (orig_index, sentence)
        for orig_idx, (_score, sent) in enumerate(
            (s for s in scored)
        ):
            # Map back to original sentence index
            pass

        # Re-derive chosen sentences in their original order
        original_sentences = sentences
        sent_to_score = {sent: score for score, sent in scored}
        ranked = sorted(
            (sent_to_score[s], i, s) for i, s in enumerate(original_sentences)
            if s in sent_to_score
        )
        ranked.sort(key=lambda x: x[0], reverse=True)

        picked: list[tuple[int, str]] = []  # (orig_idx, sentence)
        for _score, i, s in ranked:
            cost = self.token_counter.count(s) + 1  # +1 for separator
            if cost > budget and not picked:
                # Always include at least one sentence if everything exceeds
                picked.append((i, s))
                break
            if cost <= budget:
                picked.append((i, s))
                budget -= cost
            if budget <= 0:
                break

        picked.sort(key=lambda x: x[0])
        summary = " ".join(s for _, s in picked)
        return self._trim_to_budget(summary, target_tokens)

    def _trim_to_budget(self, text: str, target_tokens: int) -> str:
        """Hard-trim ``text`` so it does not exceed ``target_tokens``."""
        if target_tokens <= 0:
            return ""
        # Approximate per-word trim (cheaper than per-token for default counter)
        words = text.split()
        if not words:
            return ""
        # Estimate tokens by words × 1.3 (rule of thumb) when using heuristic
        max_words = max(1, int(target_tokens / 1.3))
        if len(words) > max_words:
            words = words[:max_words]
        return " ".join(words)


# ---------------------------------------------------------------------------
# DotDict — recursive dot-notation wrapper around declarative memory
# ---------------------------------------------------------------------------

class DotDict:
    """
    Recursive dot-notation wrapper over a :class:`DeclarativeMemory`.

    Provides ergonomic attribute-style access::

        mem = AgentMemory()
        mem.dot.user.name = "hussain"      # stores "user.name" -> "hussain"
        mem.dot.user.email = "h@x.com"     # nested grouping via dotted keys
        mem.dot.user.name                  # -> "hussain"
        mem.dot.user                       # -> DotDict view of "user.*"

    Nested groups share a prefix; setting ``mem.dot.a.b.c = v`` creates
    keys ``a.b.c`` in the underlying declarative store.

    Parameters
    ----------
    declarative : DeclarativeMemory
        Backing declarative store.
    prefix : str
        Key prefix for this view (empty for the root).
    """

    __slots__ = ("_declarative", "_prefix")

    def __init__(self, declarative: "DeclarativeMemory", prefix: str = "") -> None:
        object.__setattr__(self, "_declarative", declarative)
        object.__setattr__(self, "_prefix", prefix)

    # ------------------------------------------------------------------
    # Mapping helpers
    # ------------------------------------------------------------------

    def _resolve_key(self, name: str) -> str:
        """Convert an attribute name into the underlying store key."""
        if not name or name.startswith("_"):
            raise AttributeError(f"invalid attribute: {name!r}")
        prefix = object.__getattribute__(self, "_prefix")
        return f"{prefix}.{name}" if prefix else name

    def _child_prefix(self, name: str) -> str:
        prefix = object.__getattribute__(self, "_prefix")
        return f"{prefix}.{name}" if prefix else name

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        declarative = object.__getattribute__(self, "_declarative")
        key = self._resolve_key(name)
        val = declarative.recall(key)
        if val is None:
            # Return a nested DotDict view so chaining works for unset groups
            return DotDict(declarative, prefix=self._child_prefix(name))
        # If the stored value is itself a dict, wrap it as a DotDict
        if isinstance(val, dict):
            # We can't fully represent dict values through a prefix; return as-is
            return val
        return val

    def __setattr__(self, name: str, value: Any) -> None:
        declarative = object.__getattribute__(self, "_declarative")
        key = self._resolve_key(name)
        declarative.remember(key, value)

    def __delattr__(self, name: str) -> None:
        declarative = object.__getattribute__(self, "_declarative")
        key = self._resolve_key(name)
        if not declarative.forget(key):
            raise AttributeError(name)

    def __repr__(self) -> str:
        prefix = object.__getattribute__(self, "_prefix")
        if not prefix:
            items = self._declarative.items()
            return f"DotDict({dict(items)!r})"
        # Subview: only show keys under this prefix
        items = {
            k[len(prefix) + 1:]: v
            for k, v in self._declarative.items()
            if k.startswith(prefix + ".")
        }
        return f"DotDict({prefix!r}, {items!r})"

    def __contains__(self, name: str) -> bool:
        declarative = object.__getattribute__(self, "_declarative")
        key = self._resolve_key(name)
        return declarative.recall(key) is not None

    def __iter__(self):
        prefix = object.__getattribute__(self, "_prefix")
        for k in self._declarative.keys():
            if not prefix:
                # Top-level: yield the first segment only
                head = k.split(".", 1)[0]
                if head:
                    yield head
            elif k.startswith(prefix + "."):
                rest = k[len(prefix) + 1:]
                head = rest.split(".", 1)[0]
                if head:
                    yield head


# ---------------------------------------------------------------------------
# SlidingWindowMemory — token-budgeted conversation window w/ compression
# ---------------------------------------------------------------------------

class SlidingWindowMemory:
    """
    Token-budgeted sliding-window conversation memory.

    Keeps the most recent turns within a configured token budget. When a
    new turn would exceed the budget, the oldest turns are evicted and
    (optionally) compressed into a rolling summary that pre-pends the
    window::

        window = SlidingWindowMemory(
            max_tokens=1000,
            compressor=SummaryCompressor(),
            token_counter=TokenCounter(),
        )
        window.add("user", "Hello!")
        window.add("assistant", "Hi, how can I help?")
        messages = window.messages()  # list of {"role", "content"} dicts

    Parameters
    ----------
    max_tokens : int
        Maximum total tokens retained across all turns + summary.
    token_counter : TokenCounter | None
        Token counter; defaults to heuristic :class:`TokenCounter`.
    compressor : SummaryCompressor | None
        Compressor for evicted turns. If ``None``, evicted turns are
        dropped without summarization.
    summary_ratio : float
        Fraction of the budget reserved for the compressed summary when a
        compressor is configured. Default 0.25 (25% of budget).
    """

    def __init__(
        self,
        max_tokens: int = 1000,
        token_counter: Optional[TokenCounter] = None,
        compressor: Optional[SummaryCompressor] = None,
        summary_ratio: float = 0.25,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self.max_tokens = max_tokens
        self.token_counter = token_counter or TokenCounter()
        self.compressor = compressor
        self.summary_ratio = max(0.0, min(summary_ratio, 0.9))

        self._turns: deque[dict[str, str]] = deque()
        self._summary: str = ""
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, role: str, content: str) -> None:
        """
        Append a new turn.

        If the total token cost would exceed ``max_tokens``, the oldest
        turns are evicted (and optionally summarized) until the budget is
        satisfied.
        """
        with self._lock:
            self._turns.append({"role": str(role), "content": str(content)})
            self._enforce_budget()

    def messages(self) -> list[dict[str, str]]:
        """
        Return all turns in chronological order.

        If a summary exists, it is returned as the first item with
        ``role="system"``.
        """
        with self._lock:
            if self._summary:
                return [{"role": "system", "content": self._summary}] + list(self._turns)
            return list(self._turns)

    def clear(self) -> None:
        """Drop all turns and the summary."""
        with self._lock:
            self._turns.clear()
            self._summary = ""

    def total_tokens(self) -> int:
        """Return the current total token count (turns + summary)."""
        with self._lock:
            return self.token_counter.count_messages(self.messages())

    def summary(self) -> str:
        """Return the current rolling summary (may be empty)."""
        with self._lock:
            return self._summary

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _enforce_budget(self) -> None:
        """Evict (and summarize) oldest turns until under budget."""
        summary_budget = (
            int(self.max_tokens * self.summary_ratio) if self.compressor else 0
        )
        turn_budget = self.max_tokens - summary_budget

        while self._turns and self._total_turn_tokens() > turn_budget:
            evicted = self._turns.popleft()
            if self.compressor is not None:
                # Add the evicted turn's content to a pending summary buffer
                # We rebuild the summary lazily on each eviction to keep it
                # bounded.
                self._refresh_summary(include=evicted["content"], target=summary_budget)

        # If even after eviction we're over budget (single huge turn), keep
        # the newest turn and accept the overflow.
        if self._turns and self._total_turn_tokens() > turn_budget:
            # Drop everything except the newest turn.
            newest = self._turns.pop()
            self._turns.clear()
            self._turns.append(newest)

    def _total_turn_tokens(self) -> int:
        return self.token_counter.count_messages(list(self._turns))

    def _refresh_summary(self, include: str, target: int) -> None:
        """Recompute the rolling summary including ``include`` text."""
        # Compose the existing summary plus the new evicted turn
        parts: list[str] = []
        if self._summary:
            parts.append(self._summary)
        parts.append(include)
        # Compress to target budget
        if target <= 0:
            self._summary = ""
            return
        self._summary = self.compressor.compress(parts, target_tokens=target)


# ---------------------------------------------------------------------------
# SQLiteBackend — optional SQLite persistence for memory stores
# ---------------------------------------------------------------------------

class SQLiteBackend:
    """
    SQLite-backed persistence adapter for tiny-memory.

    Drop-in alternative to the JSON/JSONL file backends. Provides three
    tables — ``semantic``, ``episodic``, ``declarative`` — that match the
    schemas used by :class:`AgentMemory`.

    Use directly with the store constructors via ``persist_path='memory.db'``
    OR explicitly via :meth:`AgentMemory.use_sqlite`.

    The backend is fully thread-safe (``check_same_thread=False`` + a
    ``threading.Lock`` around writes).

    Parameters
    ----------
    path : str | Path
        Path to the SQLite database file. Created if missing.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # Detect driver: prefer stdlib sqlite3
        try:
            import sqlite3  # noqa: F401  (stdlib)
            self._driver = "sqlite3"
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("sqlite3 driver unavailable") from exc
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _connect(self):
        import sqlite3
        conn = sqlite3.connect(str(self.path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS semantic (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        ttl REAL,
                        expires_at REAL
                    );
                    CREATE TABLE IF NOT EXISTS episodic (
                        id TEXT PRIMARY KEY,
                        timestamp REAL NOT NULL,
                        type TEXT NOT NULL,
                        data TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS declarative (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        ttl REAL,
                        expires_at REAL
                    );
                    CREATE INDEX IF NOT EXISTS idx_semantic_ts ON semantic(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_episodic_ts ON episodic(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_decl_ts ON declarative(timestamp);
                    """
                )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Semantic operations
    # ------------------------------------------------------------------

    def save_semantic(self, entries: list[dict[str, Any]]) -> None:
        """Replace the semantic table with ``entries``."""
        import sqlite3
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM semantic")
                conn.executemany(
                    "INSERT INTO semantic (id, text, metadata, timestamp, ttl, expires_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            e["id"],
                            e["text"],
                            json.dumps(e.get("metadata") or {}),
                            float(e["timestamp"]),
                            e.get("ttl"),
                            e.get("expires_at"),
                        )
                        for e in entries
                    ],
                )
                conn.commit()
            finally:
                conn.close()

    def load_semantic(self) -> list[dict[str, Any]]:
        """Load all semantic entries from the table."""
        import sqlite3
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT id, text, metadata, timestamp, ttl, expires_at FROM semantic"
                ).fetchall()
            finally:
                conn.close()
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "id": row[0],
                    "text": row[1],
                    "metadata": json.loads(row[2]) if row[2] else {},
                    "timestamp": row[3],
                    "ttl": row[4],
                    "expires_at": row[5],
                }
            )
        return out

    # ------------------------------------------------------------------
    # Episodic operations
    # ------------------------------------------------------------------

    def save_episodic(self, events: list[dict[str, Any]]) -> None:
        import sqlite3
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM episodic")
                conn.executemany(
                    "INSERT INTO episodic (id, timestamp, type, data) VALUES (?, ?, ?, ?)",
                    [
                        (
                            e["id"],
                            float(e["timestamp"]),
                            e["type"],
                            json.dumps(e["data"]),
                        )
                        for e in events
                    ],
                )
                conn.commit()
            finally:
                conn.close()

    def load_episodic(self) -> list[dict[str, Any]]:
        import sqlite3
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT id, timestamp, type, data FROM episodic ORDER BY timestamp ASC"
                ).fetchall()
            finally:
                conn.close()
        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "type": row[2],
                "data": json.loads(row[3]),
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Declarative operations
    # ------------------------------------------------------------------

    def save_declarative(self, entries: dict[str, dict[str, Any]]) -> None:
        import sqlite3
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM declarative")
                conn.executemany(
                    "INSERT INTO declarative (key, value, timestamp, ttl, expires_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            k,
                            json.dumps(e["value"]),
                            float(e["timestamp"]),
                            e.get("ttl"),
                            e.get("expires_at"),
                        )
                        for k, e in entries.items()
                    ],
                )
                conn.commit()
            finally:
                conn.close()

    def load_declarative(self) -> dict[str, dict[str, Any]]:
        import sqlite3
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT key, value, timestamp, ttl, expires_at FROM declarative"
                ).fetchall()
            finally:
                conn.close()
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            out[row[0]] = {
                "value": json.loads(row[1]),
                "timestamp": row[2],
                "ttl": row[3],
                "expires_at": row[4],
            }
        return out

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear_all(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    "DELETE FROM semantic; DELETE FROM episodic; DELETE FROM declarative;"
                )
                conn.commit()
            finally:
                conn.close()

    def stats(self) -> dict[str, int]:
        """Return row counts per table."""
        import sqlite3
        with self._lock:
            conn = self._connect()
            try:
                sem = conn.execute("SELECT COUNT(*) FROM semantic").fetchone()[0]
                epi = conn.execute("SELECT COUNT(*) FROM episodic").fetchone()[0]
                dec = conn.execute("SELECT COUNT(*) FROM declarative").fetchone()[0]
            finally:
                conn.close()
        return {"semantic": sem, "episodic": epi, "declarative": dec}


# ---------------------------------------------------------------------------
# AgentMemory — Facade
# ---------------------------------------------------------------------------

class AgentMemory:
    """
    Unified memory facade combining SemanticMemory, EpisodicMemory, and
    DeclarativeMemory into a single entry point for AI agents.

    Parameters
    ----------
    semantic_path : str | Path | None
        Persistence path for semantic memory. Default "semantic.jsonl".
    episodic_path : str | Path | None
        Persistence path for episodic memory. Default "episodic.jsonl".
    declarative_path : str | Path | None
        Persistence path for declarative memory. Default "declarative.json".
    semantic_embedding_dim : int
        Embedding dimension for semantic memory. Default 128.
    episodic_max_size : int
        Maximum episodic events. Default 1000.
    backend : str
        Persistence backend. ``"json"`` (default, JSON / JSONL files) or
        ``"sqlite"`` (single-file SQLite database). For ``"sqlite"``, the
        ``sqlite_path`` argument overrides the database location.
    sqlite_path : str | Path | None
        SQLite database path when ``backend="sqlite"``. Defaults to
        ``"tiny_memory.db"``.
    """

    def __init__(
        self,
        semantic_path: str | Path | None = "semantic.jsonl",
        episodic_path: str | Path | None = "episodic.jsonl",
        declarative_path: str | Path | None = "declarative.json",
        semantic_embedding_dim: int = 128,
        episodic_max_size: int = 1000,
        backend: str = "json",
        sqlite_path: str | Path | None = None,
    ) -> None:
        if backend not in ("json", "sqlite"):
            raise ValueError(f"unknown backend: {backend!r} (use 'json' or 'sqlite')")
        self.backend = backend
        self.sqlite_backend: Optional[SQLiteBackend] = None

        if backend == "sqlite":
            db_path = Path(sqlite_path) if sqlite_path else Path("tiny_memory.db")
            self.sqlite_backend = SQLiteBackend(db_path)
            # Disable the JSON file paths when using SQLite
            semantic_path = episodic_path = declarative_path = None

        self.semantic = SemanticMemory(
            persist_path=Path(semantic_path) if semantic_path else None,
            embedding_dim=semantic_embedding_dim,
        )
        self.episodic = EpisodicMemory(
            max_size=episodic_max_size,
            persist_path=Path(episodic_path) if episodic_path else None,
        )
        self.declarative = DeclarativeMemory(
            persist_path=Path(declarative_path) if declarative_path else None,
        )

        # If SQLite backend requested and an existing DB is present, load it
        if self.sqlite_backend is not None and self.sqlite_backend.path.exists():
            self.load_all()

    # ------------------------------------------------------------------
    # Declarative shortcuts
    # ------------------------------------------------------------------

    def remember(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Store a fact in declarative memory.

        Alias for ``declarative.remember(key, value, ttl)``.
        """
        self.declarative.remember(key, value, ttl=ttl)

    def recall(self, key: str) -> Any:
        """
        Retrieve a fact from declarative memory.

        Alias for ``declarative.recall(key)``.
        """
        return self.declarative.recall(key)

    # ------------------------------------------------------------------
    # Episodic shortcuts
    # ------------------------------------------------------------------

    def store_episode(self, event_type: str, data: Any) -> str:
        """
        Record an event in episodic memory.

        Alias for ``episodic.add(event_type, data)``.
        """
        return self.episodic.add(event_type, data)

    # ------------------------------------------------------------------
    # Semantic shortcuts
    # ------------------------------------------------------------------

    def memorize(
        self,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        ttl: Optional[float] = None,
    ) -> str:
        """
        Store text in semantic memory.

        Alias for ``semantic.add(text, metadata, ttl)``.
        """
        return self.semantic.add(text, metadata=metadata, ttl=ttl)

    # ------------------------------------------------------------------
    # Unified query
    # ------------------------------------------------------------------

    def query(
        self,
        q: str,
        top_k: int = 5,
        hybrid: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Search all three memory stores and return ranked combined results.

        Each result dict includes a ``source`` field indicating where it
        came from: ``"semantic"``, ``"episodic"``, or ``"declarative"``.

        Parameters
        ----------
        q : str
            Search query string.
        top_k : int
            Maximum total results. Default 5.
        hybrid : bool
            Use hybrid (BM25 + cosine) scoring for semantic search.
            Default True.

        Returns
        -------
        list[dict]
            Ranked list of results from all memory stores.
        """
        results: list[dict[str, Any]] = []

        # Semantic search
        semantic_results = self.semantic.search(q, top_k=top_k * 2, hybrid=hybrid)
        for r in semantic_results:
            r["source"] = "semantic"
            results.append(r)

        # Episodic search
        episodic_matches = self.episodic.search(q)
        for event in episodic_matches[: top_k * 2]:
            # Convert event to a "text" for output consistency
            event_text = f"[{event['type']}] {json.dumps(event['data'])}"
            results.append(
                {
                    "id": event["id"],
                    "text": event_text,
                    "metadata": {"event_type": event["type"], "data": event["data"]},
                    "timestamp": event["timestamp"],
                    "score": 1.0,  # episodic search is binary match; use full score
                    "source": "episodic",
                }
            )

        # Declarative search
        declarative_matches = self.declarative.search(q)
        for key, value in declarative_matches.items():
            val_str = json.dumps(value) if not isinstance(value, str) else value
            results.append(
                {
                    "id": key,
                    "text": f"{key}: {val_str}",
                    "metadata": {"key": key, "value": value},
                    "timestamp": None,
                    "score": 1.0,
                    "source": "declarative",
                }
            )

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_all(self) -> None:
        """Persist all three memory stores."""
        if self.sqlite_backend is not None:
            # Pull entries directly from the in-memory stores
            self.sqlite_backend.save_semantic(
                [
                    {k: v for k, v in e.items() if k != "embedding_vector"}
                    for e in self.semantic._entries.values()
                ]
            )
            self.sqlite_backend.save_episodic(list(self.episodic._events))
            self.sqlite_backend.save_declarative(dict(self.declarative._store))
            return
        self.semantic.save()
        self.episodic.save()
        self.declarative.save()

    def load_all(self) -> None:
        """Load all three memory stores from disk."""
        if self.sqlite_backend is not None:
            for entry in self.sqlite_backend.load_semantic():
                # Re-add through the public API so embedder/index stay consistent
                self.semantic.add(
                    entry["text"],
                    metadata=entry.get("metadata") or {},
                    ttl=entry.get("ttl"),
                )
            for event in self.sqlite_backend.load_episodic():
                self.episodic.add(event["type"], event["data"])
            for key, stored in self.sqlite_backend.load_declarative().items():
                self.declarative.remember(
                    key,
                    stored["value"],
                    ttl=stored.get("ttl"),
                )
            return
        self.semantic.load()
        self.episodic.load()
        self.declarative.load()

    # ------------------------------------------------------------------
    # Dot-notation access
    # ------------------------------------------------------------------

    @property
    def dot(self) -> DotDict:
        """Dot-notation wrapper over declarative memory."""
        return DotDict(self.declarative)

    # ------------------------------------------------------------------
    # Token-budgeted conversation window
    # ------------------------------------------------------------------

    def window(
        self,
        max_tokens: int = 1000,
        token_counter: Optional[TokenCounter] = None,
        compressor: Optional[SummaryCompressor] = None,
        summary_ratio: float = 0.25,
    ) -> SlidingWindowMemory:
        """
        Construct a token-budgeted sliding window for conversation turns.

        The returned window is independent of this facade's other stores;
        it lives until garbage-collected or explicitly closed. To persist
        window contents, mirror them through :meth:`store_episode`.
        """
        return SlidingWindowMemory(
            max_tokens=max_tokens,
            token_counter=token_counter,
            compressor=compressor,
            summary_ratio=summary_ratio,
        )

    # ------------------------------------------------------------------
    # SQLite migration helper
    # ------------------------------------------------------------------

    def use_sqlite(self, path: str | Path) -> None:
        """
        Migrate the facade to a SQLite backend, persisting current state.

        After calling this, ``save_all`` / ``load_all`` use SQLite. Existing
        JSON/JSONL files are not modified or deleted.
        """
        new_backend = SQLiteBackend(Path(path))
        # Persist current state to the new DB
        new_backend.save_semantic(
            [
                {k: v for k, v in e.items() if k != "embedding_vector"}
                for e in self.semantic._entries.values()
            ]
        )
        new_backend.save_episodic(list(self.episodic._events))
        new_backend.save_declarative(dict(self.declarative._store))
        # Detach file backends so save_all uses SQLite
        self.semantic.persist_path = None
        self.episodic.persist_path = None
        self.declarative.persist_path = None
        self.sqlite_backend = new_backend
        self.backend = "sqlite"

    def stats(self) -> dict[str, Any]:
        """Return entry counts across the three stores."""
        out = {
            "semantic": self.semantic.count(),
            "episodic": self.episodic.count(),
            "declarative": self.declarative.count(),
            "backend": self.backend,
        }
        if self.sqlite_backend is not None:
            out["sqlite_path"] = str(self.sqlite_backend.path)
        return out


# ---------------------------------------------------------------------------
# CLI entry point (trivial)
# ---------------------------------------------------------------------------

def main() -> None:
    """Simple CLI demo — add a memory and query it."""
    import sys

    mem = AgentMemory(
        semantic_path="~/.tiny_memory/semantic.jsonl",
        episodic_path="~/.tiny_memory/episodic.jsonl",
        declarative_path="~/.tiny_memory/declarative.json",
    )

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "add" and len(sys.argv) > 2:
            mem.memorize(sys.argv[2])
            mem.save_all()
            print(f"Added: {sys.argv[2]}")
        elif cmd == "query" and len(sys.argv) > 2:
            for r in mem.query(sys.argv[2]):
                print(f"  [{r['source']}] {r['text'][:80]}")
        else:
            print("Usage: tiny-memory add <text> | query <text>")
    else:
        # Interactive demo
        print("tiny-memory interactive demo")
        print("Commands: add <text> | query <text> | recall <key> | remember <key> <value> | quit")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            parts = line.split(None, 2)
            cmd = parts[0].lower()
            if cmd == "quit":
                break
            elif cmd == "add" and len(parts) > 1:
                mem.memorize(parts[1])
                mem.save_all()
                print(f"  ✓ stored")
            elif cmd == "query" and len(parts) > 1:
                for r in mem.query(parts[1]):
                    print(f"  [{r['source']}] score={r['score']:.3f}  {r['text'][:80]}")
            elif cmd == "recall" and len(parts) > 1:
                val = mem.recall(parts[1])
                print(f"  {parts[1]} = {val}")
            elif cmd == "remember" and len(parts) > 2:
                mem.remember(parts[1], parts[2])
                mem.save_all()
                print(f"  ✓ remembered {parts[1]}")


if __name__ == "__main__":
    main()
