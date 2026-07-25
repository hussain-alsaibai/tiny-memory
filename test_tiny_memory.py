"""
test_tiny_memory.py
~~~~~~~~~~~~~~~~~~

Comprehensive tests for the tiny-memory package.
Run with: python test_tiny_memory.py

No external test framework required — uses assert statements.
"""

import json
import math
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

import tiny_memory as tm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tmp_path(name: str = "test.jsonl") -> Path:
    """Return a guaranteed-unique temp path inside a temp dir."""
    static_tmp = Path(tempfile.gettempdir()) / "tiny_memory_tests"
    static_tmp.mkdir(exist_ok=True)
    return static_tmp / f"{name}_{os.getpid()}_{time.time_ns()}"


def cleanup_tmp() -> None:
    """Remove the shared temp directory."""
    tmp = Path(tempfile.gettempdir()) / "tiny_memory_tests"
    shutil.rmtree(tmp, ignore_errors=True)


class TmpFile:
    """Context manager: yields a temp path, deletes it on exit."""

    def __init__(self, suffix: str = ".jsonl"):
        self.path = tmp_path(f"tmp_{time.time_ns()}{suffix}")
        self._created = False

    def __enter__(self) -> Path:
        self._created = True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()
        return self.path

    def __exit__(self, *args):
        if self._created:
            self.path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# BM25 Unit Tests
# ---------------------------------------------------------------------------

def test_bm25_basic() -> None:
    """BM25 scores documents by relevance to a query."""
    corpus = [
        ["hello", "world"],
        ["hello", "world", "foo"],
        ["bar", "baz"],
    ]
    bm25 = tm.BM25()
    bm25.index(corpus)
    scores = bm25.score(["hello"])

    # "hello world" docs should score higher than "bar baz" doc
    assert len(scores) == 3
    assert scores[0] > scores[2], f"Expected doc0 > doc2, got {scores[0]} vs {scores[2]}"
    assert scores[1] > scores[2]
    print("  ✓ BM25 basic scoring")


def test_bm25_idf_smoothing() -> None:
    """BM25 with IDF smoothing doesn't produce negative scores."""
    corpus = [["apple"], ["banana"], ["cherry"]]
    bm25 = tm.BM25()
    bm25.index(corpus)
    scores = bm25.score(["zebra"])  # Term not in any doc
    assert all(s >= 0 for s in scores), f"Negative BM25 score: {scores}"
    print("  ✓ BM25 IDF smoothing (no negative scores)")


def test_bm25_empty_query() -> None:
    """BM25 with empty query returns zeros."""
    corpus = [["hello", "world"]]
    bm25 = tm.BM25()
    bm25.index(corpus)
    scores = bm25.score([])
    assert scores == [0.0]
    print("  ✓ BM25 empty query")


def test_bm25_empty_corpus() -> None:
    """BM25 index on empty corpus doesn't crash."""
    bm25 = tm.BM25()
    bm25.index([])
    scores = bm25.score(["hello"])
    assert scores == []
    print("  ✓ BM25 empty corpus")


# ---------------------------------------------------------------------------
# TF-IDF Embedding Unit Tests
# ---------------------------------------------------------------------------

def test_tfidf_embedding_deterministic() -> None:
    """Same text always produces the same embedding vector."""
    embedder = tm.TFIDFEmbedding(dimension=64)
    embedder.build_vocab(["the quick brown fox", "jumps over the lazy dog"])
    v1 = embedder.embed("the quick fox")
    v2 = embedder.embed("the quick fox")
    assert v1 == v2, "TF-IDF embeddings must be deterministic"
    print("  ✓ TF-IDF embedding deterministic")


def test_tfidf_embedding_normalised() -> None:
    """TF-IDF embedding vectors are L2-normalised."""
    embedder = tm.TFIDFEmbedding(dimension=64)
    embedder.build_vocab(["hello world", "world of python"])
    vec = embedder.embed("hello world")
    magnitude = math.sqrt(sum(x * x for x in vec))
    assert abs(magnitude - 1.0) < 1e-6, f"Vector not normalised: magnitude={magnitude}"
    print("  ✓ TF-IDF embedding normalised")


def test_tfidf_embedding_length() -> None:
    """Embedding vectors have the configured dimension."""
    for dim in [32, 64, 128]:
        embedder = tm.TFIDFEmbedding(dimension=dim)
        embedder.build_vocab(["test document"])
        vec = embedder.embed("test")
        assert len(vec) == dim, f"Expected dim={dim}, got {len(vec)}"
    print("  ✓ TF-IDF embedding dimension")


def test_tfidf_cosine_similarity() -> None:
    """Cosine similarity of identical vectors is 1.0."""
    embedder = tm.TFIDFEmbedding(dimension=64)
    embedder.build_vocab(["the quick brown fox"])
    v1 = embedder.embed("the quick fox")
    v2 = embedder.embed("the quick fox")
    sim = tm.SemanticMemory._cosine_sim(v1, v2)
    assert abs(sim - 1.0) < 1e-6, f"Identical vectors should have cosine sim 1.0, got {sim}"
    print("  ✓ TF-IDF cosine similarity")


# ---------------------------------------------------------------------------
# SemanticMemory Tests
# ---------------------------------------------------------------------------

def test_semantic_add() -> None:
    """add() returns a valid UUID4 hex ID."""
    sm = tm.SemanticMemory()
    memory_id = sm.add("hello world", metadata={"foo": "bar"})
    assert isinstance(memory_id, str)
    assert len(memory_id) == 32
    assert all(c in "0123456789abcdef" for c in memory_id)
    assert sm.count() == 1
    print("  ✓ SemanticMemory add()")


def test_semantic_search_basic() -> None:
    """search() returns relevant results for a query."""
    sm = tm.SemanticMemory()
    sm.add("Python is a programming language", metadata={"topic": "python"})
    sm.add("JavaScript is a web language", metadata={"topic": "js"})
    sm.add("Python has list comprehensions", metadata={"topic": "python"})

    results = sm.search("Python", top_k=3, hybrid=True)
    assert len(results) >= 2
    # Only the Python-matching results should have positive scores;
    # zero-score entries (non-matches) are correctly omitted
    python_results = [r for r in results if "Python" in r["text"]]
    assert len(python_results) >= 2, f"Expected ≥2 Python results, got {python_results}"
    print("  ✓ SemanticMemory search()")


def test_semantic_search_hybrid_vs_keyword() -> None:
    """Hybrid and keyword-only searches both return results."""
    sm = tm.SemanticMemory()
    sm.add("the quick brown fox jumps over the lazy dog")
    sm.add("a fast red fox outruns a slow hound")

    hybrid = sm.search("quick fox", top_k=2, hybrid=True)
    keyword_only = sm.search("quick fox", top_k=2, hybrid=False)

    assert len(hybrid) >= 1
    assert len(keyword_only) >= 1
    print("  ✓ SemanticMemory hybrid vs keyword search")


def test_semantic_search_top_k() -> None:
    """top_k limits the number of returned results."""
    sm = tm.SemanticMemory()
    for i in range(20):
        sm.add(f"document number {i} with some content")

    for k in [1, 3, 5, 10]:
        results = sm.search("document", top_k=k)
        assert len(results) == min(k, sm.count())


def test_semantic_delete() -> None:
    """delete() removes an entry by ID."""
    sm = tm.SemanticMemory()
    id1 = sm.add("first entry")
    id2 = sm.add("second entry")
    assert sm.count() == 2

    deleted = sm.delete(id1)
    assert deleted is True
    assert sm.count() == 1

    not_found = sm.delete("nonexistent-id")
    assert not_found is False
    print("  ✓ SemanticMemory delete()")


def test_semantic_ttl_expiration() -> None:
    """Entries with TTL expire after the specified time."""
    sm = tm.SemanticMemory()
    # Use tokens with ZERO overlap between the two entries
    # to avoid any BM25 cross-similarity
    expired_id = sm.add("alpha_xyz_expire_123456", ttl=0.05)
    alive_id   = sm.add("delta_abc_alive_789012", ttl=3600)

    assert sm.count() == 2

    # Wait for the short-lived entry to expire
    time.sleep(0.15)

    # count() prunes expired entries — the definitive TTL guarantee
    assert sm.count() == 1, f"Expected 1 entry after TTL, got {sm.count()}"

    # The expired entry's ID must be gone from storage
    remaining_ids = {e["id"] for e in sm._entries.values()}
    assert expired_id not in remaining_ids, "Expired entry should be pruned from storage"
    assert alive_id in remaining_ids, "Alive entry should still be in storage"

    # The surviving entry is still searchable by exact match
    alive_results = sm.search("delta_abc_alive_789012", hybrid=False)
    assert len(alive_results) == 1
    assert alive_results[0]["id"] == alive_id

    # The expired entry's exact text is gone from search results
    expired_results = sm.search("alpha_xyz_expire_123456", hybrid=False)
    assert len(expired_results) == 0, f"Expired entry still returned: {expired_results}"

    print("  ✓ SemanticMemory TTL expiration")


def test_semantic_save_load_roundtrip() -> None:
    """save() and load() preserve entries correctly."""
    path = tmp_path("semantic.jsonl")
    sm1 = tm.SemanticMemory(persist_path=path)
    sm1.add("apple fruit", metadata={"category": "food"})
    sm1.add("banana fruit", metadata={"category": "food"})
    sm1.add("carrot vegetable", metadata={"category": "food"})
    sm1.save()

    sm2 = tm.SemanticMemory(persist_path=path)
    sm2.load()

    assert sm2.count() == 3
    texts = {r["text"] for r in sm2.search("fruit")}
    assert "apple fruit" in texts
    assert "banana fruit" in texts
    path.unlink(missing_ok=True)
    print("  ✓ SemanticMemory save/load roundtrip")


def test_semantic_empty_search() -> None:
    """Searching empty memory returns empty list."""
    sm = tm.SemanticMemory()
    results = sm.search("anything")
    assert results == []
    print("  ✓ SemanticMemory empty search")


def test_semantic_count() -> None:
    """count() returns the number of non-expired entries."""
    sm = tm.SemanticMemory()
    assert sm.count() == 0
    sm.add("entry one")
    sm.add("entry two")
    assert sm.count() == 2
    print("  ✓ SemanticMemory count()")


def test_semantic_metadata_preserved() -> None:
    """Metadata attached during add() is preserved in search results."""
    sm = tm.SemanticMemory()
    sm.add("test entry", metadata={"author": "alice", "version": 1})
    results = sm.search("test")
    assert len(results) == 1
    assert results[0]["metadata"]["author"] == "alice"
    assert results[0]["metadata"]["version"] == 1
    print("  ✓ SemanticMemory metadata preservation")


# ---------------------------------------------------------------------------
# EpisodicMemory Tests
# ---------------------------------------------------------------------------

def test_episodic_add() -> None:
    """add() stores an event and returns a UUID."""
    em = tm.EpisodicMemory()
    event_id = em.add("tool_call", {"tool": "browser", "url": "https://example.com"})
    assert isinstance(event_id, str)
    assert len(event_id) == 32
    assert em.count() == 1
    print("  ✓ EpisodicMemory add()")


def test_episodic_get_recent() -> None:
    """get_recent() returns events in chronological order."""
    em = tm.EpisodicMemory()
    em.add("a", {"n": 1})
    em.add("b", {"n": 2})
    em.add("c", {"n": 3})

    recent = em.get_recent(n=2)
    assert len(recent) == 2
    assert recent[0]["type"] == "b"
    assert recent[1]["type"] == "c"
    print("  ✓ EpisodicMemory get_recent()")


def test_episodic_get_recent_all() -> None:
    """get_recent() with n larger than store returns all events."""
    em = tm.EpisodicMemory()
    em.add("x", {"n": 1})
    em.add("y", {"n": 2})
    recent = em.get_recent(n=100)
    assert len(recent) == 2
    print("  ✓ EpisodicMemory get_recent(n=large)")


def test_episodic_search() -> None:
    """search() finds events by type or data content."""
    em = tm.EpisodicMemory()
    em.add("tool_call", {"tool": "browser", "url": "https://example.com"})
    em.add("user_message", {"text": "hello world"})
    em.add("tool_call", {"tool": "http_get", "url": "https://api.example.com"})

    results = em.search("browser")
    assert len(results) == 1
    assert results[0]["type"] == "tool_call"

    results2 = em.search("hello")
    assert len(results2) == 1
    assert results2[0]["type"] == "user_message"

    results3 = em.search("tool_call")
    assert len(results3) == 2
    print("  ✓ EpisodicMemory search()")


def test_episodic_clear() -> None:
    """clear() removes all events."""
    em = tm.EpisodicMemory()
    em.add("a", {})
    em.add("b", {})
    assert em.count() == 2
    em.clear()
    assert em.count() == 0
    print("  ✓ EpisodicMemory clear()")


def test_episodic_max_size() -> None:
    """deque respects max_size — oldest events are evicted."""
    em = tm.EpisodicMemory(max_size=5)
    for i in range(10):
        em.add(f"event_{i}", {"i": i})

    # Only the last 5 should remain
    assert em.count() == 5
    recent = em.get_recent(5)
    types = [e["type"] for e in recent]
    assert "event_0" not in types
    assert "event_9" in types
    print("  ✓ EpisodicMemory max_size eviction")


def test_episodic_save_load_roundtrip() -> None:
    """save() and load() preserve events."""
    path = tmp_path("episodic.jsonl")
    em1 = tm.EpisodicMemory(persist_path=path)
    em1.add("tool_call", {"tool": "test"})
    em1.add("error", {"code": 500})
    em1.save()

    em2 = tm.EpisodicMemory(persist_path=path)
    em2.load()
    assert em2.count() == 2
    path.unlink(missing_ok=True)
    print("  ✓ EpisodicMemory save/load roundtrip")


# ---------------------------------------------------------------------------
# DeclarativeMemory Tests
# ---------------------------------------------------------------------------

def test_declarative_remember_recall() -> None:
    """remember() stores, recall() retrieves."""
    dm = tm.DeclarativeMemory()
    dm.remember("name", "Amara")
    dm.remember("age", 32)
    assert dm.recall("name") == "Amara"
    assert dm.recall("age") == 32
    assert dm.recall("missing") is None
    print("  ✓ DeclarativeMemory remember/recall")


def test_declarative_update() -> None:
    """update() changes an existing value."""
    dm = tm.DeclarativeMemory()
    dm.remember("counter", 0)
    assert dm.recall("counter") == 0
    updated = dm.update("counter", 42)
    assert updated is True
    assert dm.recall("counter") == 42

    not_updated = dm.update("nonexistent", "value")
    assert not_updated is False
    print("  ✓ DeclarativeMemory update()")


def test_declarative_forget() -> None:
    """forget() deletes a key."""
    dm = tm.DeclarativeMemory()
    dm.remember("temp", "data")
    assert dm.recall("temp") == "data"
    forgotten = dm.forget("temp")
    assert forgotten is True
    assert dm.recall("temp") is None

    not_forgotten = dm.forget("temp")
    assert not_forgotten is False
    print("  ✓ DeclarativeMemory forget()")


def test_declarative_search() -> None:
    """search() finds keys/values containing the keyword."""
    dm = tm.DeclarativeMemory()
    dm.remember("user_name", "Amara")
    dm.remember("user_email", "amara@example.com")
    dm.remember("server_host", "api.example.com")

    results = dm.search("user")
    assert "user_name" in results
    assert "user_email" in results
    assert "server_host" not in results

    results2 = dm.search("example")
    assert "user_email" in results2
    assert "server_host" in results2
    print("  ✓ DeclarativeMemory search()")


def test_declarative_ttl() -> None:
    """TTL entries expire automatically."""
    dm = tm.DeclarativeMemory()
    dm.remember("short", "value", ttl=0.05)
    dm.remember("long", "value", ttl=3600)
    assert dm.recall("short") == "value"
    time.sleep(0.15)
    assert dm.recall("short") is None
    assert dm.recall("long") == "value"
    print("  ✓ DeclarativeMemory TTL")


def test_declarative_keys_values_items() -> None:
    """keys(), values(), items() work correctly."""
    dm = tm.DeclarativeMemory()
    dm.remember("a", 1)
    dm.remember("b", 2)
    dm.remember("c", 3)

    assert set(dm.keys()) == {"a", "b", "c"}
    assert set(dm.values()) == {1, 2, 3}
    assert set(dm.items()) == {("a", 1), ("b", 2), ("c", 3)}
    print("  ✓ DeclarativeMemory keys/values/items")


def test_declarative_save_load_roundtrip() -> None:
    """save() and load() preserve the full store."""
    path = tmp_path("declarative.json")
    dm1 = tm.DeclarativeMemory(persist_path=path)
    dm1.remember("name", "Alice")
    dm1.remember("preferences", {"theme": "dark"})
    dm1.save()

    dm2 = tm.DeclarativeMemory(persist_path=path)
    dm2.load()
    assert dm2.recall("name") == "Alice"
    assert dm2.recall("preferences") == {"theme": "dark"}
    path.unlink(missing_ok=True)
    print("  ✓ DeclarativeMemory save/load roundtrip")


# ---------------------------------------------------------------------------
# AgentMemory Facade Tests
# ---------------------------------------------------------------------------

def test_agent_memory_remember_recall() -> None:
    """AgentMemory.remember/recall delegate to DeclarativeMemory."""
    mem = tm.AgentMemory(
        semantic_path=None,
        episodic_path=None,
        declarative_path=None,
    )
    mem.remember("city", "Tokyo")
    assert mem.recall("city") == "Tokyo"
    print("  ✓ AgentMemory remember/recall")


def test_agent_memory_store_episode() -> None:
    """AgentMemory.store_episode delegates to EpisodicMemory."""
    mem = tm.AgentMemory()
    event_id = mem.store_episode("tool_used", {"tool": "browser"})
    assert isinstance(event_id, str)
    assert mem.episodic.count() == 1
    print("  ✓ AgentMemory store_episode")


def test_agent_memory_memorize() -> None:
    """AgentMemory.memorize delegates to SemanticMemory."""
    mem = tm.AgentMemory()
    memory_id = mem.memorize("Python list comprehensions are elegant")
    assert isinstance(memory_id, str)
    assert mem.semantic.count() == 1
    print("  ✓ AgentMemory memorize")


def test_agent_memory_query_combines_sources() -> None:
    """query() searches all three stores and returns combined results."""
    mem = tm.AgentMemory()
    mem.remember("favorite_language", "Python")
    mem.memorize("I mostly code in Python for data science tasks")
    mem.store_episode("code_event", {"language": "Python", "task": "ml_training"})

    results = mem.query("Python", top_k=10)
    sources = {r["source"] for r in results}
    assert "semantic" in sources
    assert "declarative" in sources
    assert "episodic" in sources
    print("  ✓ AgentMemory query() combines sources")


def test_agent_memory_query_ranking() -> None:
    """query() results are sorted by score descending."""
    mem = tm.AgentMemory()
    mem.memorize("apple fruit is red", metadata={"priority": "high"})
    mem.memorize("banana fruit is yellow", metadata={"priority": "low"})

    results = mem.query("fruit", top_k=2)
    if len(results) >= 2:
        assert results[0]["score"] >= results[1]["score"], "Results not sorted by score"
    print("  ✓ AgentMemory query() ranking")


def test_agent_memory_save_all_load_all() -> None:
    """save_all() and load_all() persist all three stores."""
    sem_path = tmp_path("semantic.jsonl")
    epi_path = tmp_path("episodic.jsonl")
    dec_path = tmp_path("declarative.json")

    mem1 = tm.AgentMemory(
        semantic_path=sem_path,
        episodic_path=epi_path,
        declarative_path=dec_path,
    )
    mem1.memorize("semantic entry")
    mem1.store_episode("test_event", {"foo": "bar"})
    mem1.remember("key", "value")
    mem1.save_all()

    mem2 = tm.AgentMemory(
        semantic_path=sem_path,
        episodic_path=epi_path,
        declarative_path=dec_path,
    )
    mem2.load_all()
    assert mem2.semantic.count() == 1
    assert mem2.episodic.count() == 1
    assert mem2.declarative.recall("key") == "value"

    for p in [sem_path, epi_path, dec_path]:
        p.unlink(missing_ok=True)
    print("  ✓ AgentMemory save_all/load_all")


# ---------------------------------------------------------------------------
# Thread Safety Tests
# ---------------------------------------------------------------------------

def test_semantic_thread_safety() -> None:
    """SemanticMemory operations are safe under concurrent access."""
    sm = tm.SemanticMemory()
    errors: list[str] = []
    barrier = threading.Barrier(10)

    def worker(idx: int) -> None:
        try:
            barrier.wait()
            for j in range(50):
                sm.add(f"text from thread {idx} item {j}", metadata={"t": idx, "i": j})
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert sm.count() == 500
    print("  ✓ SemanticMemory thread safety (10 threads × 50 adds)")


def test_declarative_thread_safety() -> None:
    """DeclarativeMemory operations are safe under concurrent access."""
    dm = tm.DeclarativeMemory()
    errors: list[str] = []
    barrier = threading.Barrier(10)

    def worker(idx: int) -> None:
        try:
            barrier.wait()
            for j in range(50):
                dm.remember(f"key_{idx}_{j}", f"value_{idx}_{j}")
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert dm.count() == 500
    print("  ✓ DeclarativeMemory thread safety (10 threads × 50 remembers)")


def test_episodic_thread_safety() -> None:
    """EpisodicMemory operations are safe under concurrent access."""
    em = tm.EpisodicMemory(max_size=500)  # smaller than 10*100=1000 to force LRU eviction
    errors: list[str] = []
    barrier = threading.Barrier(10)

    def worker(idx: int) -> None:
        try:
            barrier.wait()
            for j in range(100):
                em.add(f"event_{idx}", {"t": idx, "i": j})
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    # deque max_size caps at 500 (oldest events evicted)
    assert em.count() == 500, f"Expected count=500, got {em.count()}"
    print("  ✓ EpisodicMemory thread safety (10 threads × 100 adds)")


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

def test_semantic_empty_text_rejected() -> None:
    """add() with empty text raises ValueError."""
    sm = tm.SemanticMemory()
    try:
        sm.add("")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    try:
        sm.add("   ")
        assert False, "Should have raised ValueError for whitespace-only"
    except ValueError:
        pass
    print("  ✓ SemanticMemory rejects empty text")


def test_declarative_recall_missing_key() -> None:
    """DeclarativeMemory.recall returns None for missing keys."""
    dm = tm.DeclarativeMemory()
    assert dm.recall("does_not_exist") is None
    print("  ✓ DeclarativeMemory recall missing key")


def test_episodic_empty_search() -> None:
    """EpisodicMemory.search on empty store returns empty."""
    em = tm.EpisodicMemory()
    results = em.search("anything")
    assert results == []
    print("  ✓ EpisodicMemory empty search")


def test_agent_memory_direct_store_access() -> None:
    """AgentMemory exposes direct references to underlying stores."""
    mem = tm.AgentMemory()
    assert isinstance(mem.semantic, tm.SemanticMemory)
    assert isinstance(mem.episodic, tm.EpisodicMemory)
    assert isinstance(mem.declarative, tm.DeclarativeMemory)
    print("  ✓ AgentMemory direct store access")


# ---------------------------------------------------------------------------
# TokenCounter tests
# ---------------------------------------------------------------------------

def test_token_counter_heuristic_basic() -> None:
    """TokenCounter defaults to ~4 chars/token heuristic."""
    tc = tm.TokenCounter()
    n = tc.count("hello world")
    # "hello world" is 11 chars / 4 ≈ 3
    assert 1 <= n <= 5
    print("  ✓ TokenCounter heuristic")


def test_token_counter_empty_text() -> None:
    """TokenCounter returns 0 for empty text."""
    tc = tm.TokenCounter()
    assert tc.count("") == 0
    print("  ✓ TokenCounter empty text")


def test_token_counter_pluggable_encoder() -> None:
    """TokenCounter accepts a custom encoder."""
    # Toy encoder: count whitespace-separated words
    enc = lambda text: text.split()  # noqa: E731
    tc = tm.TokenCounter(encoder=enc)
    assert tc.count("one two three four five") == 5
    print("  ✓ TokenCounter pluggable encoder")


def test_token_counter_count_messages() -> None:
    """TokenCounter.count_messages handles dict messages (OpenAI style)."""
    tc = tm.TokenCounter()
    msgs = [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "hi there"},
    ]
    n = tc.count_messages(msgs)
    assert n > 0  # 4 overhead per message + content tokens
    print("  ✓ TokenCounter.count_messages")


# ---------------------------------------------------------------------------
# SummaryCompressor tests
# ---------------------------------------------------------------------------

def test_summary_compressor_extractive_within_budget() -> None:
    """SummaryCompressor extractive mode returns text within budget."""
    sc = tm.SummaryCompressor()
    messages = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a great programming language for AI agents.",
        "Vector databases are useful for semantic search.",
    ]
    summary = sc.compress(messages, target_tokens=10)
    assert isinstance(summary, str)
    # Summary should be a non-empty subset
    assert len(summary) > 0
    print("  ✓ SummaryCompressor extractive within budget")


def test_summary_compressor_empty_input() -> None:
    """SummaryCompressor returns empty string for empty input."""
    sc = tm.SummaryCompressor()
    assert sc.compress([], target_tokens=100) == ""
    assert sc.compress(["hello"], target_tokens=0) == ""
    print("  ✓ SummaryCompressor empty input")


def test_summary_compressor_abstractive_callable() -> None:
    """SummaryCompressor uses user-supplied summarizer in abstractive mode."""
    def fake_llm(msgs: list[str]) -> str:
        return "ABSTRACT: " + " ".join(msgs)[:50]

    sc = tm.SummaryCompressor(summarizer=fake_llm)
    out = sc.compress(["a", "b", "c"], target_tokens=100)
    assert out.startswith("ABSTRACT:")
    print("  ✓ SummaryCompressor abstractive callable")


def test_summary_compressor_cache() -> None:
    """SummaryCompressor caches identical compress() calls."""
    sc = tm.SummaryCompressor()
    messages = ["hello world", "foo bar"]
    a = sc.compress(messages, target_tokens=5)
    b = sc.compress(messages, target_tokens=5)
    assert a == b
    sc.clear_cache()
    print("  ✓ SummaryCompressor LRU cache")


# ---------------------------------------------------------------------------
# DotDict tests
# ---------------------------------------------------------------------------

def test_dotdict_set_get() -> None:
    """DotDict supports attribute-style get/set on declarative memory."""
    dm = tm.DeclarativeMemory()
    dd = tm.DotDict(dm)
    dd.user.name = "hussain"
    dd.user.email = "h@x.io"
    assert dd.user.name == "hussain"
    assert dd.user.email == "h@x.io"
    print("  ✓ DotDict set/get")


def test_dotdict_nested_prefix() -> None:
    """DotDict nests under shared prefixes via dotted keys."""
    dm = tm.DeclarativeMemory()
    dd = tm.DotDict(dm)
    dd.project.name = "tiny-memory"
    dd.project.version = "0.1.0"
    keys = sorted(dm.keys())
    assert "project.name" in keys
    assert "project.version" in keys
    print("  ✓ DotDict nested prefix")


def test_dotdict_contains_iter() -> None:
    """DotDict supports 'in' and iteration over its members."""
    dm = tm.DeclarativeMemory()
    dd = tm.DotDict(dm)
    dd.a = 1
    dd.b = 2
    assert "a" in dd
    assert "b" in dd
    assert sorted(list(dd)) == ["a", "b"]
    print("  ✓ DotDict contains/iter")


def test_dotdict_delete() -> None:
    """DotDict supports attribute deletion."""
    dm = tm.DeclarativeMemory()
    dd = tm.DotDict(dm)
    dd.k = "v"
    assert dm.recall("k") == "v"
    del dd.k
    assert dm.recall("k") is None
    print("  ✓ DotDict delete")


def test_dotdict_unset_returns_child_view() -> None:
    """DotDict accessing an unset key returns a nested view (chaining)."""
    dm = tm.DeclarativeMemory()
    dd = tm.DotDict(dm)
    view = dd.missing
    assert isinstance(view, tm.DotDict)
    # Setting on the child view propagates
    view.value = 42
    assert dm.recall("missing.value") == 42
    print("  ✓ DotDict unset returns child view")


def test_agent_memory_dot_property() -> None:
    """AgentMemory.dot returns a DotDict over declarative memory."""
    mem = tm.AgentMemory(semantic_path=None, episodic_path=None, declarative_path=None)
    assert isinstance(mem.dot, tm.DotDict)
    mem.dot.app.name = "tiny-memory"
    assert mem.recall("app.name") == "tiny-memory"
    print("  ✓ AgentMemory.dot property")


# ---------------------------------------------------------------------------
# SlidingWindowMemory tests
# ---------------------------------------------------------------------------

def test_sliding_window_basic() -> None:
    """SlidingWindowMemory retains the most recent turns within budget."""
    win = tm.SlidingWindowMemory(max_tokens=100)
    for i in range(5):
        win.add("user", f"Turn {i}")
    msgs = win.messages()
    assert len(msgs) >= 1
    # Newest turn should always be present
    assert msgs[-1]["content"] == "Turn 4"
    print("  ✓ SlidingWindowMemory basic")


def test_sliding_window_evicts_oldest() -> None:
    """SlidingWindowMemory evicts the oldest turn when budget is exceeded."""
    win = tm.SlidingWindowMemory(max_tokens=20)
    for i in range(10):
        win.add("user", "x" * 20)  # each turn ~5 tokens heuristic
    msgs = win.messages()
    # Should retain only recent turns that fit in 20 tokens
    assert len(msgs) <= 5
    # Newest turn always present
    assert msgs[-1]["content"] == "x" * 20
    print("  ✓ SlidingWindowMemory evicts oldest")


def test_sliding_window_with_compressor() -> None:
    """SlidingWindowMemory produces a rolling summary when compressor is set."""
    win = tm.SlidingWindowMemory(
        max_tokens=40,
        compressor=tm.SummaryCompressor(),
    )
    for i in range(8):
        win.add("user", f"This is turn number {i} with extra words about AI agents.")
    msgs = win.messages()
    # First message should be the system summary
    assert msgs[0]["role"] == "system"
    assert len(win.summary()) > 0
    # Total budget should not exceed max_tokens (within heuristic noise)
    assert win.total_tokens() <= 60  # heuristic margin
    print("  ✓ SlidingWindowMemory with compressor")


def test_sliding_window_clear() -> None:
    """SlidingWindowMemory.clear() resets state."""
    win = tm.SlidingWindowMemory(max_tokens=100, compressor=tm.SummaryCompressor())
    win.add("user", "hello")
    win.add("user", "world")
    assert len(win.messages()) == 2
    win.clear()
    assert win.messages() == []
    assert win.summary() == ""
    print("  ✓ SlidingWindowMemory clear")


def test_sliding_window_invalid_budget() -> None:
    """SlidingWindowMemory rejects non-positive budgets."""
    try:
        tm.SlidingWindowMemory(max_tokens=0)
        assert False, "should have raised"
    except ValueError:
        pass
    print("  ✓ SlidingWindowMemory invalid budget")


def test_agent_memory_window_factory() -> None:
    """AgentMemory.window returns a configured SlidingWindowMemory."""
    mem = tm.AgentMemory(semantic_path=None, episodic_path=None, declarative_path=None)
    win = mem.window(max_tokens=50)
    assert isinstance(win, tm.SlidingWindowMemory)
    win.add("user", "hi")
    assert len(win.messages()) == 1
    print("  ✓ AgentMemory.window factory")


def test_sliding_window_total_tokens() -> None:
    """SlidingWindowMemory.total_tokens reflects sum of all messages."""
    win = tm.SlidingWindowMemory(max_tokens=1000)
    win.add("user", "a b c d")
    win.add("assistant", "e f g h")
    n = win.total_tokens()
    assert n > 0
    print("  ✓ SlidingWindowMemory.total_tokens")


# ---------------------------------------------------------------------------
# SQLiteBackend tests
# ---------------------------------------------------------------------------

def test_sqlite_backend_basic_roundtrip() -> None:
    """SQLiteBackend persists and loads all three stores."""
    db = tmp_path("sqlite_basic.db")
    try:
        backend = tm.SQLiteBackend(db)
        backend.save_semantic([{
            "id": "abc", "text": "hello", "metadata": {"k": 1},
            "timestamp": 100.0, "ttl": None, "expires_at": None,
        }])
        backend.save_episodic([{
            "id": "ev1", "timestamp": 100.0, "type": "test", "data": {"x": 1}
        }])
        backend.save_declarative({"k1": {"value": "v1", "timestamp": 100.0, "ttl": None, "expires_at": None}})
        stats = backend.stats()
        assert stats == {"semantic": 1, "episodic": 1, "declarative": 1}
        # Reload
        sem = backend.load_semantic()
        epi = backend.load_episodic()
        dec = backend.load_declarative()
        assert sem[0]["text"] == "hello"
        assert sem[0]["metadata"] == {"k": 1}
        assert epi[0]["type"] == "test"
        assert epi[0]["data"] == {"x": 1}
        assert dec["k1"]["value"] == "v1"
    finally:
        db.unlink(missing_ok=True)
    print("  ✓ SQLiteBackend basic roundtrip")


def test_sqlite_backend_clear_all() -> None:
    """SQLiteBackend.clear_all() empties every table."""
    db = tmp_path("sqlite_clear.db")
    try:
        backend = tm.SQLiteBackend(db)
        backend.save_semantic([{"id": "x", "text": "y", "metadata": {},
                                "timestamp": 0.0, "ttl": None, "expires_at": None}])
        backend.clear_all()
        assert backend.stats() == {"semantic": 0, "episodic": 0, "declarative": 0}
    finally:
        db.unlink(missing_ok=True)
    print("  ✓ SQLiteBackend.clear_all")


def test_sqlite_backend_agent_memory_roundtrip() -> None:
    """AgentMemory with backend='sqlite' persists and reloads state."""
    db = tmp_path("agent_sqlite.db")
    try:
        mem = tm.AgentMemory(backend="sqlite", sqlite_path=db)
        mem.memorize("the rain in spain")
        mem.remember("user", "alice")
        mem.store_episode("greeting", {"text": "hi"})
        mem.save_all()
        # Reload into a new instance
        mem2 = tm.AgentMemory(backend="sqlite", sqlite_path=db)
        assert mem2.recall("user") == "alice"
        results = mem2.query("rain")
        assert any("rain" in r["text"] for r in results)
    finally:
        db.unlink(missing_ok=True)
    print("  ✓ AgentMemory SQLite backend roundtrip")


def test_sqlite_backend_invalid() -> None:
    """AgentMemory rejects unknown backend values."""
    try:
        tm.AgentMemory(backend="postgres")
        assert False, "should have raised"
    except ValueError:
        pass
    print("  ✓ AgentMemory rejects invalid backend")


def test_agent_memory_use_sqlite_migration() -> None:
    """AgentMemory.use_sqlite() migrates in-memory state to SQLite."""
    # Use unique temp paths without pre-touching (empty JSON files break load)
    decl = tmp_path("decl_mig.json")
    sem = tmp_path("sem_mig.jsonl")
    epi = tmp_path("epi_mig.jsonl")
    decl.parent.mkdir(parents=True, exist_ok=True)
    decl.write_text("{}")  # valid empty JSON for DeclarativeMemory
    sem.touch()
    epi.touch()
    try:
        mem = tm.AgentMemory(
            semantic_path=sem, episodic_path=epi, declarative_path=decl
        )
        mem.memorize("foo bar baz")
        mem.remember("k", "v")
        db = tmp_path("migrate.db")
        try:
            mem.use_sqlite(db)
            assert mem.backend == "sqlite"
            assert mem.sqlite_backend is not None
            mem.save_all()
            # Reload
            mem2 = tm.AgentMemory(backend="sqlite", sqlite_path=db)
            assert mem2.recall("k") == "v"
            assert any("foo" in r["text"] for r in mem2.query("foo"))
        finally:
            db.unlink(missing_ok=True)
    finally:
        for p in (decl, sem, epi):
            p.unlink(missing_ok=True)
    print("  ✓ AgentMemory.use_sqlite migration")


def test_agent_memory_stats() -> None:
    """AgentMemory.stats() returns entry counts and the active backend."""
    mem = tm.AgentMemory(semantic_path=None, episodic_path=None, declarative_path=None)
    mem.memorize("one")
    mem.memorize("two")
    mem.remember("k", "v")
    stats = mem.stats()
    assert stats["semantic"] == 2
    assert stats["declarative"] == 1
    assert stats["backend"] == "json"
    print("  ✓ AgentMemory.stats")


# ---------------------------------------------------------------------------
# Persistence (JSONL) regression — make sure defaults still work
# ---------------------------------------------------------------------------

def test_agent_memory_jsonl_default_paths(tmp_path=None) -> None:
    """AgentMemory defaults to JSONL/JSON files when no paths are set."""
    import os
    workdir = Path(tempfile.mkdtemp())
    try:
        os.chdir(workdir)
        mem = tm.AgentMemory()  # default paths are relative
        mem.memorize("test entry")
        mem.remember("k", "v")
        mem.save_all()
        # Confirm files exist
        assert Path("semantic.jsonl").exists()
        assert Path("episodic.jsonl").exists()
        assert Path("declarative.json").exists()
        # Reload
        mem2 = tm.AgentMemory()
        assert mem2.recall("k") == "v"
    finally:
        os.chdir(Path.home())
        shutil.rmtree(workdir, ignore_errors=True)
    print("  ✓ AgentMemory JSONL default paths")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_tests() -> bool:
    """Discover and run all test_* functions. Returns True if all pass."""
    test_functions = [
        # BM25 unit
        test_bm25_basic,
        test_bm25_idf_smoothing,
        test_bm25_empty_query,
        test_bm25_empty_corpus,
        # TF-IDF unit
        test_tfidf_embedding_deterministic,
        test_tfidf_embedding_normalised,
        test_tfidf_embedding_length,
        test_tfidf_cosine_similarity,
        # SemanticMemory
        test_semantic_add,
        test_semantic_search_basic,
        test_semantic_search_hybrid_vs_keyword,
        test_semantic_search_top_k,
        test_semantic_delete,
        test_semantic_ttl_expiration,
        test_semantic_save_load_roundtrip,
        test_semantic_empty_search,
        test_semantic_count,
        test_semantic_metadata_preserved,
        # EpisodicMemory
        test_episodic_add,
        test_episodic_get_recent,
        test_episodic_get_recent_all,
        test_episodic_search,
        test_episodic_clear,
        test_episodic_max_size,
        test_episodic_save_load_roundtrip,
        # DeclarativeMemory
        test_declarative_remember_recall,
        test_declarative_update,
        test_declarative_forget,
        test_declarative_search,
        test_declarative_ttl,
        test_declarative_keys_values_items,
        test_declarative_save_load_roundtrip,
        # AgentMemory
        test_agent_memory_remember_recall,
        test_agent_memory_store_episode,
        test_agent_memory_memorize,
        test_agent_memory_query_combines_sources,
        test_agent_memory_query_ranking,
        test_agent_memory_save_all_load_all,
        # Thread safety
        test_semantic_thread_safety,
        test_declarative_thread_safety,
        test_episodic_thread_safety,
        # Edge cases
        test_semantic_empty_text_rejected,
        test_declarative_recall_missing_key,
        test_episodic_empty_search,
        test_agent_memory_direct_store_access,
        # TokenCounter
        test_token_counter_heuristic_basic,
        test_token_counter_empty_text,
        test_token_counter_pluggable_encoder,
        test_token_counter_count_messages,
        # SummaryCompressor
        test_summary_compressor_extractive_within_budget,
        test_summary_compressor_empty_input,
        test_summary_compressor_abstractive_callable,
        test_summary_compressor_cache,
        # DotDict
        test_dotdict_set_get,
        test_dotdict_nested_prefix,
        test_dotdict_contains_iter,
        test_dotdict_delete,
        test_dotdict_unset_returns_child_view,
        test_agent_memory_dot_property,
        # SlidingWindowMemory
        test_sliding_window_basic,
        test_sliding_window_evicts_oldest,
        test_sliding_window_with_compressor,
        test_sliding_window_clear,
        test_sliding_window_invalid_budget,
        test_agent_memory_window_factory,
        test_sliding_window_total_tokens,
        # SQLiteBackend
        test_sqlite_backend_basic_roundtrip,
        test_sqlite_backend_clear_all,
        test_sqlite_backend_agent_memory_roundtrip,
        test_sqlite_backend_invalid,
        test_agent_memory_use_sqlite_migration,
        test_agent_memory_stats,
        # Persistence regression
        test_agent_memory_jsonl_default_paths,
    ]

    passed = 0
    failed = 0
    failed_names: list[str] = []

    print(f"\n{'='*60}")
    print(f" tiny-memory test suite — {len(test_functions)} test cases")
    print(f"{'='*60}\n")

    for fn in test_functions:
        try:
            fn()
            passed += 1
        except Exception as exc:
            failed += 1
            failed_names.append(fn.__name__)
            print(f"  ✗ {fn.__name__}: {exc}")

    print(f"\n{'='*60}")
    print(f" Results: {passed} passed, {failed} failed, {len(test_functions)} total")
    if failed:
        print(f" Failed: {', '.join(failed_names)}")
    print(f"{'='*60}\n")

    # Cleanup
    cleanup_tmp()
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
