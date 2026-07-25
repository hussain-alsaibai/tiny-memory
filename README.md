# tiny-memory

> **Persistent conversation memory for AI agents — zero dependencies, one file.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](#)
[![Tests](https://img.shields.io/badge/tests-73%20passing-brightgreen.svg)](#testing)
[![Part of tiny-* ecosystem](https://img.shields.io/badge/tiny%E2%98%8C-ecosystem-purple.svg)](https://github.com/hussain-alsaibai)

`tiny-memory` is the smallest possible answer to the hardest problem in
production AI agents: **remembering what already happened**.

It is a single-file Python module (no `pip install` of vector databases, no
API keys, no embeddings service) that gives you:

- **Sliding window** conversation memory with **token budget** enforcement
- **Extractive summary compression** when the window overflows
- **Semantic search** (BM25 + TF-IDF n-gram cosine) without an embeddings API
- **Episodic event log** (LRU, capped, JSONL-backed)
- **Declarative key/value facts** with optional TTL
- **Dot-notation access** (`mem.dot.user.name = "hussain"`)
- **Pluggable backends**: JSON / JSONL files (default) or single-file SQLite
- Thread-safe; pure stdlib; `pip install` is the *whole* install story.

---

## Why agent memory matters in 2026

In 2026 the difference between a demo and a product is almost always the same
thing: **memory**. Stateless LLM calls produce chatbots. Persistent, queryable
memory produces agents that ship to production.

The problem is that the "obvious" choices all carry significant cost:

| Approach | Cost |
|----------|------|
| Roll your own `dict` | No search, no persistence, no structure, no thread safety, no TTL. You re-invent this badly within a week. |
| `mem0`, `letta`, `zep` | Powerful but pull in Pinecone / Qdrant / Chroma, an LLM API for summarisation, and 200 MB of transitive dependencies. |
| LangChain `ConversationBufferMemory` | In-memory only, no persistence, no semantic search, no eviction policy. Resets on every restart. |
| Hand-roll FAISS | You become responsible for the persistence layer, the schema, the migrations, and the embedding bill. |

`tiny-memory` takes a third path: a **real** memory system (semantic search,
eviction policies, persistence, thread safety, compression) implemented in
**one pure-Python file** using only `hashlib`, `json`, `math`, `collections`,
`uuid`, `threading`, `time`, `pathlib`, `re`, and `sqlite3` (stdlib).

You can read the entire implementation in an afternoon. You can vendor it.
You can ship it.

---

## Install

```bash
pip install tiny-memory
```

Or just copy `tiny_memory.py` into your project — it's one file, zero deps.

```python
from tiny_memory import AgentMemory
```

---

## 60-second tour

```python
from tiny_memory import AgentMemory, SummaryCompressor, TokenCounter

mem = AgentMemory()  # JSONL/JSON persistence in current directory

# 1. Declarative facts (key/value, TTL, dot-notation)
mem.remember("user.name", "Hussain")
mem.remember("user.tier", "pro", ttl=3600)
print(mem.recall("user.name"))                # "Hussain"
mem.dot.user.name                              # "Hussain"
mem.dot.user.tier                              # "pro"
mem.dot.app.name = "tiny-memory"               # write through dot-notation

# 2. Episodic events (LRU sliding window of recent actions)
mem.store_episode("tool_call", {"tool": "search", "q": "BM25"})
mem.store_episode("user_message", "what was I working on yesterday?")

# 3. Semantic text (BM25 + TF-IDF n-gram, no embeddings API)
mem.memorize("The user prefers concise answers and dislikes emojis.")
mem.memorize("Project uses Postgres 16 with pgvector for vector storage.")

# 4. Unified query across all three stores
for hit in mem.query("what database does the project use?"):
    print(f"[{hit['source']}] score={hit['score']:.2f}  {hit['text'][:60]}")

# 5. Token-budgeted conversation window with auto-summarisation
win = mem.window(
    max_tokens=2000,
    token_counter=TokenCounter(),                    # heuristic, swap for tiktoken
    compressor=SummaryCompressor(),                  # extractive (zero-deps) summarizer
)
win.add("user", "Hi, I'm new to the project.")
win.add("assistant", "Welcome! Let me know what you'd like to build.")
# ... 200 turns later, the oldest have been summarised into a system message
prompt_msgs = win.messages()   # ready to send to your LLM
```

---

## The five memory primitives

| Class | What it stores | Search | Eviction | Persistence |
|-------|---------------|--------|----------|-------------|
| `SemanticMemory` | Text + metadata | BM25 + TF-IDF cosine (hybrid) | TTL only | JSONL |
| `EpisodicMemory` | Event log (type, data) | Substring over type/data | LRU `deque(maxlen=N)` | JSONL |
| `DeclarativeMemory` | Key/value facts | Substring over key/value | TTL only | JSON |
| `SlidingWindowMemory` | Conversation turns (role, content) | — | Token-budget + summary | In-memory (compose with episodic) |
| `AgentMemory` | Facade over all four | Unified | Inherits from each | JSON/JSONL or SQLite |

Each primitive is independently usable. `AgentMemory` is convenience, not a
requirement.

---

## Token budgets & summary compression

The single most useful feature in production. Conversation context grows
without bound; LLM context windows don't. `SlidingWindowMemory` keeps the
newest turns and rolls the oldest into a compressed summary:

```python
from tiny_memory import SlidingWindowMemory, SummaryCompressor, TokenCounter

# Drop-in tiktoken encoder for exact counts
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
counter = TokenCounter(lambda text: enc.encode(text))

win = SlidingWindowMemory(
    max_tokens=4000,                       # hard ceiling on prompt size
    token_counter=counter,                 # real token counts
    compressor=SummaryCompressor(),        # extractive (default, zero deps)
    summary_ratio=0.25,                    # 25% of budget reserved for summary
)

# Or plug in your own LLM-backed summariser
def my_llm_summariser(msgs):
    return call_my_llm("Summarise:", msgs)

win = SlidingWindowMemory(
    max_tokens=4000,
    compressor=SummaryCompressor(summarizer=my_llm_summariser),
)
```

The default `SummaryCompressor` is **extractive**: it ranks sentences by
TF-IDF weight and packs the highest-ranked ones into the target budget,
deterministically. No LLM call, no latency, no surprise cost.

---

## Dot-notation access

```python
mem = AgentMemory()
mem.dot.user.name = "Hussain"
mem.dot.user.email = "hussain@alsaibai.cloud"
mem.dot.project.stack = ["python", "rust", "postgres"]

print(mem.dot.user.name)              # "Hussain"
print(mem.dot.user)                   # DotDict('user', {'name': 'Hussain', 'email': 'hussain@alsaibai.cloud'})
print("user.name" in mem.dot.user)    # True
```

Behind the scenes `mem.dot.user.name` writes to declarative memory under the
key `user.name`. Nested groups share a prefix; `del mem.dot.user.email` removes
the underlying fact. Works equally well with `mem.recall("user.name")` if you
prefer explicit string keys.

---

## SQLite backend

For multi-process agents or just to avoid scattering three files around the
working directory:

```python
# At construction
mem = AgentMemory(backend="sqlite", sqlite_path="agent_memory.db")

# Or migrate an existing in-memory state
mem = AgentMemory()                    # JSONL files
mem.memorize("hello world")
mem.use_sqlite("agent_memory.db")      # copies current state, switches backend

mem.save_all()                         # one file, atomic, queryable
```

The SQLite path uses WAL mode for concurrent readers, so multiple agent
processes can share the same memory file safely.

---

## Comparison vs raw `dict`

What you lose by using a `dict` to remember things:

| Need | Raw `dict` | `AgentMemory` |
|------|------------|---------------|
| Store 10k facts | OK | OK |
| Find facts containing "postgres" | `for k, v in d.items(): if ...` (O(n)) | `mem.query("postgres")` (BM25 + cosine, ranked) |
| Survive a process restart | `json.dump` yourself | `mem.save_all()` |
| Know when a fact was set | DIY timestamps | Built-in `timestamp` + TTL |
| Avoid OOM from unbounded growth | DIY eviction | LRU deque + TTL + token budget |
| Use it from multiple threads | DIY locking | `threading.RLock` everywhere |
| Compress old context to fit a prompt | DIY | `SlidingWindowMemory` + `SummaryCompressor` |
| Migrate to a real database later | Rewrite everything | `mem.use_sqlite("agent.db")` |

The `dict` is free. The memory system is also free (MIT, single file). The
difference is what you ship.

---

## Performance notes

`tiny-memory` is deliberately small. A few numbers from a 2026 M-class laptop
(Apple silicon, Python 3.11):

| Operation | Cost (median) | Notes |
|-----------|---------------|-------|
| `DeclarativeMemory.remember / recall` | ~1 µs | Plain dict + lock |
| `EpisodicMemory.add` | ~2 µs | `deque.append` |
| `SemanticMemory.add` (1k corpus) | ~250 µs | TF-IDF vocab rebuild amortised |
| `SemanticMemory.search` (1k corpus, hybrid) | ~3 ms | BM25 + cosine |
| `BM25.score` (1k docs) | ~1.5 ms | Pure Python, no NumPy |
| `SlidingWindowMemory.add` | ~10 µs | Plus compressor when overflow triggers |
| `SummaryCompressor.compress` (extractive, 50 sentences → 200 tokens) | ~3 ms | Pure Python |
| `SQLiteBackend.save_semantic` (1k rows) | ~6 ms | stdlib `sqlite3`, WAL |

These are the right numbers for **per-turn** agent use. If you need to
search millions of vectors per query, you want a real vector database; that's
what `mem0` and friends exist for. `tiny-memory` is the right tool for
**conversation-scale** memory: dozens to thousands of entries per session.

The BM25 implementation is intentionally simple — no NumPy dependency. If
you want real vector search later, swap `SemanticMemory` for your favourite
embeddings store; the rest of the API stays the same.

---

## Thread safety

Every public method on every class holds an `RLock` for the duration of the
operation. The included test suite spins up 10 threads × 100 ops against each
store and asserts no data loss.

For **multi-process** use (separate agent processes sharing a memory file),
use the SQLite backend.

---

## Testing

```bash
python -m pytest test_tiny_memory.py -v
```

The suite is dependency-free and runs in under a second:

```
============================== 73 passed in 0.54s ==============================
```

It covers BM25, TF-IDF embeddings, semantic/episodic/declarative stores,
the agent facade, the new token budget / sliding window / summary compressor
classes, the DotDict wrapper, the SQLite backend, and thread safety.

---

## Architecture

```
                    ┌────────────────────────────────────┐
                    │           AgentMemory              │
                    │  (Facade — single entry point)     │
                    └────────────────────────────────────┘
                       │              │              │
                       ▼              ▼              ▼
                ┌────────────┐ ┌────────────┐ ┌──────────────┐
                │  Semantic  │ │  Episodic  │ │ Declarative  │
                │  Memory    │ │  Memory    │ │ Memory       │
                │            │ │  (LRU      │ │  (TTL        │
                │  BM25 +    │ │   deque)   │ │   key-value) │
                │  TF-IDF    │ │            │ │              │
                └────────────┘ └────────────┘ └──────────────┘
                       │              │              │
                       ▼              ▼              ▼
                ┌────────────────────────────────────────────┐
                │ SQLiteBackend  (backend="sqlite")          │
                │   or JSON / JSONL files (default)          │
                └────────────────────────────────────────────┘

  Plus (independent primitives, composable):
    • TokenCounter       (heuristic or pluggable encoder)
    • SummaryCompressor  (extractive default, or LLM-backed)
    • SlidingWindowMemory (token-budgeted conversation window)
    • DotDict            (dot-notation over declarative)
```

---

## License

MIT © 2026 Hussain Alsaibai. See [LICENSE](LICENSE).

Part of the [`tiny-*` ecosystem](https://github.com/hussain-alsaibai) of
zero-dependency, single-file Python utilities for AI agents.