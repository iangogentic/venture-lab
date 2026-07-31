"""Local semantic memory: embeddings plus a vector store, both on this machine.

Two consumers, one representation. `collect-evidence` embeds candidates to drop
near-duplicates that id-based dedup cannot see (the same complaint cross-posted
to two sources has two ids), and `op recall` searches everything past runs kept.
Neither ever blocks a run: when the model or the sqlite extension is missing the
layer raises `MemoryUnavailableError` and callers proceed as if it did not exist.
"""

from app.memory.embedder import Embedder, StaticModelEmbedder, cosine, default_embedder
from app.memory.store import EMBEDDING_DIMENSIONS, MemoryStore, RecallHit

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "Embedder",
    "MemoryStore",
    "RecallHit",
    "StaticModelEmbedder",
    "cosine",
    "default_embedder",
]
