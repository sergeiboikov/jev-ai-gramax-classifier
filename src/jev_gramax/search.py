"""Hybrid search over heading sections: BM25 (ru/en stemming) + multilingual embeddings, fused with RRF."""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import snowballstemmer

from .catalog import Catalog, Doc
from .markup import normalize, split_sections

DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RRF_K = 60
WORD = re.compile(r"\w+", re.U)
CYRILLIC = re.compile(r"[а-яё]")

_ru = snowballstemmer.stemmer("russian")
_en = snowballstemmer.stemmer("english")


def log(msg: str) -> None:
    print(f"[jev-gramax] {msg}", file=sys.stderr, flush=True)


def tokens(text: str) -> list[str]:
    words = [w for w in WORD.findall(text.casefold()) if len(w) > 1 or w.isdigit()]
    return [(_ru if CYRILLIC.search(w) else _en).stemWord(w) for w in words]


@dataclass
class Chunk:
    path: str
    heading: str  # "" for the text before the first heading
    title: str  # article title
    where: str  # "Section › Article › Heading", for people and for Jev
    text: str  # normalized Markdown, diagrams as placeholders
    props: dict[str, list[str]]

    @property
    def ref(self) -> str:
        return f"{self.path}#{self.heading}" if self.heading else self.path

    @property
    def passage(self) -> str:
        return f"{self.where}\n{self.text}"


def doc_chunks(doc: Doc, breadcrumb: str) -> list[Chunk]:
    md = normalize(doc.body, doc.file.parent)
    base = f"{breadcrumb} › {doc.title}" if breadcrumb else doc.title
    out = []
    for s in split_sections(md):
        where = " › ".join((base, *s.parents, s.heading) if s.heading else (base,))
        out.append(Chunk(doc.path, s.heading, doc.title, where, s.text, doc.props))
    return out


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 0
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t in query:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / (self.avg or 1)))
            out.append(s)
        return out


class Embedder:
    """fastembed model loaded on first use; vectors cached on disk by passage hash."""

    def __init__(self, model_name: str, cache_dir: Path):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._failed: str | None = None
        self._vecs: dict[str, list[float]] | None = None
        slug = re.sub(r"[^\w.-]+", "_", model_name)
        self.file = cache_dir / f"emb-{slug}.npz"

    @property
    def disabled(self) -> str | None:
        return self._failed

    def _load(self):
        if self._model is None and self._failed is None:
            try:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(self.model_name, cache_dir=str(self.cache_dir / "models"))
            except Exception as e:  # noqa: BLE001 - offline, unknown model, broken onnxruntime
                self._failed = f"embeddings unavailable ({e.__class__.__name__}: {e})"
                log(self._failed)
        return self._model

    def _read_cache(self) -> dict:
        if self._vecs is None:
            self._vecs = {}
            try:
                import numpy as np

                with np.load(self.file) as z:
                    self._vecs = dict(zip(z["keys"].tolist(), z["vecs"]))
            except (OSError, KeyError, ValueError):
                pass
        return self._vecs

    def _write_cache(self) -> None:
        import numpy as np

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        keys = list(self._vecs)
        buf = io.BytesIO()
        np.savez(buf, keys=np.array(keys), vecs=np.array([self._vecs[k] for k in keys], dtype=np.float32))
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            f.write(buf.getvalue())
        os.replace(tmp, self.file)

    def passages(self, texts: list[str]):
        """Unit vectors for texts, or None if embeddings are unavailable."""
        model = self._load()
        if model is None:
            return None
        import numpy as np

        cache = self._read_cache()
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in cache]
        if missing:
            for (k, _), v in zip(missing, model.passage_embed([t for _, t in missing])):
                cache[k] = np.asarray(v, dtype=np.float32) / (np.linalg.norm(v) or 1)
            try:
                self._write_cache()
            except OSError as e:
                log(f"could not write the embedding cache: {e}")
        return np.stack([cache[k] for k in keys]) if keys else np.zeros((0, 1))

    def query(self, text: str):
        model = self._load()
        if model is None:
            return None
        import numpy as np

        v = np.asarray(next(iter(model.query_embed([text]))), dtype=np.float32)
        return v / (np.linalg.norm(v) or 1)


class Index:
    def __init__(self, catalog: Catalog, embedder: Embedder | None):
        self.catalog = catalog
        self.embedder = embedder
        self._chunks: dict[str, tuple[float, str, list[Chunk]]] = {}  # path -> (mtime, breadcrumb, chunks)

    @classmethod
    def from_env(cls, catalog: Catalog) -> "Index":
        name = os.environ.get("GRAMAX_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        cache = Path(os.environ.get("GRAMAX_CACHE_DIR", "~/.cache/jev-gramax")).expanduser()
        return cls(catalog, None if name.lower() == "none" else Embedder(name, cache))

    def chunks(self) -> list[Chunk]:
        docs = self.catalog.docs()
        fresh: dict[str, tuple[float, str, list[Chunk]]] = {}
        for path, doc in docs.items():
            crumb = self.catalog.breadcrumb(docs, path) if path else ""
            cached = self._chunks.get(path)
            if cached and cached[0] == doc.mtime and cached[1] == crumb:
                fresh[path] = cached
            else:
                fresh[path] = (doc.mtime, crumb, doc_chunks(doc, crumb))
        self._chunks = fresh
        return [c for _, _, cs in fresh.values() for c in cs]

    def search(self, query: str, section: str | None = None, prop: str | None = None, k: int = 10) -> list[Chunk]:
        """Top-k chunks by reciprocal rank fusion of BM25 and embedding similarity."""
        chunks = self.chunks()
        if section:
            prefix = section.strip("/")
            chunks = [c for c in chunks if c.path == prefix or c.path.startswith(prefix + "/")]
        if prop:
            want = prop.casefold()
            chunks = [c for c in chunks if any(want == v.casefold() for vs in c.props.values() for v in vs)]
        if not chunks:
            return []

        rankings = []
        bm = BM25([tokens(c.passage) for c in chunks]).scores(tokens(query))
        rankings.append([i for i in sorted(range(len(chunks)), key=lambda i: -bm[i]) if bm[i] > 0])

        if self.embedder is not None:
            vecs = self.embedder.passages([c.passage[:2000] for c in chunks])
            q = self.embedder.query(query) if vecs is not None else None
            if q is not None:
                sims = vecs @ q
                rankings.append(sorted(range(len(chunks)), key=lambda i: -sims[i]))

        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1 / (RRF_K + rank + 1)
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [chunks[i] for i in best]
