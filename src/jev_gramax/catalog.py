"""Read a Gramax catalog straight from its files: folders with _index.md, articles as .md."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

INDEX = "_index.md"
ROOT_FILE = ".doc-root.yaml"


@dataclass
class Doc:
    path: str  # catalog-relative, no extension: "ai/skills/deck-buddy"; a folder is "ai/skills"
    file: Path
    title: str
    order: float
    props: dict[str, list[str]]  # property name -> values
    body: str  # raw body without frontmatter
    folder: bool
    mtime: float

    @property
    def parent(self) -> str:
        return self.path.rpartition("/")[0]

    @property
    def raw(self) -> str:
        return self.file.read_text(encoding="utf-8")

    @property
    def etag(self) -> str:
        return hashlib.sha256(self.file.read_bytes()).hexdigest()[:12]


def split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    try:
        meta = yaml.safe_load(text[3:end]) or {}
    except yaml.YAMLError:
        meta = {}
    rest = text[end + 4 :]
    return (meta if isinstance(meta, dict) else {}), rest.lstrip("\n")


def _order(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.inf


class Catalog:
    """Rescans the folder on every call (cheap), re-parses only files whose mtime changed."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._cache: dict[Path, Doc] = {}

    def title(self) -> str:
        return str(self._root_meta().get("title") or self.root.name)

    def _root_meta(self) -> dict:
        try:
            meta = yaml.safe_load((self.root / ROOT_FILE).read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return {}
        return meta if isinstance(meta, dict) else {}

    def prop_names(self) -> dict[str, str]:
        """Property id -> display name from .doc-root.yaml; ids without a definition map to themselves."""
        names = {}
        for p in self._root_meta().get("properties") or []:
            if isinstance(p, dict) and p.get("id"):
                names[str(p["id"])] = str(p.get("name") or p["id"])
        return names

    def docs(self) -> dict[str, Doc]:
        names = self.prop_names()
        found: dict[str, Doc] = {}
        seen: set[Path] = set()
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for name in filenames:
                if not name.endswith(".md"):
                    continue
                file = Path(dirpath) / name
                seen.add(file)
                doc = self._load(file, names)
                if doc is not None:
                    found[doc.path] = doc
        self._cache = {f: d for f, d in self._cache.items() if f in seen}
        return found

    def _load(self, file: Path, names: dict[str, str]) -> Doc | None:
        try:
            mtime = file.stat().st_mtime
        except OSError:
            return None
        cached = self._cache.get(file)
        if cached and cached.mtime == mtime:
            return cached
        try:
            meta, body = split_frontmatter(file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            return None
        rel = file.relative_to(self.root)
        folder = file.name == INDEX
        path = (rel.parent if folder else rel.with_suffix("")).as_posix()
        if path == ".":  # a root-level _index.md is not a folder of its own
            path = ""
        props: dict[str, list[str]] = {}
        for p in meta.get("properties") or []:
            if isinstance(p, dict) and p.get("id"):
                value = p.get("value")
                values = value if isinstance(value, list) else [] if value is None else [value]
                props[names.get(str(p["id"]), str(p["id"]))] = [str(v) for v in values]
        default_title = rel.parent.name if folder else rel.stem
        doc = Doc(
            path=path,
            file=file,
            title=str(meta.get("title") or default_title),
            order=_order(meta.get("order")),
            props=props,
            body=body,
            folder=folder,
            mtime=mtime,
        )
        self._cache[file] = doc
        return doc

    def resolve(self, path: str) -> Doc | None:
        """Find a doc by catalog path, with or without .md, a trailing slash or /_index."""
        p = path.strip().strip("/")
        if p.endswith(".md"):
            p = p[:-3]
        if p.endswith("/_index") or p == "_index":
            p = p[: -len("_index")].rstrip("/")
        target = (self.root / p).resolve()
        if target != self.root and self.root not in target.parents:
            return None  # outside the catalog
        return self.docs().get(p)

    def children(self, docs: dict[str, Doc], parent: str) -> list[Doc]:
        kids = [d for d in docs.values() if d.parent == parent and d.path != parent and d.path]
        return sorted(kids, key=lambda d: (d.order, d.title.casefold()))

    def breadcrumb(self, docs: dict[str, Doc], path: str) -> str:
        """Folder titles from the top down to `path`'s parent, joined with ' › '."""
        parts, cur = [], path.rpartition("/")[0]
        while cur:
            d = docs.get(cur)
            parts.append(d.title if d else cur.rpartition("/")[2])
            cur = cur.rpartition("/")[0]
        return " › ".join(reversed(parts))
