"""MCP server: agents read a Gramax wiki through compact, search-first tools; Jev filters the noise.

Environment variables:
  GRAMAX_CATALOG        required, path to the catalog folder (the one with .doc-root.yaml)
  JEV_API_KEY           optional, a TypeSafe key; without it search results are not reranked
  JEV_MODEL             optional, default jev-latest
  JEV_RERANK_THRESHOLD  optional, default 0.5
  JEV_TEXT_LIMIT        optional, characters per passage sent to Jev, default 1500
  GRAMAX_EMBED_MODEL    optional, fastembed model id, or "none" for BM25 only
  GRAMAX_CACHE_DIR      optional, default ~/.cache/jev-gramax
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import jev
from .catalog import Catalog, Doc
from .markup import extract_section, normalize, split_sections
from .search import Chunk, Index, log

CANDIDATES = 10  # chunks sent to Jev for reranking
SNIPPET = 240

INSTRUCTIONS = (
    "Personal wiki (Gramax). To find something, call wiki_search, then wiki_read with the returned "
    "path#heading to read just that section; use wiki_tree to browse. Reading is always allowed. "
    "Never create, edit or delete wiki content unless the user explicitly asks."
)

mcp = MCPServer("gramax", instructions=INSTRUCTIONS)
READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
_catalog: Catalog | None = None
_index: Index | None = None


def catalog() -> Catalog:
    global _catalog
    if _catalog is None:
        _catalog = Catalog(Path(os.environ["GRAMAX_CATALOG"]).expanduser())
    return _catalog


def index() -> Index:
    global _index
    if _index is None:
        _index = Index.from_env(catalog())
    return _index


def _props(doc: Doc) -> str:
    return "; ".join(f"{k}: {', '.join(v)}" for k, v in doc.props.items() if v)


def render_tree(cat: Catalog, path: str = "", depth: int = 2) -> str:
    docs = cat.docs()
    root = path.strip().strip("/")
    if root and root not in docs:
        return f"Not found: {path}"
    if root and not docs[root].folder:
        return f"{root} is an article; use wiki_read."

    def count(prefix: str) -> int:
        return sum(1 for d in docs.values() if not d.folder and d.path.startswith(prefix + "/"))

    lines = [docs[root].title if root else cat.title()]

    def walk(parent: str, level: int) -> None:
        for d in cat.children(docs, parent):
            indent = "  " * (level - 1)
            if d.folder:
                lines.append(f"{indent}- {d.title}/  {d.path}  ({count(d.path)})")
                if level < depth:
                    walk(d.path, level + 1)
            else:
                lines.append(f"{indent}- {d.title}  {d.path}")

    walk(root, 1)
    return "\n".join(lines)


def _snippet(chunk: Chunk) -> str:
    text = chunk.text
    if chunk.heading:  # drop the heading line, it is already in the result header
        text = text.split("\n", 1)[1] if "\n" in text else ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:SNIPPET] + ("…" if len(text) > SNIPPET else "")


def find(query: str, section: str | None, prop: str | None, limit: int) -> str:
    candidates = index().search(query, section, prop, k=max(limit, CANDIDATES))
    if not candidates:
        return "No results."
    note, scored = "", None
    if jev.available():
        try:
            probs = jev.relevance(query, [(c.where, c.text) for c in candidates])
        except jev.JevError as e:
            note = f"(not reranked: Jev did not answer: {e})"
            log(note)
        else:
            scored = sorted(zip(probs, candidates), key=lambda pc: -pc[0])
    else:
        note = "(not reranked: JEV_API_KEY is not set)"

    if scored is not None:
        threshold = jev.rerank_threshold()
        kept = [(p, c) for p, c in scored if p >= threshold][:limit]
        if not kept:
            p, c = scored[0]
            return f"No passage passed the relevance check. Closest: {c.ref} (p={p:.2f})"
        results = kept
    else:
        results = [(None, c) for c in candidates[:limit]]

    lines = [note] if note else []
    for n, (p, c) in enumerate(results, 1):
        score = f"  (p={p:.2f})" if p is not None else ""
        lines.append(f"{n}. {c.ref} — {c.title}{score}\n   {_snippet(c)}")
    return "\n".join(lines)


def read(path: str, heading: str | None = None, raw: bool = False, include_diagrams: bool = False) -> str:
    if "#" in path and not heading:
        path, heading = path.split("#", 1)
    cat = catalog()
    doc = cat.resolve(path)
    if doc is None:
        return f"Not found: {path}. Use wiki_search or wiki_tree to find the path."
    text = doc.raw if raw else normalize(doc.body, doc.file.parent, include_diagrams)
    if heading:
        section = extract_section(text, heading)
        if section is None:
            known = [s.heading for s in split_sections(normalize(doc.body)) if s.heading]
            return f"No heading '{heading}' in {doc.path}. Headings: " + ("; ".join(known) or "none")
        text = section
    header = [f"# {doc.title}", " · ".join(x for x in (f"path: {doc.path}", f"etag: {doc.etag}", _props(doc)) if x)]
    if doc.folder and not heading:
        kids = cat.children(cat.docs(), doc.path)
        if kids:
            text += "\n\nContents:\n" + "\n".join(f"- {k.title}{'/' if k.folder else ''}  {k.path}" for k in kids)
    return "\n".join(header) + "\n\n" + text.strip()


@mcp.tool(annotations=READ_ONLY, structured_output=False)
def wiki_tree(path: str = "", depth: int = 2) -> str:
    """List wiki sections and articles (title, path, article count) under `path`, `depth` levels deep."""
    return render_tree(catalog(), path, depth)


@mcp.tool(annotations=READ_ONLY, structured_output=False)
def wiki_search(query: str, section: str | None = None, property: str | None = None, limit: int = 5) -> str:
    """Search the wiki; returns relevant sections as path#heading with a snippet.

    Any language works. `section` limits search to a folder path, `property` to articles with
    that property value (e.g. "Course"). Read a hit with wiki_read(path#heading).
    """
    return find(query, section, property, limit)


@mcp.tool(annotations=READ_ONLY, structured_output=False)
def wiki_read(path: str, heading: str | None = None, raw: bool = False, include_diagrams: bool = False) -> str:
    """Read an article (or a folder's page) as clean Markdown; pass `heading` or path#heading for one section.

    raw=true returns the file as stored. Diagrams are placeholders unless include_diagrams=true.
    """
    return read(path, heading, raw, include_diagrams)


def main() -> None:
    root = os.environ.get("GRAMAX_CATALOG")
    if not root or not Path(root).expanduser().is_dir():
        log("GRAMAX_CATALOG must point to a Gramax catalog folder")
        sys.exit(1)
    if not jev.available():
        log("JEV_API_KEY is not set: search results will not be reranked")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # model download requests
    mcp.run()  # stdio


if __name__ == "__main__":
    main()
