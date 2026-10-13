"""Gramax markup -> plain Markdown, and splitting Markdown into heading sections."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

CODE_BLOCK = re.compile(r"^(```|~~~)[^\n]*\n.*?^\1[ \t]*$", re.M | re.S)
INLINE_CODE = re.compile(r"`[^`\n]+`")
PLACEHOLDER = re.compile(r"\x00(\d+)\x00")

NOTE = re.compile(r"<note([^>]*)>((?:(?!<note[\s>]).)*?)</note>", re.S)
TABLE = re.compile(r"<table[^>]*>(.*?)</table>", re.S)
ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
CELL = re.compile(r"<t[dh]([^>]*)>(.*?)</t[dh]>", re.S)
VIDEO = re.compile(r'<video\s[^>]*?path="([^"]+)"[^>]*?(?:/>|>\s*</video>)')
MERMAID_FILE = re.compile(r'<mermaid\s[^>]*?path="([^"]+)"[^>]*?(?:/>|>\s*</mermaid>)')
MERMAID_INLINE = re.compile(r"<mermaid[^>]*>(.*?)</mermaid>", re.S)
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)([^)]*)\)")
VIEW = re.compile(r"<view\b[^>]*/>")
BR = re.compile(r"<br\s*/?>")
COLGROUP = re.compile(r"</?colgroup[^>]*>|<col\b[^>]*/?>")
ATTR = re.compile(r'(\w+)="([^"]*)"')
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


def normalize(body: str, base_dir: Path | None = None, include_diagrams: bool = False) -> str:
    """Turn Gramax XML-flavoured Markdown into compact plain Markdown.

    base_dir is the article's folder: relative images become absolute paths and
    mermaid files are read from there. Code blocks and inline code are left untouched.
    """
    kept: list[str] = []

    def keep(m: re.Match) -> str:
        kept.append(m.group(0))
        return f"\x00{len(kept) - 1}\x00"

    text = CODE_BLOCK.sub(keep, body)
    text = INLINE_CODE.sub(keep, text)

    text = VIEW.sub("", text)
    text = COLGROUP.sub("", text)
    text = TABLE.sub(lambda m: _table(m.group(0), m.group(1)), text)
    text = BR.sub("\n", text)
    while True:  # innermost notes first
        new = NOTE.sub(_note, text)
        if new == text:
            break
        text = new
    text = VIDEO.sub(lambda m: f"[video]({m.group(1)})", text)
    text = MERMAID_FILE.sub(lambda m: _mermaid_file(m.group(1), base_dir, include_diagrams), text)
    text = MERMAID_INLINE.sub(
        lambda m: f"```mermaid\n{m.group(1).strip()}\n```" if include_diagrams else "[diagram]", text
    )
    if base_dir is not None:
        text = IMAGE.sub(lambda m: _image(m, base_dir), text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return PLACEHOLDER.sub(lambda m: kept[int(m.group(1))], text)


def _note(m: re.Match) -> str:
    attrs = dict(ATTR.findall(m.group(1)))
    label = f"> [!{attrs.get('type', 'note')}]"
    if attrs.get("title"):
        label += f" {attrs['title']}"
    lines = m.group(2).strip().splitlines()
    return "\n".join([label] + [f"> {line}".rstrip() for line in lines])


def _cell_text(html: str) -> str:
    text = BR.sub(" ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("|", "\\|")


def _table(whole: str, inner: str) -> str:
    rows = [CELL.findall(r) for r in ROW.findall(inner)]
    rows = [r for r in rows if r]
    if not rows:
        return ""
    if any("rowspan" in attrs or "colspan" in attrs for r in rows for attrs, _ in r):
        return re.sub(r"\s*\n\s*", "", re.sub(r'\s(align|width)="[^"]*"', "", whole))
    width = max(len(r) for r in rows)
    cells = [[_cell_text(c) for _, c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(cells[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in cells[1:]]
    return "\n" + "\n".join(lines) + "\n"


def _mermaid_file(path: str, base_dir: Path | None, include: bool) -> str:
    name = Path(path).name
    if not include or base_dir is None:
        return f"[diagram: {name}]"
    try:
        code = (base_dir / path).read_text(encoding="utf-8").strip()
    except OSError:
        return f"[diagram: {name} (file not found)]"
    return f"```mermaid\n{code}\n```"


def _image(m: re.Match, base_dir: Path) -> str:
    alt, src, rest = m.groups()
    if re.match(r"^[a-z]+:", src) or src.startswith("/"):
        return m.group(0)
    return f"![{alt}]({(base_dir / src).resolve()}{rest})"


def clean_heading(text: str) -> str:
    text = re.sub(r"\\([^\w\s])", r"\1", text)  # Markdown escapes such as "1\."
    return re.sub(r"[*_`]", "", text).strip().rstrip(":").strip()


@dataclass
class Section:
    heading: str  # "" for the text before the first heading
    level: int  # 0 for the text before the first heading
    text: str  # includes the heading line
    parents: tuple[str, ...] = ()  # enclosing headings, outermost first


def _headings(lines: list[str], max_level: int) -> list[tuple[int, int, str]]:
    """(line index, level, clean text) of headings outside code fences."""
    out, fence = [], None
    for i, line in enumerate(lines):
        f = FENCE.match(line)
        if f:
            fence = None if fence == f.group(1) else (fence or f.group(1))
            continue
        if fence:
            continue
        h = HEADING.match(line)
        if h and len(h.group(1)) <= max_level:
            out.append((i, len(h.group(1)), clean_heading(h.group(2))))
    return out


def split_sections(md: str, max_level: int = 3) -> list[Section]:
    """Flat split at every heading up to max_level; each section stops at the next heading.

    Sections with nothing but a heading line are dropped; their heading lives on in `parents`.
    """
    lines = md.splitlines()
    heads = _headings(lines, max_level)
    bounds = [(0, 0, "")] + heads
    sections, stack = [], []  # stack: (level, heading) of open headings
    for n, (start, level, heading) in enumerate(bounds):
        end = bounds[n + 1][0] if n + 1 < len(bounds) else len(lines)
        while stack and stack[-1][0] >= level:
            stack.pop()
        parents = tuple(h for _, h in stack)
        if level:
            stack.append((level, heading))
        body = lines[start + 1 : end] if level else lines[start:end]
        if "\n".join(body).strip():
            sections.append(Section(heading, level, "\n".join(lines[start:end]).strip(), parents))
    return sections


def extract_section(md: str, heading: str) -> str | None:
    """The section under `heading` with its subsections. Exact match first, then substring."""
    lines = md.splitlines()
    heads = _headings(lines, 6)
    want = clean_heading(heading).casefold()
    match = next((h for h in heads if h[2].casefold() == want), None) or next(
        (h for h in heads if want in h[2].casefold()), None
    )
    if match is None:
        return None
    start, level, _ = match
    end = next((i for i, lv, _ in heads if i > start and lv <= level), len(lines))
    return "\n".join(lines[start:end]).strip()

