"""Offline tests: no network, no Jev key, no embedding model."""

import pytest

from jev_gramax import jev, server
from jev_gramax.catalog import Catalog
from jev_gramax.markup import extract_section, normalize, split_sections
from jev_gramax.search import Index, tokens


def write(root, rel, text):
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text, encoding="utf-8")


@pytest.fixture
def wiki(tmp_path, monkeypatch):
    write(tmp_path, ".doc-root.yaml", "title: Test Wiki\nproperties:\n  - id: t1\n    name: Type\n")
    write(tmp_path, "ai/_index.md", '---\norder: 1\ntitle: AI\n---\n\n<view defs="hierarchy=none" display="List"/>')
    write(tmp_path, "ai/tools/_index.md", "---\norder: 2\ntitle: Tools\n---\n")
    write(
        tmp_path,
        "ai/tools/install-openhands.md",
        "---\norder: 1\ntitle: Install OpenHands\nproperties:\n  - id: t1\n    value:\n      - How-to\n---\n\n"
        "Intro text.\n\n## Установка в WSL\n\nУстановите WSL командой `wsl --install`.\n\n"
        "### Проверка\n\nПроверьте дистрибутивы.\n\n## Docker\n\nRun the container.\n",
    )
    write(tmp_path, "ai/notes.md", "---\norder: 0\ntitle: Notes\n---\n\nGuardrails keep agents safe.\n")
    write(tmp_path, ".git/x.md", "hidden")
    monkeypatch.setenv("GRAMAX_CATALOG", str(tmp_path))
    monkeypatch.setenv("GRAMAX_EMBED_MODEL", "none")
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.setattr(server, "_catalog", None)
    monkeypatch.setattr(server, "_index", None)
    return tmp_path


# markup


def test_normalize_note_video_view_br():
    src = (
        '<view defs="x"/>\n<note type="quote">\n\n**Def:** line one\n\n</note>\n'
        '<video path="https://youtu.be/x"/>\na<br/>b'
    )
    assert normalize(src) == "> [!quote]\n> **Def:** line one\n[video](https://youtu.be/x)\na\nb"


def test_normalize_table_to_markdown():
    src = '<table header="row">\n<tr>\n<td align="left">\n\n**A**\n\n</td>\n<td>B|x</td>\n</tr>\n<tr>\n<td>1<br/>2</td>\n</tr>\n</table>'
    assert normalize(src) == "| **A** | B\\|x |\n| --- | --- |\n| 1 2 |  |"


def test_normalize_table_with_spans_stays_html():
    out = normalize('<table>\n<tr>\n<td colspan="2">x</td>\n</tr>\n</table>')
    assert out == '<table><tr><td colspan="2">x</td></tr></table>'


def test_normalize_leaves_code_alone():
    src = "```\n<note type=\"x\">keep</note>\n```\nand `<br/>` too"
    assert normalize(src) == src


def test_normalize_mermaid_and_images(tmp_path):
    (tmp_path / "d.mermaid").write_text("graph TD; A-->B", encoding="utf-8")
    src = '<mermaid path="./d.mermaid" width="1px"/>\n![](./img.png)\n![x](https://e.com/a.png)'
    assert normalize(src, tmp_path).splitlines()[0] == "[diagram: d.mermaid]"
    assert f"![]({tmp_path.resolve() / 'img.png'})" in normalize(src, tmp_path)
    assert "![x](https://e.com/a.png)" in normalize(src, tmp_path)
    assert "```mermaid\ngraph TD; A-->B\n```" in normalize(src, tmp_path, include_diagrams=True)


def test_split_sections_tracks_parents_and_drops_empty():
    md = "Intro\n\n## **1\\. Top**\n\n### Sub\n\ntext\n\n```\n## not a heading\n```\n"
    sections = split_sections(md)
    assert [(s.heading, s.parents) for s in sections] == [("", ()), ("Sub", ("1. Top",))]
    assert "## not a heading" in sections[1].text


def test_extract_section_includes_subsections():
    md = "## A\n\na\n\n### A1\n\nx\n\n## B\n\nb"
    assert extract_section(md, "a") == "## A\n\na\n\n### A1\n\nx"
    assert extract_section(md, "A1") == "### A1\n\nx"
    assert extract_section(md, "missing") is None


def test_tokens_stem_russian_and_english():
    assert tokens("Установка установите") == ["установк", "установ"]
    assert tokens("Agents running") == ["agent", "run"]


# catalog and tools


def test_catalog_paths_titles_props(wiki):
    docs = Catalog(wiki).docs()
    assert set(docs) == {"ai", "ai/tools", "ai/tools/install-openhands", "ai/notes"}
    assert docs["ai/tools/install-openhands"].props == {"Type": ["How-to"]}
    assert docs["ai"].folder and not docs["ai/notes"].folder


def test_resolve_variants_and_traversal(wiki):
    cat = Catalog(wiki)
    for p in ("ai/notes", "ai/notes.md", "/ai/notes", "ai/notes/"):
        assert cat.resolve(p).path == "ai/notes"
    assert cat.resolve("ai/tools/_index.md").path == "ai/tools"
    assert cat.resolve("../etc/passwd") is None


def test_tree(wiki):
    assert server.wiki_tree() == (
        "Test Wiki\n- AI/  ai  (2)\n  - Notes  ai/notes\n  - Tools/  ai/tools  (1)"
    )
    assert server.wiki_tree("ai/tools", depth=1) == "Tools\n- Install OpenHands  ai/tools/install-openhands"
    assert "is an article" in server.wiki_tree("ai/notes")


def test_read_section_and_folder(wiki):
    out = server.wiki_read("ai/tools/install-openhands#Установка в WSL")
    head, body = out.split("\n\n", 1)
    assert head.startswith("# Install OpenHands\npath: ai/tools/install-openhands · etag: ")
    assert head.endswith("Type: How-to")
    assert body.startswith("## Установка в WSL") and "### Проверка" in body and "Docker" not in body
    assert "Contents:\n- Install OpenHands  ai/tools/install-openhands" in server.wiki_read("ai/tools")
    assert "Headings: Установка в WSL; Проверка; Docker" in server.wiki_read("ai/tools/install-openhands", "nope")
    assert server.wiki_read("ai/tools/install-openhands", raw=True).split("\n\n", 1)[1].startswith("---\norder: 1")


def test_index_refreshes_on_change(wiki):
    idx = Index(Catalog(wiki), None)
    assert idx.search("kubernetes") == []
    write(wiki, "ai/k8s.md", "---\ntitle: K8s\n---\n\nKubernetes cluster notes.\n")
    assert [c.ref for c in idx.search("kubernetes")] == ["ai/k8s"]


def test_search_without_jev(wiki):
    out = server.wiki_search("установить WSL")
    assert out.splitlines()[0] == "(not reranked: JEV_API_KEY is not set)"
    assert out.splitlines()[1].startswith("1. ai/tools/install-openhands#Установка в WSL — Install OpenHands")


def test_search_filters(wiki):
    assert "ai/notes" not in server.wiki_search("guardrails", section="ai/tools")
    assert server.wiki_search("guardrails", property="How-to") == "No results."
    assert "ai/tools/install-openhands" in server.wiki_search("docker", property="how-to")


def test_search_rerank_filters_and_orders(wiki, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "k")
    seen = {}

    def fake_ask(state, questions):
        seen["state"], seen["questions"] = state, questions
        return {k: {"type": "noul", "noul": 0.9 if "Docker" in q["instructions"] else 0.1} for k, q in questions.items()}

    monkeypatch.setattr(jev, "ask", fake_ask)
    out = server.wiki_search("run docker container")
    assert seen["state"] == "run docker container"
    assert all(q["type"] == "noul" for q in seen["questions"].values())
    assert out == "1. ai/tools/install-openhands#Docker — Install OpenHands  (p=0.90)\n   Run the container."


def test_search_rerank_nothing_relevant(wiki, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "k")
    monkeypatch.setattr(jev, "ask", lambda s, qs: {k: {"noul": 0.2} for k in qs})
    assert server.wiki_search("docker").startswith("No passage passed the relevance check. Closest: ")


def test_search_falls_back_when_jev_fails(wiki, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "k")

    def boom(state, questions):
        raise jev.JevError("HTTP 429")

    monkeypatch.setattr(jev, "ask", boom)
    out = server.wiki_search("docker")
    assert out.startswith("(not reranked: Jev did not answer: HTTP 429)\n1. ai/tools/install-openhands#Docker")


def test_relevance_truncates_passages(monkeypatch):
    monkeypatch.setenv("JEV_TEXT_LIMIT", "10")
    captured = {}

    def fake_ask(state, questions):
        captured.update(questions)
        return {k: {"noul": 0.5} for k in questions}

    monkeypatch.setattr(jev, "ask", fake_ask)
    assert jev.relevance("q", [("Where", "x" * 50)]) == [0.5]
    assert captured["p0"]["instructions"].endswith("Passage (Where):\n" + "x" * 10)


def test_yes_probability_rejects_bad_answers():
    with pytest.raises(jev.JevError):
        jev.yes_probability({"type": "noul"})
    with pytest.raises(jev.JevError):
        jev.yes_probability(None)
