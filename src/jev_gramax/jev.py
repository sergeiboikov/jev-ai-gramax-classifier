"""Jev questions over TypeSafe's System One API: typed answers, no generated text."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
TIMEOUT = 30


class JevError(Exception):
    pass


def model() -> str:
    return os.environ.get("JEV_MODEL", "jev-latest")


def text_limit() -> int:
    return int(os.environ.get("JEV_TEXT_LIMIT", "1500"))


def rerank_threshold() -> float:
    return float(os.environ.get("JEV_RERANK_THRESHOLD", "0.5"))


def available() -> bool:
    return bool(os.environ.get("JEV_API_KEY"))


def ask(state: str, questions: dict[str, dict]) -> dict[str, dict]:
    """Send named questions about one state; returns the `answers` object. Raises JevError."""
    key = os.environ.get("JEV_API_KEY")
    if not key:
        raise JevError("JEV_API_KEY is not set")
    body = json.dumps({"model": model(), "state": state, "questions": questions}).encode()
    req = urllib.request.Request(
        TYPESAFE_URL,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            res = json.load(r)
    except urllib.error.HTTPError as e:
        raise JevError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise JevError(str(getattr(e, "reason", e))) from e
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise JevError(f"response was not valid JSON: {e}") from e
    answers = res.get("answers") if isinstance(res, dict) else None
    if not isinstance(answers, dict):
        raise JevError(f"unexpected response: {str(res)[:300]}")
    return answers


def yes_probability(answer: dict | None) -> float:
    """Probability of "yes" from a yes/no (noul) answer."""
    if not isinstance(answer, dict):
        raise JevError(f"missing answer: {answer}")
    for k in ("noul", "probability", "yes"):
        v = answer.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    raise JevError(f"no probability in answer: {answer}")


RERANK_INSTRUCTIONS = (
    "The state is a request from an AI agent searching a personal wiki. Does the wiki passage below "
    "contain information that helps answer that request? Yes means the passage is on topic and useful; "
    "no means it is unrelated or only shares keywords.\n\nPassage ({where}):\n{text}"
)


def relevance(query: str, passages: list[tuple[str, str]]) -> list[float]:
    """One request, one yes/no question per passage. passages: (where, text). Returns P(yes) each."""
    limit = text_limit()
    questions = {
        f"p{i}": {"type": "noul", "instructions": RERANK_INSTRUCTIONS.format(where=where, text=text[:limit])}
        for i, (where, text) in enumerate(passages)
    }
    answers = ask(query, questions)
    return [yes_probability(answers.get(f"p{i}")) for i in range(len(passages))]
