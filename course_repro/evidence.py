"""Deterministic, question-aware context reduction for the course study.

This module never sees attack labels or target phrases. It is an intentionally
small black-box defense, not a semantic guarantee that selected text is safe.
"""

from __future__ import annotations

import re


_STOPWORDS = {
    "a", "all", "an", "and", "are", "by", "can", "do", "does", "for",
    "from", "how", "in", "is", "it", "many", "much", "of", "on", "or",
    "q", "the", "there", "to", "was", "were", "what", "when", "where",
    "which", "who", "with", "you", "your",
}


def _terms(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.casefold())
            if (len(word) >= 3 or word.isdigit()) and word not in _STOPWORDS}


def _select_email(question: str, context: str) -> str:
    if "|CONTENT:" in context:
        header, body = context.split("|CONTENT:", 1)
        # BIPIA's email records have a structured header; an inserted prefix
        # before SUBJECT is external text, not part of that header.
        match = re.search(r"(?:^|\n)(SUBJECT:.*\|EMAIL_FROM:)", header, re.DOTALL)
        if match:
            header = header[match.start(1):]
        prefix = header + "|CONTENT:"
    else:
        prefix, body = "", context
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    query = _terms(question)
    scored = [(len(_terms(line) & query), index, line)
              for index, line in enumerate(lines)]
    chosen = sorted((item for item in scored if item[0] > 0),
                    key=lambda item: (-item[0], item[1]))[:3]
    if "$" in question or any(word in question.casefold() for word in ("paid", "amount", "cost", "price", "total")):
        money_lines = [item for item in scored if re.search(r"[$€£]\s*\d", item[2])]
        chosen = list({item[1]: item for item in chosen + money_lines}.values())
    if not chosen:
        return context
    selected = "\n".join(line for _, _, line in sorted(chosen, key=lambda item: item[1]))
    return (prefix + "\n" if prefix else "") + selected


_WHOLE_TABLE_CUES = (
    "how many", "number of", "count", "total", "sum", "average", "mean",
    "median", "first", "last", "largest", "smallest", "highest", "lowest",
    "most", "least", "all", "list", "between",
    "prior", "previous", "before", "after", "next", "same", "other",
    "above", "below", "only", "single", "top",
)


def _select_table(question: str, context: str) -> str:
    table_lines = [line.strip() for line in context.splitlines()
                   if line.strip().startswith("|") and line.strip().endswith("|")]
    if len(table_lines) < 2:
        return context
    header = table_lines[0]
    rows = [line for line in table_lines[1:] if line.count("|") == header.count("|")]
    if not rows:
        return context
    normalized_question = question.casefold()
    if any(cue in normalized_question for cue in _WHOLE_TABLE_CUES):
        return "\n".join([header] + rows)
    query = _terms(question) - _terms(header)
    ranked = sorted(((len(_terms(row) & query), index) for index, row in enumerate(rows)),
                    key=lambda item: (-item[0], item[1]))
    selected = sorted(index for score, index in ranked[:3] if score > 0)
    if not selected:
        return "\n".join([header] + rows)
    return "\n".join([header] + [rows[index] for index in selected])


def select_evidence(task: str, question: str, context: str) -> str:
    """Keep a small task-relevant excerpt, falling back to full content if unsure."""
    if task == "email":
        return _select_email(question, context)
    if task == "table":
        return _select_table(question, context)
    raise ValueError(f"unknown task: {task}")
