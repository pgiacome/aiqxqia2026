"""Check that every identifier in a BibTeX file resolves to a real record.

Fabricated citations are the characteristic failure mode of AI-assisted writing, so
this is a gate, not a courtesy: it runs before submission and any entry that does not
resolve is removed from ``refs.bib`` or marked unverified in the manuscript.

It checks existence and, where the record exposes a title, that the title in the
BibTeX entry actually matches the record. An identifier that resolves to a *different*
paper is worse than one that does not resolve at all, because it looks fine.

Exit status is non-zero if any entry fails, so it can be wired into CI.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["Entry", "check_entry", "parse_bib"]

USER_AGENT = "padic-kernel-refcheck/1.0 (academic citation verification)"
TIMEOUT = 30


@dataclass(frozen=True)
class Entry:
    key: str
    title: str
    arxiv: str | None
    doi: str | None


def _normalise(text: str) -> str:
    """Reduce a title to lowercase ASCII alphanumerics for comparison.

    The BibTeX side writes accents as LaTeX escapes (``Schr{\\"o}dinger``) while the
    upstream record uses Unicode (``Schrodinger`` with a combining diaeresis). Both must
    reduce to the same string, or every accented title reports a spurious mismatch and
    the check gets ignored -- which would defeat its purpose.
    """
    text = re.sub(r"\\[a-zA-Z]+", " ", text)  # \H, \v, \c, \textit, ...
    text = re.sub(r"\\.", " ", text)  # \", \', \~, \`
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _field(body: str, name: str) -> str | None:
    """Read one BibTeX field, tolerating a single level of nested braces."""
    pattern = rf"{name}\s*=\s*[{{\"]((?:[^{{}}]|\{{[^{{}}]*\}})*)[}}\"]"
    m = re.search(pattern, body, re.I)
    return m.group(1).strip() if m else None


def parse_bib(path: Path) -> list[Entry]:
    """A deliberately small BibTeX reader: enough for cite key, title, arXiv ID, DOI."""
    text = path.read_text(encoding="utf-8")
    entries: list[Entry] = []
    for match in re.finditer(r"@\w+\s*\{\s*([^,]+),", text):
        key = match.group(1).strip()
        start = match.end()
        nxt = text.find("\n@", start)
        body = text[start : nxt if nxt != -1 else len(text)]

        title = _field(body, "title") or ""
        doi = _field(body, "doi")
        arxiv = _field(body, "eprint")
        if arxiv is None:
            note = " ".join(
                filter(
                    None,
                    [_field(body, "note"), _field(body, "journal"), _field(body, "url") or ""],
                )
            )
            m = re.search(r"(\d{4}\.\d{4,5})", note)
            arxiv = m.group(1) if m else None
        if arxiv:
            arxiv = re.sub(r"^arxiv[:/]*", "", arxiv.strip(), flags=re.I)
        entries.append(Entry(key=key, title=title, arxiv=arxiv, doi=doi))
    return entries


def _get(url: str, accept: str | None = None) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if accept:
        request.add_header("Accept", accept)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), b""
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.warning("network error for %s: %s", url, exc)
        return 0, b""


def _check_arxiv(entry: Entry) -> tuple[bool, str]:
    query = urllib.parse.urlencode({"id_list": entry.arxiv})
    status, body = _get(f"https://export.arxiv.org/api/query?{query}")
    if status != 200 or not body:
        return False, f"arXiv {entry.arxiv}: HTTP {status}"
    # The feed's first <title> is the query echo; the entry's own title is the second.
    titles = re.findall(rb"<title>(.*?)</title>", body, re.S)
    if len(titles) < 2:
        return False, f"arXiv {entry.arxiv}: no entry returned (withdrawn or wrong id?)"
    found = " ".join(titles[1].decode("utf-8", "replace").split())
    if entry.title and _normalise(entry.title) != _normalise(found):
        return False, (
            f"arXiv {entry.arxiv}: TITLE MISMATCH"
            f"\n      bib:  {entry.title}\n      real: {found}"
        )
    return True, f"arXiv {entry.arxiv}: {found}"


def _check_doi(entry: Entry) -> tuple[bool, str]:
    url = f"https://doi.org/{urllib.parse.quote(entry.doi or '', safe='/:')}"
    status, body = _get(url, accept="application/vnd.citationstyles.csl+json")
    if status != 200 or not body:
        return False, f"DOI {entry.doi}: HTTP {status}"
    try:
        record = json.loads(body)
    except json.JSONDecodeError:
        return True, f"DOI {entry.doi}: resolves (no CSL metadata)"
    found = record.get("title")
    if isinstance(found, list):
        found = found[0] if found else ""
    if entry.title and found and _normalise(entry.title) != _normalise(str(found)):
        return False, (
            f"DOI {entry.doi}: TITLE MISMATCH"
            f"\n      bib:  {entry.title}\n      real: {found}"
        )
    return True, f"DOI {entry.doi}: {found or 'resolves'}"


def check_entry(entry: Entry) -> tuple[bool, list[str]]:
    """Verify one entry. An entry with no identifier at all cannot pass."""
    if not entry.arxiv and not entry.doi:
        return False, ["no arXiv id and no DOI: cannot be verified"]
    notes, ok = [], True
    if entry.arxiv:
        passed, note = _check_arxiv(entry)
        ok &= passed
        notes.append(note)
        time.sleep(3.0)  # arXiv asks for one request every three seconds
    if entry.doi:
        passed, note = _check_doi(entry)
        ok &= passed
        notes.append(note)
    return ok, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bib", type=Path, nargs="?", default=Path("papers/aiqxqia2026/refs.bib"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    entries = parse_bib(args.bib)
    if not entries:
        logger.warning("no entries found in %s", args.bib)
        return 0

    failures = []
    for entry in entries:
        ok, notes = check_entry(entry)
        mark = "PASS" if ok else "FAIL"
        logger.info("%s  %s", mark, entry.key)
        for note in notes:
            logger.info("      %s", note)
        if not ok:
            failures.append(entry.key)

    logger.info("%d entries, %d failed", len(entries), len(failures))
    if failures:
        logger.error("unverified: %s", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
