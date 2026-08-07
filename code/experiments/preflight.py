"""Mechanical submission gate for the AIQxQIA 2026 paper.

Everything here is a check that a human would otherwise have to remember. Run it
immediately before uploading:

    uv run python -m experiments.preflight

Exit status is non-zero if any check fails, so it can gate a release. It deliberately
does NOT check the things only the author can supply -- see notes/submission-checklist.md
for those.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["run_checks"]

MIN_BODY_PAGES = 10


def _pdf_text(pdf: Path) -> list[str]:
    out = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True
    ).stdout
    return [p for p in out.split("\f") if p.strip()]


def run_checks(paper: Path) -> list[tuple[bool, str]]:
    """Return (passed, message) for each check."""
    results: list[tuple[bool, str]] = []
    sources = [paper / "main.tex", *sorted((paper / "sections").glob("*.tex"))]
    body = "\n".join(p.read_text(encoding="utf-8") for p in sources)

    placeholders = re.findall(r"TODO-[A-Z]+", body)
    results.append(
        (not placeholders, f"author placeholders: {sorted(set(placeholders)) or 'none'}")
    )

    todos = len(re.findall(r"\\todo\{", body))
    results.append((todos == 0, f"unresolved \\todo markers: {todos}"))

    pdf = paper / "main.pdf"
    if not pdf.exists():
        results.append((False, "main.pdf missing; run make"))
        return results

    pages = _pdf_text(pdf)
    refs_page = next(
        (i for i, p in enumerate(pages, 1) if re.search(r"^\s*References\s*$", p, re.M)), None
    )
    if refs_page is None:
        results.append((False, "no References heading found in the PDF"))
    else:
        n = refs_page - 1
        note = f"body length {n} pages excluding references (need >= {MIN_BODY_PAGES})"
        results.append((n >= MIN_BODY_PAGES, note))

    log = paper / "main.log"
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="replace")
        undefined = re.findall(r"(?:Reference|Citation) `([^']+)'", text)
        results.append(
            (not undefined, f"undefined references/citations: {sorted(set(undefined)) or 'none'}")
        )
    else:
        results.append((False, "main.log missing; run make"))

    # Check the \documentclass options only. Searching the whole file gives a false
    # positive off the preamble comment that explains why the option is omitted.
    main = (paper / "main.tex").read_text(encoding="utf-8")
    decl = re.search(r"^\s*\\documentclass(\[[^\]]*\])?\{ceurart\}", main, re.M)
    options = (decl.group(1) or "") if decl else "MISSING"
    blind_ok = decl is not None and not {"singleblind", "doubleblind"} & set(
        options.strip("[]").split(",")
    )
    results.append(
        (blind_ok, f"single-blind: no anonymising class option (options: {options or 'none'})")
    )
    for phrase, where in (("no quantum speedup", "honest-scope statement"),):
        results.append((phrase in body.lower(), f"{where} present in the source"))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=Path("papers/aiqxqia2026"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results = run_checks(args.paper)
    failed = 0
    for ok, message in results:
        logger.info("%s  %s", "PASS" if ok else "FAIL", message)
        failed += not ok
    logger.info("%d checks, %d failed", len(results), failed)
    if failed:
        logger.info("See notes/submission-checklist.md for the author-supplied items.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
