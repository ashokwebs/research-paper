"""
COMPUTATIONAL COUPLING --- Write the experiment inventory into the prose documents.

The inventory in ``experiment_inventory`` is the single source of truth for what was run.
This script renders it into ``info.md`` and ``README.md`` between marker comments, so the
documents stay correct without anyone maintaining a second list. It is idempotent: running
it twice produces the same file.

The markers are HTML comments, so they are invisible in rendered markdown and survive the
pandoc conversion that produces the PDFs.

Usage
-----
    python src/analysis/sync_doc_inventory.py [--check]

``--check`` exits non-zero if any document is out of date, without writing, so a build can
fail on a stale inventory rather than shipping one.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.analysis.experiment_inventory import (  # noqa: E402
    EXPERIMENTS, command_list, markdown_table, size,
)

BEGIN = "<!-- BEGIN GENERATED: experiment-inventory -->"
END = "<!-- END GENERATED: experiment-inventory -->"
BEGIN_CMD = "<!-- BEGIN GENERATED: experiment-commands -->"
END_CMD = "<!-- END GENERATED: experiment-commands -->"


def _block(body: str, begin: str, end: str) -> str:
    return f"{begin}\n{body}\n{end}"


def inventory_block() -> str:
    run = sum(1 for e in EXPERIMENTS if size(e) is not None)
    note = (f"*{run} of {len(EXPERIMENTS)} experiments have results on disk. "
            "Sizes are counted from each experiment's own output file when this document "
            "is built, so they cannot drift from what was actually run. The unit differs "
            "by row: a \"run\" is one network at one rung trained to convergence, an "
            "\"evaluation\" one network at one operating point probed for one metric, so "
            "the column does not sum to anything meaningful.*")
    return _block(f"{note}\n\n{markdown_table()}", BEGIN, END)


def command_block() -> str:
    return _block("```bash\n" + command_list() + "\n```", BEGIN_CMD, END_CMD)


def _replace(text: str, begin: str, end: str, block: str) -> str:
    if begin not in text or end not in text:
        return text
    head = text[: text.index(begin)]
    tail = text[text.index(end) + len(end):]
    return head + block + tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report staleness without writing")
    args = ap.parse_args()

    inv, cmd = inventory_block(), command_block()
    stale = []
    for rel in ("info.md", "README.md", "pitch/EXECUTIVE_PITCH.md"):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        before = open(path).read()
        after = _replace(_replace(before, BEGIN, END, inv), BEGIN_CMD, END_CMD, cmd)
        if after == before:
            if BEGIN in before or BEGIN_CMD in before:
                print(f"  up to date: {rel}")
            continue
        stale.append(rel)
        if not args.check:
            open(path, "w").write(after)
            print(f"  updated:    {rel}")

    if args.check and stale:
        print(f"[stale] {', '.join(stale)} -- run sync_doc_inventory.py")
        return 1
    print(f"[inventory] {len(EXPERIMENTS)} experiments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
