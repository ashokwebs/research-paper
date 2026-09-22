"""Static check that the experiment drivers have no undefined names.

The experiment drivers are long-running and their setup blocks execute once, minutes
before anything else.  A name that went undefined during a refactor therefore surfaces
as a crash after the pool has spawned, not at import, and costs a whole run: this suite
gained the test after moving a shared computation into ``census_scale`` left
``run_structural_ladder`` referencing a ``real`` that no longer existed, which was
discovered only when a 25-minute job died in its first second.

The check is deliberately narrow.  It reports undefined names and nothing else -- not
style, not unused imports -- so it stays useful rather than becoming something to
silence.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["experiments", "src"]

# Superseded code from the retracted v4 manuscript, kept for traceability (see
# AUDIT.md) but not part of any current result.  It is excluded rather than fixed
# because editing withdrawn code to satisfy a linter would make the record of what was
# actually run less accurate, not more.
EXCLUDE = (
    "experiments/architecture/",
    "experiments/benchmarks/",
    "experiments/continual_learning/",
    "experiments/criticality/",
    "experiments/robotics/",
    "experiments/simulation/",
)

pyflakes = pytest.importorskip("pyflakes", reason="pyflakes not installed")


def _undefined_name_reports() -> list:
    files = sorted(
        str(p)
        for d in TARGETS
        for p in (ROOT / d).rglob("*.py")
        if not any(str(p.relative_to(ROOT)).startswith(x) for x in EXCLUDE)
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pyflakes", *files],
        capture_output=True, text=True, cwd=ROOT,
    )
    out = proc.stdout + proc.stderr
    return [ln for ln in out.splitlines() if "undefined name" in ln]


def test_no_undefined_names_in_experiments_or_src():
    bad = _undefined_name_reports()
    assert not bad, "undefined names found:\n" + "\n".join(bad)
