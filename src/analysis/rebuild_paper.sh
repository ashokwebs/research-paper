#!/usr/bin/env bash
# COMPUTATIONAL COUPLING --- regenerate every derived artefact from results/.
#
# This is the whole path from raw result files to the compiled PDF.  It exists so that
# the sequence is executable rather than described: statistics, then figures, then
# tables, then the macro file the manuscript reads, then three LaTeX passes.  Nothing
# here re-runs an experiment; if a result file is missing, the corresponding number
# appears in the PDF as a visible "??" rather than as a stale value.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=.venv/bin/python
PAPER=paper/computational_coupling_v5

echo "== statistics =="
$PY experiments/ladder/analyse_ladder.py --metric accuracy
# The dynamical probes get the same per-property treatment as the task metric, and the
# manuscript quotes their effect sizes, so their contrast files have to be regenerated
# here rather than left to whoever remembered to run them by hand.
if [ -f results/ladder/dynamics_ladder.csv ]; then
  for metric in memory_capacity kernel_rank; do
    $PY experiments/ladder/analyse_ladder.py \
        --raw results/ladder/dynamics_ladder.csv --metric "$metric" > /dev/null
  done
fi

echo "== figures =="
$PY src/analysis/make_all_figures.py

echo "== tables =="
$PY src/analysis/generate_tables.py

echo "== numbers =="
$PY src/analysis/generate_numbers.py | tee /tmp/cc_numbers.$$
UNDERPOWERED=$(grep -c "networks," /tmp/cc_numbers.$$ || true)
rm -f /tmp/cc_numbers.$$

echo "== latex =="
cd paper
pdflatex -interaction=nonstopmode computational_coupling_v5.tex > /dev/null
bibtex computational_coupling_v5 > /dev/null 2>&1 || true
pdflatex -interaction=nonstopmode computational_coupling_v5.tex > /dev/null
pdflatex -interaction=nonstopmode computational_coupling_v5.tex > /dev/null
cd ..

echo
echo "== build report =="
# A silently-undefined reference or a leftover placeholder is the failure mode this
# whole pipeline exists to prevent, so both are reported rather than buried in the log.
missing=$(grep -c 'textbf{??}' paper/generated_numbers.tex || true)
undef=$(grep -cE 'Reference .* undefined|Citation .* undefined' "${PAPER}.log" || true)
# pdflatex in nonstopmode reports a hard error and then produces a PDF anyway, so the
# exit status and the existence of the file say nothing.  A duplicate \newcommand, for
# instance, builds fine and silently keeps whichever definition came first.
errors=$(grep -c '^! ' "${PAPER}.log" || true)
# Boxes more than a few points over run off the column and are a layout defect, not a
# warning: a table or equation silently extends past the text block and the PDF still
# builds. Small ones are normal and not worth reporting.
badbox=$(grep -oE 'Overfull \\hbox \(([0-9.]+)pt' "${PAPER}.log" \
         | grep -oE '[0-9.]+' | awk '$1 > 10' | wc -l || true)
pages=$(pdfinfo "${PAPER}.pdf" 2>/dev/null | awk '/^Pages/{print $2}')
echo "  pages:                 ${pages:-unknown}"
echo "  latex errors:          ${errors}"
echo "  underpowered inputs:   ${UNDERPOWERED:-0}"
echo "  overfull boxes >10pt:  ${badbox}"
echo "  unresolved numbers:    ${missing}"
echo "  undefined refs/cites:  ${undef}"
if [ "${errors}" != "0" ]; then
  grep -A1 '^! ' "${PAPER}.log" | head -20
fi
if [ "${undef}" != "0" ]; then
  grep -E 'Reference .* undefined|Citation .* undefined' "${PAPER}.log" | sort -u | head -20
fi
if [ "${missing}" != "0" ]; then
  grep 'textbf{??}' paper/generated_numbers.tex | sed 's/.*newcommand{\\\([A-Za-z]*\)}.*/    \1/;s/.*providecommand{\\\([A-Za-z]*\)}.*/    \1/' | head -30
fi
echo "  -> ${PAPER}.pdf"
