#!/usr/bin/env bash
# COMPUTATIONAL COUPLING --- build a PDF beside every Markdown document.
#
# Scope note. Two Markdown files in this repository are deliberately NOT converted:
#
#   paper/superseded_v4/pitch/EXECUTIVE_PITCH.md
#   paper/superseded_v4/computational_coupling_manuscript.md
#
# Both are retracted content whose every hardware number is fabricated (AUDIT.md §A1).
# Their original PDFs are already quarantined; rendering a fresh, clean-looking PDF of
# them would put another distributable copy of withdrawn claims into the world, which is
# the opposite of what the quarantine is for. The retraction *notices* in those
# directories are converted, so the directory still reads correctly as a PDF set.
#
# Vendored third-party Markdown (hardware/tools/, .venv/, upstream/) and tooling caches
# are out of scope.
set -euo pipefail

cd "$(dirname "$0")/../.."

command -v pandoc >/dev/null || { echo "pandoc not found; install it to build doc PDFs" >&2; exit 1; }

# info.md and the pitch quote figures by hand rather than through the macro file, so they
# can go stale against a re-run in a way the manuscript cannot. Check before rendering:
# a PDF of a stale number is worse than no PDF, because it looks finished.
if [ -x .venv/bin/python ]; then
  # Refresh the experiment inventory from the single source of truth before checking
  # anything, so the documents describe the experiments that exist rather than the ones
  # that existed when someone last edited the list by hand.
  .venv/bin/python src/analysis/sync_doc_inventory.py || {
    echo "could not sync the experiment inventory into the documents" >&2
    exit 1
  }
  echo
  .venv/bin/python src/analysis/check_doc_numbers.py || {
    echo "refusing to build doc PDFs from prose that disagrees with the results" >&2
    exit 1
  }
  echo
fi

SKIP=(
  "paper/superseded_v4/pitch/EXECUTIVE_PITCH.md"
  "paper/superseded_v4/computational_coupling_manuscript.md"
)

# pdflatex, with every non-ASCII character these documents use declared in
# doc_preamble.tex. xelatex is not installed here and lualatex cannot load its fonts
# ("metric data not found or bad"), so the engine that actually works is the one the
# manuscript already builds with. The tradeoff is that an undeclared character fails the
# build loudly, which is the behaviour we want: see the preamble's header.
ENGINE=pdflatex
command -v "$ENGINE" >/dev/null || { echo "$ENGINE not found" >&2; exit 1; }
PREAMBLE="$(pwd)/src/analysis/doc_preamble.tex"

built=0; skipped=0; failed=0
TMPSRC=()
cleanup() { [ ${#TMPSRC[@]} -gt 0 ] && rm -f "${TMPSRC[@]}" || true; }
trap cleanup EXIT
while IFS= read -r md; do
  rel="${md#./}"
  for s in "${SKIP[@]}"; do
    if [ "$rel" = "$s" ]; then
      echo "  skip (retracted content): $rel"
      skipped=$((skipped + 1))
      continue 2
    fi
  done

  out="${md%.md}.pdf"
  # --toc only where the document is long enough to need one.
  toc=()
  [ "$(grep -c '^## ' "$md")" -ge 5 ] && toc=(--toc --toc-depth=2)

  # Two source rewrites, both into a temp copy so the .md on disk is untouched:
  #
  #   1. Remote SVG shield badges are web decoration and need rsvg-convert, which is not
  #      installed. Badge-only lines are dropped rather than failing the document over
  #      them; every other image is left alone.
  #   2. The leading H1 becomes document metadata, so pandoc typesets a title block and
  #      then the contents. Left in the body it renders *after* the table of contents,
  #      which reads as though the ToC belongs to something else.
  src="$(mktemp --suffix=.md)"
  TMPSRC+=("$src")
  sed -E '/^\[!\[[^]]*\]\(https:\/\/img\.shields\.io[^)]*\)\]\([^)]*\)[[:space:]]*$/d' \
      "$md" > "$src"

  title="$(grep -m1 '^# ' "$src" | sed 's/^# //')"
  meta=()
  if [ -n "$title" ]; then
    sed -i '0,/^# /{/^# /d}' "$src"
    meta=(--metadata "title=${title}")
  fi

  if pandoc "$src" -o "$out" \
        --pdf-engine="$ENGINE" \
        --from=gfm+tex_math_dollars \
        --include-in-header="$PREAMBLE" \
        -V geometry:margin=1in \
        -V linkcolor:blue \
        -V fontsize=10pt \
        -V colorlinks=true \
        "${meta[@]}" "${toc[@]}" \
        --syntax-highlighting=tango \
        2> /tmp/pandoc_err.$$; then
    echo "  built: $out"
    built=$((built + 1))
  else
    echo "  FAILED: $rel"
    sed 's/^/      /' /tmp/pandoc_err.$$ | head -5
    failed=$((failed + 1))
  fi
  rm -f /tmp/pandoc_err.$$
done < <(find . -name "*.md" \
           -not -path "./.venv/*" \
           -not -path "./upstream/*" \
           -not -path "./hardware/tools/*" \
           -not -path "./.pytest_cache/*" \
           -not -path "./.git/*" | sort)

echo
echo "== doc build report =="
echo "  built:   ${built}"
echo "  skipped: ${skipped} (retracted content, see header)"
echo "  failed:  ${failed}"
[ "$failed" -eq 0 ]
