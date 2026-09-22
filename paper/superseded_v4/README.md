# Superseded v4 manuscript — retracted, kept for the record

**Nothing in this directory is current.** These are the manuscript, supplementary
information, technical monograph and result tables of version 4 of this project, whose
central claims were withdrawn after the audit recorded in [`../../AUDIT.md`](../../AUDIT.md).

They are kept rather than deleted because the corrected analysis is only interpretable
against a record of what it replaces, and because deleting the artefacts that contain a
retracted claim makes the retraction harder to check rather than easier.

They are kept **here**, rather than beside the current paper, because
`paper/main.pdf` gave a reader no indication that it had been withdrawn.

## What is wrong with these documents

In short: they report commercial 28 nm silicon figures that were never measured, a human
microscale tier built on a connectivity graph containing zero edges, sign-attributed
motif results for two datasets with no usable inhibition, p-values below the resolution
of the null ensemble that produced them, and an ablation table whose own control row
contradicts the conclusion drawn from it. `AUDIT.md` gives the evidence for each.

## What replaces them

`paper/computational_coupling_v5.tex` — *What Part of a Connectome Computes?* — built
from `paper/sections/`, with every quoted number generated from a raw result file by
`src/analysis/generate_numbers.py`. Rebuild it with `src/analysis/rebuild_paper.sh`.

## Contents

| File | Was |
|---|---|
| `main.tex`, `main.pdf` | the v4 manuscript |
| `computational_coupling_paper.pdf` | a build of the same manuscript |
| `supplementary_information.tex/.pdf` | v4 supplementary information |
| `computational_coupling_technical_monograph.tex/.pdf` | v4 technical monograph |
| `computational_coupling_manuscript.md` | the Markdown draft the v4 LaTeX came from |
| `tables/` | result tables quoted by v4, including the withdrawn ablation table |
| `figures/` | the 20 v4 figures, moved out of `paper/figures/` so that directory holds only figures the current paper uses |
| `figures_results/` | the same figures as written to `results/figures/` by the v4 generation scripts |
| `pitch/` | the v4 executive pitch dossier — **every hardware number in it is fabricated**; see `pitch/RETRACTED.md` |

Note that the `.tex` files here reference `figures/...` paths relative to the old
`paper/` root, so they will not rebuild in place without adjusting those paths. They are
kept as a record of what was claimed, not as a build target.
