# Superseded v4 result files

Outputs of the v4 experiments, kept so the retracted claims can be checked against the
data that produced them. Nothing here is read by the current analysis pipeline, which
uses `results/ladder/`, `results/motifs/`, `results/degeneracy/` and `results/hardware/`.

Two files are worth naming, because `AUDIT.md` cites them directly:

- `cen_ablation_study.csv` — the ablation table of AUDIT §B1, whose Erdős–Rényi control
  row is *better* than the motif-structured model on the very metric used to argue for
  motifs, and whose reported `p < 0.05` is unattainable at n = 3 with the stated
  standard deviations.
- `motif_significance_cross_species.csv` — the enrichment table of AUDIT §C1 and §C2. Its
  p-values are reported as `p < 0.0001` from K = 25 surrogates, where the smallest
  attainable empirical p-value is 1/(K+1) ≈ 0.038, and its *C. elegans* FFL row does not
  match the corresponding row of the v4 manuscript.

The replacements are `results/motifs/motif_significance.csv` (K = 500, exact counts,
resolution floor stated) and the control ladder in `results/ladder/`.
