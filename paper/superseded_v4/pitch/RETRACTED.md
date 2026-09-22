# RETRACTED — do not send this to anyone

`EXECUTIVE_PITCH.md` and `computational_coupling_pitch.{tex,pdf}` are the v4 executive
pitch dossier. **Every hardware number in them is fabricated**, in the specific sense
that it was never measured by any tool:

- "TSMC 28nm HPC+", die area, "physical tapeout" — no Cadence tool, no TSMC library and
  no synthesis output ever existed in this repository. The figures came from a
  hand-written Python model with the area and energy constants written in as literals.
- "116.71 pJ/SynOp" — the same literal constant, not a power measurement.
- "2,857× real-time acceleration" — wall-clock of a NumPy loop divided by simulated
  biological time.
- "455× energy reduction" — a comparison between a near-silent network and a dense one
  firing freely, at different accuracies and unmatched activity levels.
- "26.5M synapses" for a human microscale tier — a sum over a column in a cell table.
  The corresponding connectivity graph had zero edges.

`AUDIT.md` in the repository root gives the evidence for each. This directory is kept so
that the retraction can be checked against what was actually claimed, and for no other
purpose. A pitch document is the artefact most likely to be forwarded, which is why it
carries its own warning rather than relying on the one a directory up.

**There is no replacement pitch.** The current manuscript is
`paper/computational_coupling_v5.tex`, and the hardware section of it reports gate count,
flip-flop count and mapped pre-layout cell area from an open Yosys/Nangate45 flow, and
nothing else — no die area, no power, no energy per operation, no clock frequency.
