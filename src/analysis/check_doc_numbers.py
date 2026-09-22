"""
COMPUTATIONAL COUPLING --- Cross-check the prose documents against the generated numbers.

`info.md` and `pitch/EXECUTIVE_PITCH.md` are written for people rather than compiled from
macros, so their figures are transcribed by hand. That is exactly the failure recorded in
AUDIT.md §D8: six protocol constants sat in the macro generator as literals, all correct
when written, and correct only until somebody changed a default.

The manuscript is immune to this because every number in it comes from
`paper/generated_numbers.tex`. The prose documents cannot be, so they are checked instead:
each entry below names a macro and the string the documents are expected to contain for
it. A mismatch prints both values and fails.

Run it after any re-run of the experiments, and before circulating either document:

    .venv/bin/python src/analysis/check_doc_numbers.py
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MACROS = os.path.join(ROOT, "paper", "generated_numbers.tex")
DOCS = ["info.md", os.path.join("pitch", "EXECUTIVE_PITCH.md")]

# macro name -> how the documents render it. A value of None means "render the macro's
# own value verbatim after stripping LaTeX escaping".
EXPECTED = {
    "NMotifTests": "29",
    "NMotifEnriched": "23",
    "NMotifDepleted": "3",
    "ZDrosCXCffl": "-10.9",
    "AnnealErrCeleVal": "0.3%",
    "OverlapCele": "13.9%",
    "OverlapLtwoCele": "11.6%",
    "NDegreeSig": "3",
    "DegreeEffectMedian": "-4.65",
    "DegreeEffectMax": "-8.02",
    "DegreeEffectMaxG": "-2.38",
    "NMotifSig": "2",
    "MotifEffectMedian": "-3.50",
    "MotifEffectMax": "-5.94",
    "MotifEffectMaxG": "-2.06",
    "ValenceEffectMedian": "-0.82",
    "ResidualEffectMedian": "1.06",
    "ZeroRecFull": "59.6%",
    "ZeroRecSplit": "4.9%",
    "ZeroRecSplitConn": "35.2%",
    "RateRatioLzeroLfour": "1.77",
    "RateAccCorr": "0.38",
    "RmLzeroMinusLfour": "13.95",
    "RmLzeroMinusLfourMin": "9.00",
    "MgLzeroMinusLfour": "11.74",
    "RmDegreeShareMin": "67%",
    "RmDegreeShareMax": "91%",
    "MDEHolm": "1.54",
    "PowerLevel": "80%",
    "NGammaOrdered": "3",
    "GammaProductSpread": "2.2",
    "RNinetyRand": "83%",
    "RNinetyHub": "73%",
    "RNinetyAdv": "64%",
    "NRandomStartConverged": "4",
    "InitMaxGap": "1.07",
    "SweepMedianSpread": "1.04",
    "MaxCVRatio": "1.00",
    "MinCVRatio": "0.79",
    "NMotifNulls": "500",
    "NinstancesLadder": "15",
    "NRateUnmatched": "27",
    "NRateCells": "250",
    "CellsCenAcceleratorTop": "590,116",
    "FfCenAcceleratorTop": "25,644",
    "AreaCenAcceleratorTop": "749,263",
    # The unit-model comparison, measured against the published null ensemble.
    "NUnitSpkLeads": "0",
    "NUnitSpkNetworks": "5",
    "UnitRateFracLeadsAllGains": "29%",
    "UnitDeadRate": "0.6%",
    "UnitDeadSpiking": "60.7%",
    "UnitSilenceGapR": "-0.58",
    "UnitCapRatio": "4.9",
    # Which rewiring counts as the null. These are the numbers the prose documents lead
    # with, so a change in any of them has to be reflected in all three of them.
    "NNullCells": "10",
    "NNullSignFlip": "6",
    "NNullVerdictFlip": "3",
    "NullDisagreeMedian": "0.73",
    "NullDisagreeMax": "6.98",
    "NBctRateLeads": "1",
    "NBctRateTrails": "3",
    "NBctRateNetworks": "5",
    "BctRateMedian": "-0.51",
    "NRankRateLeads": "5",
    "NRankRateNetworks": "5",
    "RankRateMedian": "+2.13",
    "NullShiftBct": "+0.06",
    "NullShiftRank": "+0.66",
    "NNullWeighted": "4",
    # The project's own scale. These had already drifted once -- the test count was two
    # revisions old and the audit tally predated four entries -- so they are checked.
    "NTests": "108",
    "NExperiments": "15",
    "NAuditVFour": "15",
    "NAuditReplacement": "12",
    "NAuditConclusion": "7",
    "NAuditTotal": "34",
}


def macro_values() -> dict:
    src = open(MACROS).read()
    out = {}
    for name, val in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", src, re.M):
        # Undo the LaTeX escaping the generator applies, so values compare as prose.
        out[name] = val.replace("\\%", "%").replace("\\,", "").strip()
    return out


def main() -> int:
    if not os.path.exists(MACROS):
        print(f"[skip] {MACROS} not found; run rebuild_paper.sh first")
        return 0

    vals = macro_values()
    text = "\n".join(open(os.path.join(ROOT, d)).read() for d in DOCS)
    # Prose uses typographic characters the macros do not: U+2212 for minus and U+2013
    # in ranges such as "67-91%". Normalise both sides so the comparison is between
    # numbers rather than between glyph choices.
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")

    stale, missing, ok = [], [], 0
    for name, expected in EXPECTED.items():
        actual = vals.get(name)
        if actual is None:
            missing.append(name)
            continue
        # The macro's current value must match what the documents were written against.
        # A leading "+" is a formatting choice of the generator, not part of the number,
        # so it is stripped from BOTH sides -- stripping only one made every signed
        # macro report as stale against a prose value identical to it.
        norm_actual = actual.replace("+", "").replace("\u2212", "-")
        norm_expected = expected.replace("+", "").replace("\u2212", "-")
        if norm_expected.rstrip("%") not in norm_actual:
            stale.append((name, expected, actual))
            continue
        # ...and the documents must actually contain it. Compare on the bare numeral,
        # because prose legitimately writes "67-91%" or "-4.65 pp" for a macro whose
        # value is "67%" or "-4.65%".
        if norm_expected.rstrip("%") not in text:
            stale.append((name, f"{expected} (not present in prose)", actual))
            continue
        ok += 1

    print(f"[doc-numbers] {ok}/{len(EXPECTED)} verified against {os.path.relpath(MACROS, ROOT)}")
    for name in missing:
        print(f"  MISSING MACRO  {name}")
    for name, expected, actual in stale:
        print(f"  STALE          {name}: prose says {expected!r}, macros say {actual!r}")

    if stale or missing:
        print("\nThe prose documents disagree with the generated numbers. Update "
              "info.md and pitch/EXECUTIVE_PITCH.md, or this file if a name changed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
