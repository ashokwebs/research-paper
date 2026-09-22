"""
COMPUTATIONAL COUPLING --- Reproducible logic synthesis of the CEN RTL.

Earlier revisions of this project reported silicon metrics (die area, pJ/SynOp,
process node) attributed to a commercial TSMC 28 nm flow run under Cadence Genus
and Innovus.  No such flow was ever run: the numbers came from constants embedded
in a Python model.  This script replaces them with a synthesis that anyone can
reproduce from this repository, using only open tooling:

* **Yosys** (via the ``yowasp-yosys`` wheel) for elaboration and technology mapping.
* **Nangate45 Open Cell Library** (typical corner) as the standard-cell target.

What this does and does not establish
-------------------------------------
Yosys + ABC performs logic synthesis and technology mapping.  It yields a real gate
count, a real flip-flop count and a real mapped cell area from the liberty file.  It
does **not** perform placement,
routing, clock-tree synthesis, parasitic extraction or signoff timing, so the area
reported here is pre-layout cell area, not die area, and no power number is claimed:
switching power depends on activity factors and parasitics this flow cannot see.  The
toolchain also ships no static timing analyser, so no critical path is reported either.
The paper reports exactly these quantities and no others.

Fixing the RTL to make this possible
------------------------------------
The Verilog in ``hardware/rtl/`` did not compile.  ``synaptic_accumulator`` and
``cen_accelerator_top`` declared unpacked arrays as module ports, which is not
synthesizable Verilog-2001 (despite the files' header claim) and is rejected by
standard frontends.  Both were rewritten to use flat packed buses.  These errors had
gone unnoticed because the RTL had never been passed through any synthesis or
simulation tool.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RTL_DIR = os.path.join(ROOT, "hardware", "rtl")
PDK_LIB = os.path.join(ROOT, "hardware", "pdk", "Nangate45_typ.lib")
RESULTS = os.path.join(ROOT, "results", "hardware")

# Each top module is synthesised separately so that per-block cost is visible.
TOP_MODULES = [
    ("dendritic_lif_core", ["dendritic_lif_core.v"]),
    ("synaptic_accumulator", ["synaptic_accumulator.v"]),
    ("aer_crossbar_router", ["aer_crossbar_router.v"]),
    ("cen_accelerator_top", ["cen_accelerator_top.v", "dendritic_lif_core.v",
                             "synaptic_accumulator.v", "aer_crossbar_router.v"]),
]


def _yosys_binary() -> tuple:
    """Locate a Yosys to run, preferring a native build.

    The ``yowasp-yosys`` WASM wheel elaborates and maps fine on its own, but its
    bundled ABC aborts while reading the Nangate45 liberty file -- including on a
    three-gate test design -- so technology mapping never completes and no area is
    produced.  A native Yosys (for example from the OSS CAD Suite, unpacked under
    ``hardware/tools/``) is therefore required for the numbers reported in the paper.
    The WASM wheel is retained as a fallback so the script still runs, and the result
    records which one was used.
    """
    local = os.path.join(ROOT, "hardware", "tools", "oss-cad-suite", "bin", "yosys")
    if os.path.exists(local):
        return local, "native (oss-cad-suite)"
    from shutil import which

    native = which("yosys")
    if native:
        return native, "native (system)"
    wasm = os.path.join(os.path.dirname(sys.executable), "yowasp-yosys")
    if os.path.exists(wasm):
        return wasm, "wasm (yowasp)"
    raise SystemExit(
        "No Yosys found. Install the OSS CAD Suite under hardware/tools/, "
        "put yosys on PATH, or `pip install yowasp-yosys` (WASM fallback; its ABC "
        "cannot map against this liberty file)."
    )


def run_yosys(script: str, workdir: str) -> str:
    """Execute a Yosys script and return its log.

    Paths are kept relative to ``workdir`` because the WASM fallback runs in a WASI
    sandbox rooted at the process working directory.
    """
    script_name = "synth.ys"
    with open(os.path.join(workdir, script_name), "w") as f:
        f.write(script)

    exe, _ = _yosys_binary()
    # The OSS CAD Suite binaries resolve their own libraries; injecting
    # LD_LIBRARY_PATH makes them pick up the wrong ones and segfault, so the
    # environment is passed through untouched.
    proc = subprocess.run(
        [exe, script_name],
        cwd=workdir, capture_output=True, text=True, timeout=3600,
    )
    if proc.returncode != 0:
        return (proc.stdout + proc.stderr
                + f"\n[driver] yosys exited with code {proc.returncode}\n")
    return proc.stdout + proc.stderr


def parse_stats(log: str) -> Dict[str, float]:
    """Pull mapped cell count, area and flip-flop count out of the Yosys stat report.

    No timing number is extracted: this toolchain ships no static timing analyser, so
    a critical path would have to be guessed rather than measured, and is therefore
    not reported.
    """
    out: Dict[str, float] = {}
    # Final stat block lines look like "  4401  5.3E+03 cells" and, per cell type,
    # "    70   372.4   DFFR_X1".
    m = re.findall(r"^\s*(\d+)\s+\S+\s+cells\s*$", log, flags=re.M)
    if m:
        out["cells"] = float(m[-1])
    m = re.findall(r"^\s*(\d+)\s+-\s+wires\s*$", log, flags=re.M)
    if m:
        out["wires"] = float(m[-1])
    m = re.findall(r"Chip area for (?:top )?module\s+'[^']*':\s+([0-9.]+)", log)
    if m:
        out["cell_area_um2"] = float(m[-1])
    m = re.findall(r"used for sequential elements:\s+([0-9.]+)", log)
    if m:
        out["sequential_area_um2"] = float(m[-1])

    ff = 0.0
    # Count only the cell-type lines of the final stat block.
    tail = log[log.rfind("Printing statistics"):] if "Printing statistics" in log else log
    for count, name in re.findall(r"^\s*(\d+)\s+\S+\s+([A-Z][A-Z0-9_]+)\s*$",
                                  tail, flags=re.M):
        if re.search(r"DFF|DLL|DLH|LAT", name, re.I):
            ff += float(count)
    out["flip_flops"] = ff
    return out


def prepare_workdir(lib: str) -> str:
    """Stage the RTL, the include header and the liberty file into a flat build dir."""
    import shutil

    workdir = os.path.join(RESULTS, "build")
    os.makedirs(workdir, exist_ok=True)
    for name in os.listdir(RTL_DIR):
        if name.endswith((".v", ".vh")):
            shutil.copy(os.path.join(RTL_DIR, name), os.path.join(workdir, name))
    shutil.copy(lib, os.path.join(workdir, "target.lib"))
    return workdir


def synthesise(top: str, files: List[str], workdir: str,
               keep_log: Optional[str] = None) -> Dict:
    src = " ".join(f'"{f}"' for f in files)
    # Standard Yosys ASIC flow: elaborate, run the generic `synth` script, map
    # sequential cells with dfflibmap and combinational logic with ABC against the
    # liberty file, then report mapped statistics.
    script = f"""
read_verilog {src}
hierarchy -check -top {top}
synth -top {top}
dfflibmap -liberty "target.lib"
abc -liberty "target.lib"
setundef -zero
clean
stat -liberty "target.lib"
"""
    log = run_yosys(script, workdir)
    stats = parse_stats(log)
    ok = "cell_area_um2" in stats
    if keep_log:
        with open(keep_log, "w") as f:
            f.write(log)
    return dict(top=top, files=files, ok=ok, **stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=PDK_LIB)
    args = ap.parse_args()

    if not os.path.exists(args.lib):
        sys.exit(f"Liberty file not found: {args.lib}")
    os.makedirs(RESULTS, exist_ok=True)

    workdir = prepare_workdir(args.lib)
    _, tool_kind = _yosys_binary()
    print(f"[synth] tool: {tool_kind}", flush=True)
    rows = []
    for top, files in TOP_MODULES:
        print(f"[synth] {top} ...", flush=True)
        log_path = os.path.join(RESULTS, f"yosys_{top}.log")
        r = synthesise(top, files, workdir, keep_log=log_path)
        rows.append(r)
        if r["ok"]:
            print(f"    cells={r.get('cells', 0):,.0f}  "
                  f"area={r.get('cell_area_um2', 0):,.1f} um^2  "
                  f"ff={r.get('flip_flops', 0):,.0f}  "
                  f"seq={r.get('sequential_area_um2', 0):,.1f} um^2")
        else:
            print(f"    FAILED (see {log_path})")

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "open_synthesis.csv"), index=False)
    with open(os.path.join(RESULTS, "open_synthesis.json"), "w") as f:
        json.dump(dict(library=os.path.basename(args.lib),
                       tool=f"yosys {tool_kind} + abc",
                       note="pre-layout mapped cell area; no place-and-route, "
                            "no parasitics, no power signoff, no static timing",
                       modules=rows), f, indent=2)
    print(f"\n[done] -> {RESULTS}/open_synthesis.csv")


if __name__ == "__main__":
    main()
