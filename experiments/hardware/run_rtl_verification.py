"""
COMPUTATIONAL COUPLING --- Measured fixed-point error of the dendritic core RTL.

Earlier revisions reported "bit-exact fixed-point acceleration with sub-2 mV error
and zero spike divergence against FP64", derived from a Python model that *emulated*
the RTL rather than from the RTL itself; the Verilog had never been compiled, and in
fact did not compile. This script measures the quantity instead:

1. Generate a stimulus sequence spanning subthreshold drive, plateau-triggering
   drive and perisomatic inhibition.
2. Run the **actual Verilog** (Icarus Verilog) and capture per-step traces.
3. Run a double-precision reference implementing the same update equations.
4. Report the distribution of the deviation, and whether the spike trains agree.

Units are reported in Q8.8 least-significant bits and in the model's normalised
voltage units, where the somatic threshold is 1.0. A conversion to millivolts
requires assuming a physical voltage scale for that threshold, which the model does
not fix, so no millivolt figure is claimed.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "hardware")
WORK = os.path.join(RESULTS, "sim")

Q = 256.0  # Q8.8 scale: 1.0 == 256

# Parameters mirrored from hardware/rtl/cen_defs.vh
P = dict(
    v_thresh_soma=0x0100, theta_dend=0x0073, v_plateau=0x00D9,
    g_nmda=0x005A, g_axial=0x0040, alpha_soma=0x00F3,
    alpha_dend=0x00F9, alpha_plateau=0x00FE, shunt_eff=0x00F3,
)


def _tool(name: str) -> str:
    local = os.path.join(ROOT, "hardware", "tools", "oss-cad-suite", "bin", name)
    if os.path.exists(local):
        return local
    found = shutil.which(name)
    if found:
        return found
    raise SystemExit(f"{name} not found; install the OSS CAD Suite under hardware/tools/")


def make_stimulus(n_steps: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Stimulus covering subthreshold, plateau-triggering and shunted regimes."""
    rng = np.random.default_rng(seed)
    dend = np.zeros(n_steps, dtype=np.int64)
    inh = np.zeros(n_steps, dtype=np.int64)

    # Regime 1: sparse subthreshold excitatory events.
    seg = n_steps // 4
    ev = rng.random(seg) < 0.25
    dend[:seg] = np.where(ev, rng.integers(4, 24, seg), 0)

    # Regime 2: strong drive that should cross the dendritic NMDA threshold.
    ev = rng.random(seg) < 0.45
    dend[seg:2 * seg] = np.where(ev, rng.integers(20, 80, seg), 0)

    # Regime 3: strong drive plus perisomatic inhibition (shunting path).
    ev = rng.random(seg) < 0.45
    dend[2 * seg:3 * seg] = np.where(ev, rng.integers(20, 80, seg), 0)
    ev = rng.random(seg) < 0.35
    inh[2 * seg:3 * seg] = np.where(ev, rng.integers(20, 90, seg), 0)

    # Regime 4: mixed, including negative (hyperpolarising) dendritic drive.
    rest = n_steps - 3 * seg
    ev = rng.random(rest) < 0.4
    dend[3 * seg:] = np.where(ev, rng.integers(-40, 70, rest), 0)
    return dend, inh


def q88_mult(a: int, b: int) -> int:
    """Bit-accurate model of the RTL's multiplier: prod[23:8] of a 32-bit product."""
    prod = (a * b) & 0xFFFFFFFF
    sliced = (prod >> 8) & 0xFFFF
    return sliced - 0x10000 if sliced & 0x8000 else sliced


def reference_float(dend: np.ndarray, inh: np.ndarray) -> Dict[str, np.ndarray]:
    """Double-precision reference with the same update equations, no quantisation."""
    n = len(dend)
    v_d = v_s = 0.0
    nmda = False
    refr = 0
    out = dict(v_dend=np.zeros(n), v_soma=np.zeros(n),
               nmda=np.zeros(n, dtype=int), spike=np.zeros(n, dtype=int))

    th_d = P["theta_dend"] / Q
    v_pl = P["v_plateau"] / Q
    g_n = P["g_nmda"] / Q
    g_a = P["g_axial"] / Q
    a_s = P["alpha_soma"] / Q
    a_d = P["alpha_dend"] / Q
    a_p = P["alpha_plateau"] / Q
    v_th = P["v_thresh_soma"] / Q
    shunt_keep = 1.0 - P["shunt_eff"] / Q

    for t in range(n):
        i_d = dend[t] / Q
        i_i = inh[t] / Q
        # Registered outputs lag the state update by one step, matching the RTL.
        out["v_dend"][t] = v_d
        out["v_soma"][t] = v_s
        out["nmda"][t] = int(nmda)

        v_diff = v_d - v_s
        i_ax = g_a * v_diff
        nmda_drive = g_n * (v_pl - v_d) if nmda else 0.0
        alpha_d = a_p if nmda else a_d
        v_d_dec = alpha_d * v_d
        v_s_dec = a_s * v_s
        shunt = i_i > (0x0010 / Q)

        if v_d >= th_d:
            nmda_next = True
        elif v_d < (th_d - 0x0020 / Q):
            nmda_next = False
        else:
            nmda_next = nmda

        spike = 0
        if refr > 0:
            refr -= 1
            v_s_next = 0.0
        elif shunt:
            v_s_next = v_s_dec * shunt_keep
        elif (v_s_dec + i_ax) >= v_th:
            spike = 1
            v_s_next = 0.0
            refr = 2
        else:
            v_s_next = v_s_dec + i_ax

        out["spike"][t] = spike
        v_d = v_d_dec + i_d + nmda_drive
        v_s = v_s_next
        nmda = nmda_next
    return out


def run_rtl(dend: np.ndarray, inh: np.ndarray, n_steps: int) -> np.ndarray:
    os.makedirs(WORK, exist_ok=True)
    with open(os.path.join(WORK, "stimulus.txt"), "w") as f:
        for a, b in zip(dend, inh):
            f.write(f"{int(a)} {int(b)}\n")

    vvp = os.path.join(WORK, "tb_trace.vvp")
    cmd = [
        _tool("iverilog"), "-g2005", "-o", vvp,
        "-DNSTEPS=%d" % n_steps,
        "-P", f"tb_dendritic_core_trace.NSTEPS={n_steps}",
        "-I", os.path.join(ROOT, "hardware", "rtl"),
        os.path.join(ROOT, "hardware", "sim", "tb_dendritic_core_trace.v"),
        os.path.join(ROOT, "hardware", "rtl", "dendritic_lif_core.v"),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("iverilog failed:\n" + r.stdout + r.stderr)

    r = subprocess.run([_tool("vvp"), vvp], cwd=WORK, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("vvp failed:\n" + r.stdout + r.stderr)
    print("  " + r.stdout.strip())

    import pandas as pd
    return pd.read_csv(os.path.join(WORK, "rtl_trace.csv"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    dend, inh = make_stimulus(args.steps, args.seed)

    print(f"[rtl] simulating {args.steps} steps of dendritic_lif_core ...", flush=True)
    trace = run_rtl(dend, inh, args.steps)
    ref = reference_float(dend, inh)

    v_d_rtl = trace["v_dend"].to_numpy() / Q
    v_s_rtl = trace["v_soma"].to_numpy() / Q
    spike_rtl = trace["spike"].to_numpy().astype(int)
    nmda_rtl = trace["nmda"].to_numpy().astype(int)

    err_d = v_d_rtl - ref["v_dend"]
    err_s = v_s_rtl - ref["v_soma"]

    spikes_rtl = int(spike_rtl.sum())
    spikes_ref = int(ref["spike"].sum())
    agree = int((spike_rtl == ref["spike"]).sum())

    out = dict(
        n_steps=args.steps,
        tool="icarus verilog (oss-cad-suite)",
        note=("deviation of the actual synthesizable RTL from a double-precision "
              "reference of the same update equations; units are normalised so that "
              "the somatic threshold is 1.0, and 1 LSB = 1/256"),
        dendritic=dict(
            max_abs_err_lsb=float(np.abs(err_d).max() * Q),
            rms_err_lsb=float(np.sqrt((err_d ** 2).mean()) * Q),
            max_abs_err_norm=float(np.abs(err_d).max()),
            corr=float(np.corrcoef(v_d_rtl, ref["v_dend"])[0, 1]),
        ),
        somatic=dict(
            max_abs_err_lsb=float(np.abs(err_s).max() * Q),
            rms_err_lsb=float(np.sqrt((err_s ** 2).mean()) * Q),
            max_abs_err_norm=float(np.abs(err_s).max()),
            corr=float(np.corrcoef(v_s_rtl, ref["v_soma"])[0, 1])
            if v_s_rtl.std() > 0 and ref["v_soma"].std() > 0 else float("nan"),
        ),
        spikes=dict(rtl=spikes_rtl, reference=spikes_ref,
                    step_agreement=agree / args.steps),
        nmda=dict(rtl_active_steps=int(nmda_rtl.sum()),
                  reference_active_steps=int(ref["nmda"].sum())),
    )

    with open(os.path.join(RESULTS, "rtl_verification.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"  dendritic: max |err| = {out['dendritic']['max_abs_err_lsb']:.2f} LSB "
          f"({out['dendritic']['max_abs_err_norm']:.4f} normalised), "
          f"r = {out['dendritic']['corr']:.5f}")
    print(f"  somatic:   max |err| = {out['somatic']['max_abs_err_lsb']:.2f} LSB, "
          f"r = {out['somatic']['corr']:.5f}")
    print(f"  spikes: RTL {spikes_rtl}, reference {spikes_ref}, "
          f"per-step agreement {100*agree/args.steps:.2f}%")
    print(f"\n[done] -> {RESULTS}/rtl_verification.json")


if __name__ == "__main__":
    main()
