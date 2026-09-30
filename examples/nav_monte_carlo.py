"""Is the filter honest? Monte Carlo consistency test and filter design trades.

1. Consistency: 200 runs of the crosswind flight, each with fresh sensor
   errors. The average NEES (ANEES) over the runs must stay inside its
   chi-square 95 % band (gnc/consistency.py).
2. Design trades: take one design decision out at a time and rerun
   100 flights, to show what each decision buys.
3. Transonic barometer: one flight flown with and without the Mach lockout
   and innovation gate.

Run from the GNC folder (about 2 minutes on 24 cores):
    python examples/nav_monte_carlo.py
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from gnc.consistency import anees, anees_bounds, summary
from gnc.ekf import NavConfig, active_states
from gnc.montecarlo import Case, _truth, run_many, run_one
from gnc.plots import INK_2, MUTED, SERIES, plt, shade_spans, spans_from_times
from gnc.sensors import port_error

OUT = ROOT / "outputs"
N_MAIN, N_TRADE = 200, 100

TRADES = [
    ("Baseline (all of the below)", NavConfig(), True),
    ("GPS error treated as white noise", NavConfig(gps_error_states=False), True),
    ("No IMU scale-factor states", NavConfig(scale_factor_states=False), True),
    ("No pad calibration (ZUPT / ZARU)", NavConfig(pad_update_every=10 ** 9), True),
    ("No barometer Mach lockout (gate only)", NavConfig(baro_mach_lockout=None), True),
    ("No lockout, no innovation gate", NavConfig(baro_mach_lockout=None, gate_prob=None), True),
]


def plot_anees(results, path):
    t = results[0].t
    M = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, constrained_layout=True)
    n_full = len(active_states(NavConfig()))
    for ax, (block, n, title) in zip(axes.flat, [("position", 3, "Position"), ("velocity", 3, "Velocity"),
                                                 ("attitude", 3, "Attitude"), ("full", n_full, f"All {n_full} states")]):
        lo, hi = anees_bounds(n, M)
        A = anees(results, block)
        ax.axhspan(lo, hi, color=SERIES[2], alpha=0.15, lw=0)
        ax.axhline(n, color=SERIES[2], lw=1.0)
        ax.plot(t, A, color=SERIES[0], lw=1.2)
        ax.set_title(f"{title} (expected {n})")
        ax.set_ylim(0, 2.2 * n)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.annotate("95 % band", (t[0], hi), xytext=(4, 3), textcoords="offset points", fontsize=8,
                    color=SERIES[2])
    axes[0, 0].set_ylabel("ANEES")
    axes[1, 0].set_ylabel("ANEES")
    axes[0, 1].annotate("pad: ZUPT assumes 1 cm/s of sway\n(the truth rocket is perfectly still),\nso deliberately conservative",
                        (-27, 1.3), fontsize=8, color=INK_2)
    for ax in axes[1]:
        ax.set_xlabel("Time from ignition (s)")
    fig.suptitle(f"Filter consistency: average NEES over {M} Monte Carlo flights", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_baro(case, path, seed=11):
    """Altitude error through Mach 1, with and without barometer protection."""
    runs = [("Mach lockout + innovation gate", case.nav),
            ("No protection", replace(case.nav, baro_mach_lockout=None, gate_prob=None))]
    truth = _truth(case.truth, case.pad_time)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    m = truth.t >= 0
    err = port_error(case.sensors.baro, truth.mach[m], truth.p_static[m])
    ax1.plot(truth.t[m], err, color=SERIES[1])
    ax1.set_title("Static-port pressure error near Mach 1 (illustrative model)")
    ax1.set_ylabel("Pa")
    up = truth.t[m][np.argmax(truth.mach[m] > 1)]
    down = truth.t[m][np.argmax((truth.mach[m] < 1) & (truth.t[m] > 8))]
    for tm, text in ((up, "Mach 1,\naccelerating"), (down, "Mach 1,\ndecelerating")):
        ax1.annotate(text, (tm, -1300), xytext=(10, 0), textcoords="offset points", fontsize=8, color=INK_2)
    res = {label: run_one(replace(case, nav=nav, record_every=2), seed) for label, nav in runs}
    good, bad = res[runs[0][0]], res[runs[1][0]]
    fl = good.t >= 0
    lock = spans_from_times(good.baro_rejected_t[good.baro_rejected_t > 0], 0.1)
    for ax in (ax1, ax2, ax3):
        shade_spans(ax, lock)
    ax2.plot(bad.t[fl], bad.err[fl, 2], color=SERIES[1], lw=1.4)
    ax2.set_title("No protection: altitude error (m)")
    ax2.annotate("the spike is absorbed into the barometer-offset (20 kPa wrong, filter claims 4 Pa)\n"
                 "and accelerometer scale-factor estimates; confident in wrong values, it never recovers",
                 (10.5, 400), fontsize=9, color=INK_2)
    ax3.fill_between(good.t[fl], -3 * good.sigma[fl, 2], 3 * good.sigma[fl, 2], color=SERIES[0], alpha=0.14, lw=0)
    ax3.plot(good.t[fl], good.err[fl, 2], color=SERIES[0], lw=1.4)
    ax3.set_title("Mach lockout + innovation gate: altitude error (m), with the filter's 3σ")
    ax3.annotate("barometer ignored\n(0.7 < Mach < 1.3)", (lock[1][0], 1.0), xycoords=("data", "axes fraction"),
                 xytext=(4, -26), textcoords="offset points", fontsize=8, color=MUTED)
    ax3.set_xlabel("Time from ignition (s)")
    fig.suptitle("Transonic barometer error: why the filter locks the barometer out near Mach 1", x=0.01,
                 ha="left", fontweight="bold", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fmt(v):
    return round(v, 3) if isinstance(v, float) else v


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    base = Case()
    t0 = time.time()
    main = run_many(base, N_MAIN)
    n_full = len(active_states(base.nav))
    s_main = summary(main, n_full)
    print(f"{N_MAIN} runs in {time.time() - t0:.0f} s")
    for k, v in s_main.items():
        print(f"  {k:45s} {fmt(v)}")
    pad = summary(main, n_full, t_min=-1e9)
    plot_anees(main, OUT / "anees.png")

    trades = []
    for label, nav, port in TRADES:
        res = main[:N_TRADE] if nav == base.nav else run_many(replace(base, nav=nav, transonic_port_error=port), N_TRADE)
        s = summary(res, len(active_states(nav)))
        s["label"] = label
        trades.append(s)
        print(f"{label:42s} ANEES pos/vel/att {s['ANEES position']:.2f} / {s['ANEES velocity']:.2f} / "
              f"{s['ANEES attitude']:.2f}   full/state {s['ANEES full per state']:.2f}   "
              f"RMS pos {s['RMS at apogee: position 3D (m)']:.2f} m   worst alt {s['worst altitude error, any run (m)']:.1f} m"
              f"   tilt {s['RMS at apogee: tilt (deg)']:.3f} deg  roll {s['RMS at apogee: roll (deg)']:.2f} deg")

    plot_baro(base, OUT / "baro_transonic.png")
    out = {"consistency, flight": {k: fmt(v) for k, v in s_main.items()},
           "ANEES bounds (3 states, 200 runs)": anees_bounds(3, N_MAIN),
           f"ANEES bounds ({n_full} states, 200 runs)": anees_bounds(n_full, N_MAIN),
           "trades (100 runs each)": [{k: fmt(v) for k, v in s.items()} for s in trades]}
    (OUT / "monte_carlo_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"saved to {OUT}")
