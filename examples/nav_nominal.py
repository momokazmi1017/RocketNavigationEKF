"""Navigate one flight: the 6 m/s crosswind mission from project 2, launch to apogee.

Plots the estimation errors with the filter's own 3-sigma bounds, and how the
attitude uncertainty evolves (what the sensors can and cannot observe), for
the calm and the crosswind flights.

Run from the GNC folder:
    python examples/nav_nominal.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from gnc.ekf import BA, BP, P_, SA, TH, V_, NavConfig
from gnc.montecarlo import Case, _truth
from gnc.navigation import run
from gnc.plots import INK_2, MUTED, SERIES, plt, shade_spans, spans_from_times
from gnc.sensors import generate

OUT = ROOT / "outputs"
SEED = 7
DEG = 180 / np.pi


def gps_outage_spans(res, truth):
    """Time spans in flight when the receiver had no fix."""
    epochs = np.round(truth.t[(truth.t >= 0) & (np.abs(truth.t * 10 - np.round(truth.t * 10)) < 1e-6)], 3)
    lost = epochs[~np.isin(epochs, np.round(res.gps_valid_t, 3))]
    return spans_from_times(lost, 0.15)


def plot_errors(res, truth, path, t0=-3.0):
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.5), sharex=True, constrained_layout=True)
    m = res.t >= t0
    t = res.t[m]
    rows = [("Position error (m)", P_, 1.0, ["East", "North", "Up"]),
            ("Velocity error (m/s)", V_, 1.0, ["East", "North", "Up"]),
            ("Attitude error (deg)", TH, DEG, ["Roll", "Pitch", "Yaw"])]
    gps_out = gps_outage_spans(res, truth)
    for r, (title, s, scale, names) in enumerate(rows):
        for c in range(3):
            ax = axes[r, c]
            i = s.start + c
            sig = 3 * res.sigma[m, i] * scale
            shade_spans(ax, gps_out)
            ax.fill_between(t, -sig, sig, color=SERIES[0], alpha=0.14, lw=0)
            ax.plot(t, sig, color=SERIES[0], lw=1.0)
            ax.plot(t, -sig, color=SERIES[0], lw=1.0)
            ax.plot(t, res.err[m, i] * scale, color=SERIES[1], lw=1.3)
            ax.set_title(names[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(title)
            ax.axvline(0, color=MUTED, lw=0.8)
    axes[0, 0].annotate("filter's 3σ", (t[-1], 3 * res.sigma[m, 0][-1]), xytext=(-4, 6),
                        textcoords="offset points", ha="right", fontsize=9, color=SERIES[0])
    j = len(t) // 3
    axes[0, 0].annotate("actual error", (t[j], res.err[m, 0][j]), xytext=(0, -18),
                        textcoords="offset points", fontsize=9, color=SERIES[1])
    a, b = gps_out[0]
    axes[1, 0].annotate("no GPS fix\n(over 1,000 kn)", (b, 0.0), xycoords=("data", "axes fraction"),
                        xytext=(3, 6), textcoords="offset points", fontsize=8, color=MUTED)
    for ax in axes[2]:
        ax.set_xlabel("Time from ignition (s)")
    fig.suptitle("Navigation error, 6 m/s crosswind flight (one run)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_attitude_sigma(runs, path):
    """runs: [(label, NavResult)], calm first, crosswind second."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, constrained_layout=True)
    wind = runs[1][1]
    for c, name in [(1, "Pitch"), (2, "Yaw")]:
        ax1.plot(wind.t, wind.sigma[:, TH.start + c] * DEG, color=SERIES[c - 1], label=name,
                 ls="-" if c == 1 else "--")
    ax1.set_title("Pitch and yaw (tilt of the rocket's axis)")
    ax1.annotate("burn: a tilt error points the thrust\nthe wrong way, and GPS sees it", (8, 0.0075),
                 fontsize=9, color=INK_2)
    ax1.annotate("coast: nothing to feel but drag,\nso the tilt drifts on the gyros", (12, 0.3),
                 fontsize=9, color=INK_2)
    ax1.legend(loc="lower left")
    for k, (label, res) in enumerate(runs):
        ax2.plot(res.t, res.sigma[:, TH.start] * DEG, color=SERIES[k], label=label)
    ax2.set_title("Roll (about the rocket's axis)")
    ax2.annotate("Thrust is along the roll axis, so the accelerometers\ncannot see roll. Once the rocket turns, a roll\n"
                 "error sends the turn the wrong way, and GPS sees\nthat. The crosswind turn (weathercocking) is\n"
                 "sharper and earlier than the calm gravity turn.", (8, 0.012), fontsize=9, color=INK_2)
    ax2.legend(loc="lower left")
    for ax in (ax1, ax2):
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.annotate("liftoff", (0, 1.0), xycoords=("data", "axes fraction"), xytext=(-4, -12),
                    textcoords="offset points", ha="right", fontsize=9, color=INK_2)
        ax.set_yscale("log")
        ax.set_ylim(0.005, 3)
        ax.set_xlabel("Time from ignition (s)")
    ax1.set_ylabel("Filter's 1σ (deg)")
    fig.suptitle("Attitude uncertainty: what the sensors can and cannot see", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def at(res, t):
    return min(int(np.searchsorted(res.t, t)), len(res.t) - 1)


def summarise(res, truth):
    ev = truth.meta["events"]
    out = {"liftoff detected (s)": res.t_liftoff, "updates used / rejected": res.stats}
    for label, t in [("on the pad (T-1 s)", -1.0), ("burnout", ev["burnout"]), ("apogee", res.t[-1])]:
        i = at(res, t)
        out[label] = {
            "position error (m)": np.round(res.err[i, P_], 2).tolist(),
            "position 1-sigma (m)": np.round(res.sigma[i, P_], 2).tolist(),
            "velocity error (m/s)": np.round(res.err[i, V_], 3).tolist(),
            "velocity 1-sigma (m/s)": np.round(res.sigma[i, V_], 3).tolist(),
            "attitude error roll/pitch/yaw (deg)": np.round(res.err[i, TH] * DEG, 3).tolist(),
            "attitude 1-sigma (deg)": np.round(res.sigma[i, TH] * DEG, 3).tolist(),
            "accel bias 1-sigma (mg)": np.round(res.sigma[i, BA] / 9.80665 * 1e3, 2).tolist(),
            "accel scale factor 1-sigma (ppm)": np.round(res.sigma[i, SA] * 1e6, 0).tolist(),
            "baro offset error / 1-sigma (Pa)": [round(float(res.err[i, BP]), 1), round(float(res.sigma[i, BP]), 1)],
        }
    fl = res.t > 0
    out["flight max |error|"] = {
        "position (m)": float(np.abs(res.err[fl, P_]).max()),
        "velocity (m/s)": float(np.abs(res.err[fl, V_]).max()),
        "pitch/yaw (deg)": float(np.abs(res.err[fl, 7:9]).max() * DEG),
        "roll (deg)": float(np.abs(res.err[fl, 6]).max() * DEG),
    }
    return out


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    runs, data = [], {}
    for name, label in [("calm", "Calm"), ("wind", "6 m/s crosswind")]:
        case = Case(truth=name)
        truth = _truth(case.truth, case.pad_time)
        rng = np.random.default_rng(SEED)
        sens = generate(truth, case.sensors, rng, case.transonic_port_error)
        runs.append((label, run(truth, sens, NavConfig(), rng, record_every=2)))
        data[name] = (truth, sens)
    truth, sens = data["wind"]
    res = runs[1][1]                     # the crosswind flight is the headline case
    s = summarise(res, truth)
    s["true sensor errors"] = {
        "accel turn-on bias (mg)": np.round(sens.b_a[0] / 9.80665 * 1e3, 2).tolist(),
        "gyro turn-on bias (deg/s)": np.round(sens.b_g[0] * DEG, 3).tolist(),
        "accel scale factor (ppm)": np.round(sens.s_a * 1e6, 0).tolist(),
        "baro offset (Pa)": round(sens.b_p, 1),
    }
    s["calm flight, apogee"] = summarise(runs[0][1], data["calm"][0])["apogee"]
    (OUT / "nav_nominal_summary.json").write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(s, indent=2))
    plot_errors(res, truth, OUT / "nav_errors.png")
    plot_attitude_sigma(runs, OUT / "attitude_sigma.png")
    print(f"saved plots to {OUT}")
