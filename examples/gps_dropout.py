"""GPS dropout: coast through a 20 s outage on the IMU (and barometer), and an IMU trade.

The receiver loses its fix from T+2 s to T+22 s: through burnout, max speed and
most of the coast. The filter must dead-reckon on the IMU, with only the
barometer (outside its transonic lockout) for altitude. When the fix comes
back, the filter must accept it and pull the estimate back in.

The same outage is then flown with three grades of IMU, to see what a better
(more expensive) IMU buys.

Run from the GNC folder (about 1 minute on 24 cores):
    python examples/gps_dropout.py
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from gnc.consistency import anees, anees_bounds
from gnc.ekf import DEG, NavConfig
from gnc.montecarlo import Case, run_many, stack, with_outage
from gnc.plots import INK_2, MUTED, SERIES, plt, shade_spans
from gnc.sensors import G0, ImuSpec, SensorSpec

OUT = ROOT / "outputs"
N_RUNS = 100
OUTAGE = (2.0, 22.0)

# Representative grades (1 sigma); consumer MEMS is the baseline in gnc/sensors.py
IMUS = {
    "Consumer MEMS (baseline)": ImuSpec(),
    "Industrial MEMS": ImuSpec(accel_noise=40e-6 * G0, accel_bias_sigma=2e-3 * G0, accel_bias_walk=5e-5,
                               accel_scale_sigma=0.001, gyro_noise=0.003 * DEG, gyro_bias_sigma=0.1 * DEG,
                               gyro_bias_walk=5e-6, gyro_scale_sigma=0.001),
    "Tactical grade": ImuSpec(accel_noise=20e-6 * G0, accel_bias_sigma=0.3e-3 * G0, accel_bias_walk=1e-5,
                              accel_scale_sigma=0.0003, gyro_noise=0.0017 * DEG, gyro_bias_sigma=1.0 / 3600 * DEG,
                              gyro_bias_walk=1e-6, gyro_scale_sigma=0.0001),
}


def case_for(imu: ImuSpec, outage=OUTAGE) -> Case:
    sensors = SensorSpec(imu=imu)
    case = Case(sensors=sensors, nav=NavConfig(sensors=sensors))
    return with_outage(case, *outage) if outage else case


def horizontal(results):
    E = stack(results, "err")
    S = stack(results, "sigma")
    return np.hypot(E[..., 0], E[..., 1]), np.sqrt(S[..., 0] ** 2 + S[..., 1] ** 2)


def plot_dropout(results, path):
    t = results[0].t
    m = t >= -2
    E = stack(results, "err")[:, m]
    S = stack(results, "sigma")[:, m].mean(axis=0)
    t = t[m]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, i, title in [(axes[0], 0, "East position error (m)"), (axes[1], 2, "Altitude error (m)")]:
        shade_spans(ax, [OUTAGE])
        for e in E[:, :, i]:
            ax.plot(t, e, color=MUTED, lw=0.4, alpha=0.35)
        ax.plot(t, 3 * S[:, i], color=SERIES[0], lw=1.6)
        ax.plot(t, -3 * S[:, i], color=SERIES[0], lw=1.6)
        ax.set_title(title)
        ax.set_xlabel("Time from ignition (s)")
        ax.annotate("GPS lost", (OUTAGE[0], 1.0), xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", fontsize=9, color=INK_2)
    axes[0].annotate("filter's 3σ", (OUTAGE[1], 3 * S[np.searchsorted(t, OUTAGE[1]) - 1, 0]), xytext=(-8, 4),
                     textcoords="offset points", ha="right", fontsize=9, color=SERIES[0])
    axes[0].annotate(f"grey: {len(results)} flights", (t[-1], -3 * S[-1, 0]), xytext=(-4, -14),
                     textcoords="offset points", ha="right", fontsize=9, color=MUTED)
    axes[1].annotate("the barometer holds altitude\n(except for 0.7 < Mach < 1.3)", (24, -7.6), fontsize=9,
                     color=INK_2)
    fig.suptitle("GPS dropout from T+2 s to T+22 s: consumer MEMS IMU, 6 m/s crosswind", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_imu_trade(grades, path):
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    shade_spans(ax, [OUTAGE])
    for k, (label, res) in enumerate(grades.items()):
        h, _ = horizontal(res)
        t = res[0].t
        m = t >= -2
        rms = np.sqrt(np.mean(h[:, m] ** 2, axis=0))
        ax.plot(t[m], rms, color=SERIES[k], label=label)
        i = np.searchsorted(t, OUTAGE[1]) - 1
        ax.annotate(f"{rms[i - np.argmax(m)]:.0f} m" if rms[i - np.argmax(m)] >= 10 else f"{rms[i - np.argmax(m)]:.1f} m",
                    (t[i], rms[i - np.argmax(m)]), xytext=(4, 0), textcoords="offset points", fontsize=9,
                    color=SERIES[k], va="center")
    ax.set_yscale("log")
    ax.set_xlabel("Time from ignition (s)")
    ax.set_ylabel("RMS horizontal error (m)")
    ax.set_title(f"What a better IMU buys through a 20 s GPS outage ({N_RUNS} flights each)")
    ax.annotate("GPS lost", (OUTAGE[0], 1.0), xycoords=("data", "axes fraction"), xytext=(4, -12),
                textcoords="offset points", fontsize=9, color=INK_2)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def stats(results, t_eval):
    t = results[0].t
    i = np.searchsorted(t, t_eval) - 1
    h, hs = horizontal(results)
    E = stack(results, "err")
    S = stack(results, "sigma")
    during = (t >= OUTAGE[0]) & (t < OUTAGE[1])
    A = anees(results, "position")[during]
    lo, hi = anees_bounds(3, len(results))
    return {
        "RMS horizontal error (m)": float(np.sqrt(np.mean(h[:, i] ** 2))),
        "filter's predicted horizontal 1-sigma (m)": float(hs[:, i].mean()),
        "RMS altitude error (m)": float(np.sqrt(np.mean(E[:, i, 2] ** 2))),
        "RMS horizontal velocity error (m/s)": float(np.sqrt(np.mean(E[:, i, 3] ** 2 + E[:, i, 4] ** 2))),
        "RMS tilt error at liftoff (deg)": float(np.sqrt(np.mean(np.sum(E[:, np.searchsorted(t, 0.0), 7:9] ** 2,
                                                                         axis=1))) / DEG),
        "position ANEES during outage": float(A.mean()),
        "position ANEES inside 95% band": float(np.mean((A >= lo) & (A <= hi))),
        "RMS horizontal error 3 s after the fix returns (m)": float(np.sqrt(np.mean(
            h[:, np.searchsorted(t, OUTAGE[1] + 3.0)] ** 2))),
        "max filter-reported sigma ratio": float((S[:, i, 0] / S[:, i, 0].mean()).max()),
    }


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    grades = {label: run_many(case_for(imu), N_RUNS) for label, imu in IMUS.items()}
    no_gps = run_many(case_for(ImuSpec(), outage=(0.0, 1e9)), N_RUNS)
    summary = {label: stats(res, OUTAGE[1]) for label, res in grades.items()}
    summary["Consumer MEMS, no GPS after liftoff (at apogee)"] = stats(no_gps, 41.5)
    for label, s in summary.items():
        print(label)
        for k, v in s.items():
            print(f"   {k:50s} {v:.3f}")
    (OUT / "gps_dropout_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_dropout(grades["Consumer MEMS (baseline)"], OUT / "gps_dropout.png")
    plot_imu_trade(grades, OUT / "imu_trade.png")
    print(f"saved to {OUT}")
