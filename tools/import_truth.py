"""Export truth trajectories from the 6-DOF flight simulator (project 2).

Flies the sized rocket with FlightSim6DOF and saves the launch-to-apogee
trajectory on a uniform 200 Hz grid to data/truth_<case>.npz: position,
velocity and attitude quaternion (ENU, body -> ENU), plus the static
pressure and Mach number the air-data sensors need.

The simulator's own output is sampled at 1 kHz and linearly interpolated onto
the grid (interpolation error below 1e-4 m in position). The IMU readings are
built later from these states (gnc/truth.py), so they are exactly consistent
with the trajectory.

Only the free-flight part is exported: under parachute the simulator is 3-DOF
and has no attitude.

Run from the GNC folder, with the FlightSim6DOF venv (it has the simulator's
dependencies) and the FlightSim6DOF folder next to this one:
    ..\\FlightSim6DOF\\.venv\\Scripts\\python.exe tools\\import_truth.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLIGHTSIM = ROOT.parent / "FlightSim6DOF"
sys.path.insert(0, str(FLIGHTSIM))

from flightsim.atmosphere import atmosphere  # noqa: E402
from flightsim.design import load_sized  # noqa: E402
from flightsim.sim import SimConfig, Wind, simulate  # noqa: E402

RATE = 200.0          # Hz, truth grid (the IMU rate)

CASES = {
    "calm": dict(wind=Wind()),
    "wind": dict(wind=Wind(6.0, 270.0)),     # 6 m/s from the west, as in project 2
}


def export(name: str, wind: Wind) -> dict:
    spec, vehicle, site = load_sized()
    cfg = SimConfig(site=site, wind=wind, dt_out=0.001)
    res = simulate(vehicle, cfg)
    m = res.phase <= 2                        # rail, powered, coast (6-DOF, has attitude)
    ts = res.t[m]

    t = np.arange(0.0, ts[-1], 1.0 / RATE)
    col = lambda a: np.interp(t, ts, a)       # noqa: E731
    r = np.column_stack([col(res.r[m, i]) for i in range(3)])
    v = np.column_stack([col(res.v[m, i]) for i in range(3)])
    q = np.column_stack([col(res.q[m, i]) for i in range(4)])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    p_static = np.array([atmosphere(site.altitude + z).p for z in r[:, 2]])
    mach = col(res.mach[m])
    phase = res.phase[m][np.searchsorted(ts, t, side="right") - 1]

    try:
        commit = subprocess.run(["git", "-C", str(FLIGHTSIM), "log", "-1", "--format=%h"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    meta = {
        "source": f"FlightSim6DOF (commit {commit}), sized vehicle, case '{name}'",
        "site_altitude_m": site.altitude,
        "rail_elevation_deg": site.rail_elevation_deg,
        "rail_azimuth_deg": cfg.rail_azimuth_deg,
        "wind_speed_10m": wind.speed_10m,
        "wind_from_deg": wind.from_deg,
        "rate_hz": RATE,
        "events": {k: float(v) for k, v in res.events.items()},
    }
    out = ROOT / "data" / f"truth_{name}.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, t=t, r=r, v=v, q=q, p_static=p_static, mach=mach, phase=phase,
                        meta=json.dumps(meta))
    speed = np.linalg.norm(v, axis=1)
    print(f"wrote {out.name}: {len(t)} samples, 0-{t[-1]:.2f} s, apogee {r[:, 2].max():.0f} m, "
          f"max speed {speed.max():.0f} m/s, max Mach {mach.max():.2f}")
    return meta


if __name__ == "__main__":
    for case, kw in CASES.items():
        export(case, **kw)
