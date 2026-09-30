"""Truth trajectory: the rocket's real motion, from the 6-DOF simulator.

Loads data/truth_<case>.npz (written by tools/import_truth.py), puts the
rocket on the pad for `pad_time` seconds before ignition (the filter
initialises and calibrates there), and derives what a perfect IMU would read
over each 5 ms interval k -> k+1:

- Specific force (what an accelerometer feels: everything but gravity),
      f = a - g_vec,  averaged over the interval:
      f_avg = R_mid^T [ (v_k+1 - v_k) / dt + (0, 0, g) ],
  with R_mid the attitude at mid-interval. On the pad f = (0, 0, g): the pad
  pushes up at 1 g.
- Angular rate, averaged over the interval, from the attitude change:
      q_k+1 = q_k (x) Exp(w_avg dt)   ->   w_avg = Log(q_k^-1 (x) q_k+1) / dt.

These are the "delta-v / delta-theta" outputs a real strapdown IMU gives, so
dead-reckoning them reproduces the truth exactly (tests/test_ekf.py).
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .atmosphere import gravity
from .rotations import quat_conj, quat_exp, quat_log, quat_multiply, quat_to_dcm

DATA = Path(__file__).resolve().parents[1] / "data"


@dataclass
class Truth:
    t: np.ndarray            # (N,) s from ignition; negative on the pad
    r: np.ndarray            # (N, 3) position ENU from the pad, m
    v: np.ndarray            # (N, 3) velocity ENU, m/s
    q: np.ndarray            # (N, 4) attitude, body -> ENU
    p_static: np.ndarray     # (N,) Pa
    mach: np.ndarray         # (N,)
    phase: np.ndarray        # (N,) -1 pad, 0 rail, 1 powered, 2 coast
    f_b: np.ndarray          # (N-1, 3) mean specific force over interval k, body axes, m/s^2
    w_b: np.ndarray          # (N-1, 3) mean angular rate over interval k, body axes, rad/s
    site_altitude: float     # m MSL
    meta: dict

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])


def load(case: str = "wind", pad_time: float = 30.0) -> Truth:
    d = np.load(DATA / f"truth_{case}.npz")
    meta = json.loads(str(d["meta"]))
    dt = 1.0 / meta["rate_hz"]
    n_pad = int(round(pad_time / dt))

    def pad(a, fill):
        return np.concatenate([np.repeat(np.asarray(fill)[None], n_pad, axis=0), a])

    t = np.concatenate([(np.arange(n_pad) - n_pad) * dt, d["t"]])
    r = pad(d["r"], d["r"][0])
    v = pad(d["v"], d["v"][0])
    q = pad(d["q"], d["q"][0])
    p_static = np.concatenate([np.full(n_pad, d["p_static"][0]), d["p_static"]])
    mach = np.concatenate([np.zeros(n_pad), d["mach"]])
    phase = np.concatenate([np.full(n_pad, -1), d["phase"]])

    site = meta["site_altitude_m"]
    n = len(t)
    f_b = np.empty((n - 1, 3))
    w_b = np.empty((n - 1, 3))
    for k in range(n - 1):
        dq = quat_multiply(quat_conj(q[k]), q[k + 1])
        phi = quat_log(dq)
        w_b[k] = phi / dt
        R_mid = quat_to_dcm(quat_multiply(q[k], quat_exp(0.5 * phi)))
        g = gravity(site + 0.5 * (r[k, 2] + r[k + 1, 2]))
        f_b[k] = R_mid.T @ ((v[k + 1] - v[k]) / dt + np.array([0.0, 0.0, g]))
    return Truth(t, r, v, q, p_static, mach, phase, f_b, w_b, site, meta)
