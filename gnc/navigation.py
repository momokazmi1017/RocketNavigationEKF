"""Run the navigation filter along a truth trajectory, and score it.

Timeline, as the flight computer sees it:
    pad (t < 0)   IMU prediction + GPS + barometer + zero-velocity and
                  zero-rate updates: the filter levels itself, learns the gyro
                  biases and calibrates the barometer against GPS altitude.
    liftoff       detected from the accelerometer; pad updates stop.
    flight        IMU prediction at 200 Hz, GPS at 10 Hz (when it has a fix),
                  barometer at 50 Hz (outside the transonic lockout).

The filter starts from the truth plus a random error drawn from its own
initial covariance, so across a Monte Carlo set the initial errors are
exactly as uncertain as the filter believes.

Scoring: the estimation error in error-state form,
    e = [p - p_hat, v - v_hat, Log(q_hat^-1 (x) q), b_a - b_a_hat, b_g - b_g_hat,
         s_a - s_a_hat, s_g - s_g_hat, b_gps - b_gps_hat, b_p - b_p_hat],
and the normalised estimation error squared, NEES = e^T P^-1 e. If the
filter's covariance is honest, NEES follows a chi-square distribution with as
many degrees of freedom as states in e (see consistency.py).
"""
from dataclasses import dataclass

import numpy as np

from .ekf import (BA, BG, BP, GP, N_ERR, P_, SA, SG, TH, V_, ErrorStateEKF, NavConfig, active_states,
                  initial_covariance)
from .atmosphere import G0
from .rotations import quat_conj, quat_exp, quat_log, quat_multiply
from .sensors import SensorData
from .truth import Truth

BLOCKS = {"position": P_, "velocity": V_, "attitude": TH}


@dataclass
class NavResult:
    t: np.ndarray            # (n,) s, the recorded samples
    k: np.ndarray            # (n,) truth grid index of each sample
    p: np.ndarray            # (n, 3) estimate
    v: np.ndarray
    q: np.ndarray            # (n, 4)
    b_a: np.ndarray          # (n, 3)
    b_g: np.ndarray
    b_p: np.ndarray          # (n,)
    err: np.ndarray          # (n, 25) truth minus estimate, error-state form
    sigma: np.ndarray        # (n, 25) sqrt(diag P)
    nees: dict               # name -> (n,) NEES of the full state and of each block
    t_liftoff: float         # detected, s
    stats: dict              # updates used / rejected by type
    baro_rejected_t: np.ndarray
    gps_valid_t: np.ndarray  # GPS epochs with a fix (for plots)


def error_state(truth: Truth, sens: SensorData, k: int, f: ErrorStateEKF) -> np.ndarray:
    kb = min(k, len(sens.b_a) - 1)
    e = np.empty(N_ERR)
    e[P_] = truth.r[k] - f.p
    e[V_] = truth.v[k] - f.v
    e[TH] = quat_log(quat_multiply(quat_conj(f.q), truth.q[k]))
    e[BA] = sens.b_a[kb] - f.b_a
    e[BG] = sens.b_g[kb] - f.b_g
    e[SA] = sens.s_a - f.s_a
    e[SG] = sens.s_g - f.s_g
    e[GP] = sens.b_gps[k] - f.b_gps
    e[BP] = sens.b_p - f.b_p
    return e


def run(truth: Truth, sens: SensorData, cfg: NavConfig = NavConfig(), rng: np.random.Generator | None = None,
        record_every: int = 4, t_end: float | None = None) -> NavResult:
    rng = rng or np.random.default_rng(0)
    dt = truth.dt
    P0 = initial_covariance(cfg)
    e0 = rng.multivariate_normal(np.zeros(N_ERR), P0)
    x0 = dict(p=truth.r[0] - e0[P_], v=truth.v[0] - e0[V_],
              q=quat_multiply(truth.q[0], quat_conj(quat_exp(e0[TH]))),
              b_a=sens.b_a[0] - e0[BA], b_g=sens.b_g[0] - e0[BG],
              s_a=sens.s_a - e0[SA] if cfg.scale_factor_states else np.zeros(3),
              s_g=sens.s_g - e0[SG] if cfg.scale_factor_states else np.zeros(3),
              b_gps=sens.b_gps[0] - e0[GP] if cfg.gps_error_states else np.zeros(3),
              b_p=sens.b_p - e0[BP])
    f = ErrorStateEKF(x0, P0, cfg, truth.site_altitude)

    gps_at = {int(k): i for i, k in enumerate(sens.gps_k)}
    baro_at = {int(k): i for i, k in enumerate(sens.baro_k)}
    n = len(truth.t) if t_end is None else int(np.searchsorted(truth.t, t_end)) + 1
    on_pad, above = True, 0
    t_liftoff = np.nan
    every = cfg.pad_update_every
    rec = {key: [] for key in ("k", "p", "v", "q", "b_a", "b_g", "b_p", "err", "sigma")}
    baro_rejected = []

    for k in range(n):
        # --- measurements at t_k -------------------------------------------------
        if on_pad and k >= every and k % every == 0:
            f.update_zupt()
            f.update_zaru(sens.imu_w[k - every:k].mean(axis=0), every * dt)
        if k in gps_at and sens.gps_valid[gps_at[k]]:
            i = gps_at[k]
            f.update_gps(sens.gps_pos[i], sens.gps_vel[i])
        if k in baro_at:
            if f.baro_locked_out():
                f.stats["baro"][1] += 1
                baro_rejected.append(truth.t[k])
            elif not f.update_baro(sens.baro_p[baro_at[k]])[0]:
                baro_rejected.append(truth.t[k])

        if k % record_every == 0 or k == n - 1:
            rec["k"].append(k)
            rec["p"].append(f.p.copy())
            rec["v"].append(f.v.copy())
            rec["q"].append(f.q.copy())
            rec["b_a"].append(f.b_a.copy())
            rec["b_g"].append(f.b_g.copy())
            rec["b_p"].append(f.b_p)
            rec["err"].append(error_state(truth, sens, k, f))
            rec["sigma"].append(np.sqrt(np.diag(f.P)))
            rec.setdefault("P", []).append(f.P.copy())
        if k == n - 1:
            break

        # --- liftoff detection, then propagate with the IMU over interval k ---------
        if on_pad:
            above = above + 1 if np.linalg.norm(sens.imu_f[k]) > cfg.launch_g * G0 else 0
            if above >= cfg.launch_samples:
                on_pad, t_liftoff = False, truth.t[k]
        f.predict(sens.imu_f[k], sens.imu_w[k], dt)

    kk = np.array(rec["k"])
    err = np.array(rec["err"])
    P = np.array(rec.pop("P"))
    nees = {}
    for name, s in {"full": active_states(cfg), **BLOCKS}.items():
        e = err[:, s]
        Ps = P[:, s][:, :, s]
        nees[name] = np.einsum("ni,ni->n", e, np.linalg.solve(Ps, e[..., None])[..., 0])
    return NavResult(t=truth.t[kk], k=kk, p=np.array(rec["p"]), v=np.array(rec["v"]), q=np.array(rec["q"]),
                     b_a=np.array(rec["b_a"]), b_g=np.array(rec["b_g"]), b_p=np.array(rec["b_p"]),
                     err=err, sigma=np.array(rec["sigma"]), nees=nees, t_liftoff=float(t_liftoff),
                     stats={k: tuple(v) for k, v in f.stats.items()},
                     baro_rejected_t=np.array(baro_rejected),
                     gps_valid_t=truth.t[sens.gps_k[sens.gps_valid]])
