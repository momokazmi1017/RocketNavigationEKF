"""Error-state extended Kalman filter (ES-EKF) for IMU + GPS + barometer.

Two layers, as in most flight navigation software:

1. The nominal state is integrated from the IMU at 200 Hz (strapdown
   mechanization), exactly like dead reckoning:
       f = (f_meas - b_a) / (1 + s_a),  w = (w_meas - b_g) / (1 + s_g)   (corrected IMU)
       q <- q (x) Exp(w dt)                                (attitude)
       v <- v + (R_mid f + g_vec) dt                       (Newton's 2nd law)
       p <- p + (v_old + v_new) / 2 dt
   Nominal state: p, v (ENU, m and m/s), q (body -> ENU), the IMU biases b_a,
   b_g and scale-factor errors s_a, s_g, the GPS's slowly wandering position
   error b_gps (m) and the barometer offset b_p (Pa).

2. The filter tracks the error of that nominal state, a 25-vector
       dx = [dp, dv, dtheta, db_a, db_g, ds_a, ds_g, db_gps, db_p],
   with covariance P. Errors are small, so their dynamics are linear
   (Sola 2017, "Quaternion kinematics for the error-state Kalman filter"):
       dp'     = dv
       dv'     = -R [f]x dtheta - R D_a (db_a + f * ds_a) + (2 g / (Re + z)) dp_z e_z - R n_a
       dtheta' = -[w]x dtheta - D_g (db_g + w * ds_g) - n_g
       db_a' = n_ba,  db_g' = n_bg,  db_p' = n_bp,  ds_a' = ds_g' = 0
       db_gps' = -db_gps / tau + n_gps               (Gauss-Markov)
   with D = diag(1 / (1 + s)) and * the per-axis product. The dv' row is the
   physics of why attitude matters: an attitude error dtheta tips the measured
   thrust f sideways, so the velocity error grows as |f| * dtheta. It is also
   why GPS can observe attitude while the engine burns.

A measurement z = h(x) + noise updates the error (Kalman gain K = P H^T S^-1),
the error is added into the nominal state, and it is reset to zero. The
covariance update uses the Joseph form, which stays symmetric and positive.
Every update is checked first with an innovation (chi-square) gate: a reading
more than a few sigma from the prediction is rejected, not trusted.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .atmosphere import R_EARTH, air, gravity
from .rotations import quat_exp, quat_multiply, quat_to_dcm, skew
from .sensors import SensorSpec

# Error-state layout
P_, V_, TH = slice(0, 3), slice(3, 6), slice(6, 9)
BA, BG, SA, SG = slice(9, 12), slice(12, 15), slice(15, 18), slice(18, 21)
GP, BP = slice(21, 24), 24
N_ERR = 25
STATE_NAMES = ["p_E", "p_N", "p_U", "v_E", "v_N", "v_U", "roll", "pitch", "yaw",
               "b_ax", "b_ay", "b_az", "b_gx", "b_gy", "b_gz", "s_ax", "s_ay", "s_az",
               "s_gx", "s_gy", "s_gz", "gps_E", "gps_N", "gps_U", "b_baro"]
DEG = np.pi / 180.0


@dataclass(frozen=True)
class NavConfig:
    sensors: SensorSpec = SensorSpec()        # the filter's model of its sensors
    gps_error_states: bool = True             # False: treat all GPS position error as white noise
    scale_factor_states: bool = True          # False: ignore IMU scale-factor error
    baro_bias_walk: float = 0.05              # Pa/sqrt(s), lets the offset drift slowly
    # Initial uncertainty (1 sigma). Attitude comes from the surveyed launch
    # rail: its elevation and azimuth tilt body x (pitch/yaw); how the rocket
    # sits on the rail buttons sets roll.
    init_pos: float = 5.0                     # m
    init_vel: float = 0.1                     # m/s
    init_roll: float = 2.0 * DEG              # rad
    init_tilt: float = 0.5 * DEG              # rad
    # Pad calibration: zero-velocity (ZUPT) and zero-rate (ZARU) updates while
    # the rocket sits on the pad. Liftoff is detected when the accelerometer
    # reads more than launch_g for launch_samples samples in a row.
    zupt_sigma: float = 0.01                  # m/s
    pad_update_every: int = 20                # samples (10 Hz)
    launch_g: float = 1.2
    launch_samples: int = 3
    # Measurement protection
    gate_prob: float | None = 0.9999          # chi-square innovation gate; None = accept everything
    baro_mach_lockout: tuple | None = (0.7, 1.3)   # ignore the barometer in this Mach band


def active_states(cfg: NavConfig) -> np.ndarray:
    """Indices of the error states the filter estimates."""
    off = []
    if not cfg.gps_error_states:
        off += list(range(N_ERR))[GP]
    if not cfg.scale_factor_states:
        off += list(range(N_ERR))[SA] + list(range(N_ERR))[SG]
    return np.array([i for i in range(N_ERR) if i not in off])


class ErrorStateEKF:
    def __init__(self, x0: dict, P0, cfg: NavConfig, site_altitude: float):
        """x0: initial nominal state, keys p, v, q, b_a, b_g, s_a, s_g, b_gps, b_p."""
        self.p, self.v, self.q = (np.array(x0[k], float) for k in ("p", "v", "q"))
        self.b_a, self.b_g, self.s_a, self.s_g, self.b_gps = (
            np.array(x0[k], float) for k in ("b_a", "b_g", "s_a", "s_g", "b_gps"))
        self.b_p = float(x0["b_p"])
        self.P = np.array(P0, float)
        self.cfg = cfg
        self.site_altitude = site_altitude
        imu, gps = cfg.sensors.imu, cfg.sensors.gps
        self.gm = np.array([gps.gm_sigma_h, gps.gm_sigma_h, gps.gm_sigma_v])
        # Continuous process-noise intensities on the error-state diagonal
        self.qc = np.concatenate([
            np.zeros(3), np.full(3, imu.accel_noise ** 2), np.full(3, imu.gyro_noise ** 2),
            np.full(3, imu.accel_bias_walk ** 2), np.full(3, imu.gyro_bias_walk ** 2), np.zeros(6),
            2 * self.gm ** 2 / gps.gm_tau if cfg.gps_error_states else np.zeros(3),
            [cfg.baro_bias_walk ** 2]])
        self.stats = {"gps": [0, 0], "baro": [0, 0], "zupt": [0, 0], "zaru": [0, 0]}  # [used, rejected]

    def corrected_imu(self, f_meas, w_meas):
        return (f_meas - self.b_a) / (1 + self.s_a), (w_meas - self.b_g) / (1 + self.s_g)

    # --- propagation -----------------------------------------------------------------
    def predict(self, f_meas, w_meas, dt):
        f, w = self.corrected_imu(f_meas, w_meas)
        R = quat_to_dcm(self.q)
        dq_half = quat_exp(0.5 * w * dt)
        R_mid = quat_to_dcm(quat_multiply(self.q, dq_half))
        z = self.site_altitude + self.p[2]
        g = gravity(z)
        v_new = self.v + (R_mid @ f + np.array([0.0, 0.0, -g])) * dt
        self.p = self.p + 0.5 * (self.v + v_new) * dt
        self.v = v_new
        q = quat_multiply(quat_multiply(self.q, dq_half), dq_half)
        self.q = q / np.linalg.norm(q)
        tau = self.cfg.sensors.gps.gm_tau
        self.b_gps = self.b_gps * np.exp(-dt / tau)

        Da, Dg = 1 / (1 + self.s_a), 1 / (1 + self.s_g)
        F = np.zeros((N_ERR, N_ERR))
        F[P_, V_] = np.eye(3)
        F[V_, TH] = -R @ skew(f)
        F[V_, BA] = -R * Da
        F[5, 2] = 2.0 * g / (R_EARTH + z)        # gravity weakens with height
        F[TH, TH] = -skew(w)
        F[TH, BG] = -np.diag(Dg)
        if self.cfg.scale_factor_states:
            F[V_, SA] = -R * (Da * f)
            F[TH, SG] = -np.diag(Dg * w)
        if self.cfg.gps_error_states:
            F[GP, GP] = -np.eye(3) / tau
        Fdt = F * dt
        Phi = np.eye(N_ERR) + Fdt + 0.5 * Fdt @ Fdt
        self.P = Phi @ self.P @ Phi.T + np.diag(self.qc * dt)
        self.Phi = Phi                           # kept for the Jacobian test

    # --- measurement updates -------------------------------------------------------
    def _update(self, kind, y, H, R):
        S = H @ self.P @ H.T + R
        nis = float(y @ np.linalg.solve(S, y))
        if self.cfg.gate_prob is not None and nis > chi2.ppf(self.cfg.gate_prob, len(y)):
            self.stats[kind][1] += 1
            return False, nis
        K = np.linalg.solve(S, H @ self.P).T              # P H^T S^-1 (S is symmetric)
        dx = K @ y
        IKH = np.eye(N_ERR) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self._inject(dx)
        self.stats[kind][0] += 1
        return True, nis

    def _inject(self, dx):
        """Add the estimated error into the nominal state, then reset the error to zero."""
        self.p += dx[P_]
        self.v += dx[V_]
        q = quat_multiply(self.q, quat_exp(dx[TH]))
        self.q = q / np.linalg.norm(q)
        self.b_a += dx[BA]
        self.b_g += dx[BG]
        self.s_a += dx[SA]
        self.s_g += dx[SG]
        self.b_gps += dx[GP]
        self.b_p += dx[BP]
        # The attitude error is now measured about the corrected attitude (reset Jacobian)
        G = np.eye(N_ERR)
        G[TH, TH] -= skew(0.5 * dx[TH])
        self.P = G @ self.P @ G.T

    def update_gps(self, pos, vel):
        """GPS position = p + b_gps + white noise; velocity = v + white noise."""
        gps = self.cfg.sensors.gps
        H = np.zeros((6, N_ERR))
        H[0:3, P_] = np.eye(3)
        H[3:6, V_] = np.eye(3)
        white = np.array([gps.pos_sigma_h, gps.pos_sigma_h, gps.pos_sigma_v])
        if self.cfg.gps_error_states:
            H[0:3, GP] = np.eye(3)
        else:
            white = np.hypot(white, self.gm)      # lump the correlated part in with the noise
        R = np.diag(np.concatenate([white, [gps.vel_sigma_h, gps.vel_sigma_h, gps.vel_sigma_v]]) ** 2)
        y = np.concatenate([pos - self.p - self.b_gps, vel - self.v])
        return self._update("gps", y, H, R)

    def mach_estimate(self):
        """Ground speed over the local speed of sound (the filter does not know the wind)."""
        return float(np.linalg.norm(self.v) / air(self.site_altitude + self.p[2])[2])

    def baro_locked_out(self):
        band = self.cfg.baro_mach_lockout
        return band is not None and band[0] < self.mach_estimate() < band[1]

    def update_baro(self, p_meas):
        """Static pressure. Predicted reading p_std(z) + b_p; its slope dp/dz = -rho g."""
        z = self.site_altitude + self.p[2]
        p_std, rho, _ = air(z)
        H = np.zeros((1, N_ERR))
        H[0, 2] = -rho * gravity(z)
        H[0, BP] = 1.0
        y = np.array([p_meas - (p_std + self.b_p)])
        R = np.array([[self.cfg.sensors.baro.noise ** 2]])
        return self._update("baro", y, H, R)

    def update_zupt(self):
        """The rocket is sitting still on the pad: velocity is zero."""
        H = np.zeros((3, N_ERR))
        H[:, V_] = np.eye(3)
        return self._update("zupt", -self.v, H, np.eye(3) * self.cfg.zupt_sigma ** 2)

    def update_zaru(self, w_mean, duration):
        """Sitting still, the gyros read only their bias (the truth model has no Earth rotation).
        The mean of the readings over `duration` has noise N_g^2 / duration."""
        H = np.zeros((3, N_ERR))
        H[:, BG] = np.eye(3)
        R = np.eye(3) * self.cfg.sensors.imu.gyro_noise ** 2 / duration
        return self._update("zaru", w_mean - self.b_g, H, R)


def initial_covariance(cfg: NavConfig) -> np.ndarray:
    imu, gps, baro = cfg.sensors.imu, cfg.sensors.gps, cfg.sensors.baro
    gm = np.array([gps.gm_sigma_h, gps.gm_sigma_h, gps.gm_sigma_v]) if cfg.gps_error_states else np.zeros(3)
    sf = cfg.scale_factor_states
    s = np.concatenate([np.full(3, cfg.init_pos), np.full(3, cfg.init_vel),
                        [cfg.init_roll, cfg.init_tilt, cfg.init_tilt],
                        np.full(3, imu.accel_bias_sigma), np.full(3, imu.gyro_bias_sigma),
                        np.full(3, imu.accel_scale_sigma if sf else 0.0),
                        np.full(3, imu.gyro_scale_sigma if sf else 0.0),
                        gm, [baro.bias_sigma]])
    return np.diag(s ** 2)
