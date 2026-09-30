"""Validation of the building blocks: rotations, atmosphere, truth, sensors, filter equations."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnc.atmosphere import air, gravity
from gnc.consistency import anees_bounds
from gnc.ekf import BA, BG, BP, GP, N_ERR, P_, SA, SG, TH, V_, ErrorStateEKF, NavConfig, initial_covariance
from gnc.rotations import quat_conj, quat_exp, quat_log, quat_multiply, quat_to_dcm
from gnc.sensors import SensorSpec, generate
from gnc.truth import load

TRUTH = load("wind", pad_time=10.0)


def nominal(truth, k, **kw):
    x = dict(p=truth.r[k], v=truth.v[k], q=truth.q[k], b_a=np.zeros(3), b_g=np.zeros(3), s_a=np.zeros(3),
             s_g=np.zeros(3), b_gps=np.zeros(3), b_p=0.0)
    x.update(kw)
    return x


def test_quaternion_exp_log_and_dcm():
    rng = np.random.default_rng(0)
    for _ in range(20):
        phi = rng.normal(0, 1.0, 3)
        q = quat_exp(phi)
        assert quat_log(q) == pytest.approx(phi, abs=1e-12)
        R = quat_to_dcm(q)
        assert R @ R.T == pytest.approx(np.eye(3), abs=1e-12)
        # Rotating a vector about phi leaves the component along phi unchanged
        assert R @ phi == pytest.approx(phi, abs=1e-12)
    assert quat_log(quat_multiply(q, quat_conj(q))) == pytest.approx(np.zeros(3), abs=1e-15)


# U.S. Standard Atmosphere 1976 tabulated values (geometric altitude): z, p, rho
@pytest.mark.parametrize("z, p, rho", [(0, 101_325.0, 1.2250), (5_000, 54_048.0, 0.73643),
                                       (10_000, 26_500.0, 0.41351), (20_000, 5_529.3, 0.088910),
                                       (30_000, 1_197.0, 0.018410)])
def test_atmosphere_matches_standard_table(z, p, rho):
    pz, rz, _ = air(z)
    assert pz == pytest.approx(p, rel=1e-3)
    assert rz == pytest.approx(rho, rel=1e-3)


@pytest.mark.parametrize("z", [1400.0, 6000.0, 10_500.0, 15_000.0])
def test_barometer_slope_is_hydrostatic(z):
    # The filter's barometer Jacobian dp/dz = -rho g must match the model's own slope
    dz = 0.5
    slope = (air(z + dz)[0] - air(z - dz)[0]) / (2 * dz)
    assert slope == pytest.approx(-air(z)[1] * gravity(z), rel=1e-5)


def test_perfect_imu_dead_reckoning_reproduces_truth():
    # The strapdown mechanization, fed error-free IMU data, must fly the same
    # trajectory as the 6-DOF simulator: launch to apogee (42 s, 9 km).
    tr = TRUTH
    f = ErrorStateEKF(nominal(tr, 0), initial_covariance(NavConfig()), NavConfig(), tr.site_altitude)
    for k in range(len(tr.t) - 1):
        f.predict(tr.f_b[k], tr.w_b[k], tr.dt)
    assert np.abs(tr.r[-1] - f.p).max() < 0.01                      # m, after 9 km of flight
    assert np.abs(tr.v[-1] - f.v).max() < 1e-3                      # m/s
    assert np.linalg.norm(quat_log(quat_multiply(quat_conj(f.q), tr.q[-1]))) < 1e-8   # rad


def test_pad_specific_force_is_one_g_up():
    k = 10
    g_body = quat_to_dcm(TRUTH.q[k]).T @ np.array([0, 0, gravity(TRUTH.site_altitude)])
    assert TRUTH.f_b[k] == pytest.approx(g_body, abs=1e-9)
    assert TRUTH.w_b[k] == pytest.approx(np.zeros(3), abs=1e-12)


ALL_STATES = np.concatenate([[0.3, -0.2, 0.1], [0.02, -0.01, 0.03], [2e-4, -3e-4, 1e-4], [2e-3, -1e-3, 3e-3],
                             [1e-4, -2e-4, 1e-4], [3e-4, -2e-4, 1e-4], [1e-4, 2e-4, -3e-4], [0.5, -0.4, 0.3], [5.0]])
ALTITUDE_ONLY = np.zeros(N_ERR)
ALTITUDE_ONLY[2] = 100.0


@pytest.mark.parametrize("t0, steps, dx", [
    (3.0, 40, ALL_STATES),          # T+3 s, 0.2 s of 10 g thrust and weathercocking: every term in F
    (15.0, 2000, ALTITUDE_ONLY),    # 10 s of coast from 100 m too high: the gravity-gradient term
])
def test_error_state_jacobian_matches_finite_difference(t0, steps, dx):
    # Fly two nominal states that differ by a small error dx, with the same IMU
    # data. Their difference must follow the linear error dynamics dx(t) = Phi dx(0).
    tr = TRUTH
    k0 = int(np.searchsorted(tr.t, t0))
    cfg = NavConfig()
    x0 = nominal(tr, k0, b_a=np.array([0.05, -0.03, 0.02]), b_g=np.array([1e-3, -2e-3, 5e-4]),
                 s_a=np.array([2e-3, -1e-3, 3e-3]), s_g=np.array([-2e-3, 1e-3, 2e-3]))
    x1 = dict(p=x0["p"] + dx[P_], v=x0["v"] + dx[V_], q=quat_multiply(x0["q"], quat_exp(dx[TH])),
              b_a=x0["b_a"] + dx[BA], b_g=x0["b_g"] + dx[BG], s_a=x0["s_a"] + dx[SA], s_g=x0["s_g"] + dx[SG],
              b_gps=x0["b_gps"] + dx[GP], b_p=x0["b_p"] + dx[BP])
    P0 = np.eye(N_ERR)
    f0 = ErrorStateEKF(x0, P0, cfg, tr.site_altitude)
    f1 = ErrorStateEKF(x1, P0, cfg, tr.site_altitude)
    Phi = np.eye(N_ERR)
    for k in range(k0, k0 + steps):
        f_m = (1 + x0["s_a"]) * tr.f_b[k] + x0["b_a"]
        w_m = (1 + x0["s_g"]) * tr.w_b[k] + x0["b_g"]
        f0.predict(f_m, w_m, tr.dt)
        f1.predict(f_m, w_m, tr.dt)
        Phi = f0.Phi @ Phi
    actual = np.concatenate([f1.p - f0.p, f1.v - f0.v, quat_log(quat_multiply(quat_conj(f0.q), f1.q)),
                             f1.b_a - f0.b_a, f1.b_g - f0.b_g, f1.s_a - f0.s_a, f1.s_g - f0.s_g,
                             f1.b_gps - f0.b_gps, [f1.b_p - f0.b_p]])
    predicted = Phi @ dx
    # Agreement to second order in dx (relative to how much each state changed)
    scale = np.abs(actual - dx) + np.abs(dx) * 1e-3 + 1e-12
    assert np.all(np.abs(actual - predicted) < 0.01 * scale + 1e-9)


def test_kalman_update_matches_scalar_formula():
    # Uncorrelated prior sigma_p on position, GPS noise sigma_z: the posterior
    # variance is sigma_p^2 sigma_z^2 / (sigma_p^2 + sigma_z^2).
    cfg = NavConfig(gps_error_states=False, gate_prob=None)
    f = ErrorStateEKF(nominal(TRUTH, 0), initial_covariance(cfg), cfg, TRUTH.site_altitude)
    sp = cfg.init_pos
    gps = cfg.sensors.gps
    sz = np.hypot([gps.pos_sigma_h, gps.pos_sigma_h, gps.pos_sigma_v], [gps.gm_sigma_h, gps.gm_sigma_h, gps.gm_sigma_v])
    f.update_gps(TRUTH.r[0] + np.array([1.0, -2.0, 3.0]), TRUTH.v[0])
    assert np.diag(f.P)[P_] == pytest.approx(sp ** 2 * sz ** 2 / (sp ** 2 + sz ** 2), rel=1e-9)
    # ... and the estimate moves by the gain times the innovation
    gain = sp ** 2 / (sp ** 2 + sz ** 2)
    assert f.p - TRUTH.r[0] == pytest.approx(gain * np.array([1.0, -2.0, 3.0]), rel=1e-9)


def test_sensor_error_statistics():
    rng = np.random.default_rng(3)
    spec = SensorSpec()
    s = generate(TRUTH, spec, rng)
    pad = TRUTH.t[:-1] < -1.0
    dt = TRUTH.dt
    noise_a = s.imu_f[pad] - (1 + s.s_a) * TRUTH.f_b[pad] - s.b_a[pad]
    noise_g = s.imu_w[pad] - (1 + s.s_g) * TRUTH.w_b[pad] - s.b_g[pad]
    assert noise_a.std(axis=0) == pytest.approx(spec.imu.accel_noise / np.sqrt(dt), rel=0.05)
    assert noise_g.std(axis=0) == pytest.approx(spec.imu.gyro_noise / np.sqrt(dt), rel=0.05)
    # No GPS fix exactly when faster than the CoCom limit
    fast = np.linalg.norm(TRUTH.v[s.gps_k], axis=1) > spec.gps.speed_limit
    assert np.array_equal(~s.gps_valid, fast)
    assert fast.sum() > 5                                    # the rocket does exceed 1,000 knots


def test_gauss_markov_gps_error_is_stationary():
    rng = np.random.default_rng(5)
    spec = SensorSpec()
    b = np.stack([generate(TRUTH, spec, rng).b_gps for _ in range(60)])
    sd = b[:, ::200].reshape(-1, 3).std(axis=0)
    gps = spec.gps
    assert sd == pytest.approx([gps.gm_sigma_h, gps.gm_sigma_h, gps.gm_sigma_v], rel=0.1)


def test_anees_bounds():
    # Two-sided 95 % bounds on the mean of M chi-square(n) samples
    lo, hi = anees_bounds(3, 1)
    assert (lo, hi) == pytest.approx((0.2158, 9.3484), rel=1e-3)
    lo, hi = anees_bounds(3, 200)
    assert lo < 3 < hi and hi - lo < 0.7
