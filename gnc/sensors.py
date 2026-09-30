"""Sensor models: strapdown IMU, GPS receiver and barometer.

Each sensor reads the truth trajectory and adds the errors real parts have.
The numbers are representative of the MEMS parts flown on student rocket
avionics (a consumer 6-axis IMU, a u-blox-class GPS, an MS5611-class barometer);
they are assumptions, not a specific datasheet.

IMU (200 Hz, one reading per 5 ms interval):
    f_meas = (1 + s_a) f_true + b_a + n_a,     w_meas = (1 + s_g) w_true + b_g + n_g
  (per axis)
  - White noise n with density N (units/sqrt(Hz)). Averaged over dt, one
    sample has standard deviation N / sqrt(dt). Integrated, it becomes a random
    walk: velocity random walk (accelerometer) and angle random walk (gyro).
  - Bias b = a random turn-on value (different every power-up) plus a slow
    random walk (in-run drift): b_k+1 = b_k + N_b sqrt(dt) w.
  - Scale-factor error s: the sensor's gain is off by a fixed fraction. It
    matters most under thrust: 0.3 % of 10 g is 0.03 g.
  - Saturation at the full-scale range.
GPS (10 Hz): position = truth + slowly wandering error + white noise;
  velocity (from Doppler) = truth + white noise. Errors are larger vertically
  (the satellites are all above you). The wandering part (atmospheric delay,
  orbit and clock errors) is a first-order Gauss-Markov process, correlated
  over about a minute:
      b_k+1 = exp(-dt/tau) b_k + sigma sqrt(1 - exp(-2 dt/tau)) w.
  No fix when the receiver's export-control (CoCom) limit trips: speed over
  1,000 knots (514 m/s). Optional scheduled outages.
Barometer (50 Hz): static pressure with white noise and an unknown constant
  offset (weather: the day's pressure differs from the standard atmosphere).
  Near Mach 1 the shock waves around the airframe corrupt the static port;
  modelled as an extra pressure error
      dp = Cp_port(M) q_bar,   Cp_port = C_peak exp(-((M - 1) / w)^2),
  with an illustrative magnitude (the real one depends on where the port is,
  and is found by CFD or from flight data).
"""
from dataclasses import dataclass, field

import numpy as np

from .atmosphere import G0, GAMMA
from .truth import Truth

DEG = np.pi / 180.0
KNOT = 0.514444


@dataclass(frozen=True)
class ImuSpec:
    accel_noise: float = 100e-6 * G0          # m/s^2/sqrt(Hz)   (100 ug/sqrt(Hz))
    accel_bias_sigma: float = 0.010 * G0      # m/s^2, turn-on bias (10 mg, 1 sigma)
    accel_bias_walk: float = 2e-4             # m/s^2/sqrt(s)
    accel_scale_sigma: float = 0.003          # scale-factor error (0.3 %, 1 sigma)
    accel_range: float = 16.0 * G0            # m/s^2
    gyro_noise: float = 0.005 * DEG           # rad/s/sqrt(Hz)   (0.005 deg/s/sqrt(Hz))
    gyro_bias_sigma: float = 0.5 * DEG        # rad/s, turn-on bias (0.5 deg/s, 1 sigma)
    gyro_bias_walk: float = 2e-5              # rad/s/sqrt(s)
    gyro_scale_sigma: float = 0.003
    gyro_range: float = 2000.0 * DEG          # rad/s


@dataclass(frozen=True)
class GpsSpec:
    rate: float = 10.0                        # Hz
    pos_sigma_h: float = 0.5                  # m, white noise per horizontal axis
    pos_sigma_v: float = 1.0                  # m
    gm_sigma_h: float = 1.5                   # m, correlated error per horizontal axis
    gm_sigma_v: float = 3.0                   # m
    gm_tau: float = 60.0                      # s, correlation time
    vel_sigma_h: float = 0.15                 # m/s
    vel_sigma_v: float = 0.30                 # m/s
    speed_limit: float = 1000 * KNOT          # m/s, CoCom: no fix above this
    outages: tuple = ()                       # ((t_start, t_end), ...) s from ignition


@dataclass(frozen=True)
class BaroSpec:
    rate: float = 50.0                        # Hz
    noise: float = 2.5                        # Pa (about 0.2 m at the pad)
    bias_sigma: float = 300.0                 # Pa, the day's offset from the standard atmosphere
    port_cp_peak: float = -0.04               # transonic static-port error, peak Cp (illustrative)
    port_cp_width: float = 0.10               # in Mach number


@dataclass(frozen=True)
class SensorSpec:
    imu: ImuSpec = ImuSpec()
    gps: GpsSpec = GpsSpec()
    baro: BaroSpec = BaroSpec()


@dataclass
class SensorData:
    imu_f: np.ndarray        # (N-1, 3) accelerometer, m/s^2, one per interval
    imu_w: np.ndarray        # (N-1, 3) gyro, rad/s
    b_a: np.ndarray          # (N-1, 3) true accelerometer bias
    b_g: np.ndarray          # (N-1, 3) true gyro bias
    s_a: np.ndarray          # (3,) true accelerometer scale-factor error
    s_g: np.ndarray          # (3,) true gyro scale-factor error
    gps_k: np.ndarray        # grid indices of GPS epochs
    gps_pos: np.ndarray      # (m, 3)
    gps_vel: np.ndarray      # (m, 3)
    gps_valid: np.ndarray    # (m,) bool: receiver has a fix
    b_gps: np.ndarray        # (N, 3) true correlated GPS position error, on the truth grid
    baro_k: np.ndarray       # grid indices of barometer samples
    baro_p: np.ndarray       # Pa
    b_p: float               # true barometer offset, Pa
    saturated: dict = field(default_factory=dict)


def port_error(spec: BaroSpec, mach, p_static):
    """Static-port pressure error near Mach 1, Pa."""
    mach = np.asarray(mach)
    qbar = 0.5 * GAMMA * p_static * mach ** 2
    return spec.port_cp_peak * np.exp(-((mach - 1.0) / spec.port_cp_width) ** 2) * qbar


def generate(truth: Truth, spec: SensorSpec, rng: np.random.Generator,
             transonic_port_error: bool = True) -> SensorData:
    dt = truth.dt
    n = len(truth.t) - 1
    imu = spec.imu

    def bias(sigma, walk):
        b0 = rng.normal(0.0, sigma, 3)
        steps = rng.normal(0.0, walk * np.sqrt(dt), (n, 3))
        steps[0] = 0.0
        return b0 + np.cumsum(steps, axis=0)

    b_a = bias(imu.accel_bias_sigma, imu.accel_bias_walk)
    b_g = bias(imu.gyro_bias_sigma, imu.gyro_bias_walk)
    s_a = rng.normal(0.0, imu.accel_scale_sigma, 3)
    s_g = rng.normal(0.0, imu.gyro_scale_sigma, 3)
    f = (1 + s_a) * truth.f_b + b_a + rng.normal(0.0, imu.accel_noise / np.sqrt(dt), (n, 3))
    w = (1 + s_g) * truth.w_b + b_g + rng.normal(0.0, imu.gyro_noise / np.sqrt(dt), (n, 3))
    saturated = {"accel": int(np.sum(np.abs(f) > imu.accel_range)),
                 "gyro": int(np.sum(np.abs(w) > imu.gyro_range))}
    f = np.clip(f, -imu.accel_range, imu.accel_range)
    w = np.clip(w, -imu.gyro_range, imu.gyro_range)

    gps = spec.gps
    every = int(round(1.0 / (gps.rate * dt)))
    gps_k = np.arange(0, n + 1, every)
    sig_p = np.array([gps.pos_sigma_h, gps.pos_sigma_h, gps.pos_sigma_v])
    sig_v = np.array([gps.vel_sigma_h, gps.vel_sigma_h, gps.vel_sigma_v])
    sig_gm = np.array([gps.gm_sigma_h, gps.gm_sigma_h, gps.gm_sigma_v])
    decay = np.exp(-dt / gps.gm_tau)
    kick = rng.normal(0.0, 1.0, (n + 1, 3)) * sig_gm * np.sqrt(1 - decay ** 2)
    b_gps = np.empty((n + 1, 3))
    b_gps[0] = rng.normal(0.0, 1.0, 3) * sig_gm           # stationary start
    for k in range(n):
        b_gps[k + 1] = decay * b_gps[k] + kick[k + 1]
    gps_pos = truth.r[gps_k] + b_gps[gps_k] + rng.normal(0.0, 1.0, (len(gps_k), 3)) * sig_p
    gps_vel = truth.v[gps_k] + rng.normal(0.0, 1.0, (len(gps_k), 3)) * sig_v
    valid = np.linalg.norm(truth.v[gps_k], axis=1) <= gps.speed_limit
    for t0, t1 in gps.outages:
        valid &= ~((truth.t[gps_k] >= t0) & (truth.t[gps_k] < t1))

    baro = spec.baro
    every = int(round(1.0 / (baro.rate * dt)))
    baro_k = np.arange(0, n + 1, every)
    b_p = float(rng.normal(0.0, baro.bias_sigma))
    p = truth.p_static[baro_k] + b_p + rng.normal(0.0, baro.noise, len(baro_k))
    if transonic_port_error:
        p = p + port_error(baro, truth.mach[baro_k], truth.p_static[baro_k])

    return SensorData(f, w, b_a, b_g, s_a, s_g, gps_k, gps_pos, gps_vel, valid, b_gps, baro_k, p, b_p, saturated)
