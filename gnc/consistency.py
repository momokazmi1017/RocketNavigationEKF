"""Filter consistency tests (Bar-Shalom, Li & Kirubarajan 2001, section 5.4).

A Kalman filter reports its own uncertainty, P. It is *consistent* when that
uncertainty is honest: the actual errors are as big as P says, no bigger (an
overconfident filter ignores good measurements and can diverge) and no smaller
(an underconfident filter wastes information).

Test: the normalised estimation error squared, NEES = e^T P^-1 e. For an
honest n-state filter each NEES is chi-square with n degrees of freedom (mean n).
Averaged over M independent Monte Carlo runs at the same time step, M * ANEES
is chi-square with M n degrees of freedom, which gives a two-sided 95 %
acceptance band for ANEES.
"""
import numpy as np
from scipy.stats import chi2


def anees_bounds(n_states: int, n_runs: int, prob: float = 0.95) -> tuple[float, float]:
    lo, hi = chi2.ppf([(1 - prob) / 2, (1 + prob) / 2], n_states * n_runs)
    return lo / n_runs, hi / n_runs


def fraction_inside(err: np.ndarray, sigma: np.ndarray, k: float = 3.0) -> float:
    """Fraction of error samples inside +/- k sigma (0.9973 expected for 3 sigma, if Gaussian)."""
    return float(np.mean(np.abs(err) <= k * sigma))


def anees(results, block: str) -> np.ndarray:
    """ANEES time history of one NEES block over a list of NavResults."""
    return np.mean([r.nees[block] for r in results], axis=0)


def summary(results, n_full: int, t_min: float = 0.0) -> dict:
    """Consistency and accuracy of a Monte Carlo set, over the flight (t > t_min)."""
    t = results[0].t
    fl = t > t_min
    M = len(results)
    out = {"runs": M}
    for block, n in [("position", 3), ("velocity", 3), ("attitude", 3), ("full", n_full)]:
        A = anees(results, block)[fl]
        lo, hi = anees_bounds(n, M)
        out[f"ANEES {block}"] = float(A.mean())
        out[f"ANEES {block} per state"] = float(A.mean() / n)
        out[f"ANEES {block} inside 95% band"] = float(np.mean((A >= lo) & (A <= hi)))
    E = np.stack([r.err for r in results])[:, fl]
    S = np.stack([r.sigma for r in results])[:, fl]
    out["inside 3 sigma, nav states"] = fraction_inside(E[..., :9], S[..., :9])
    end = E[:, -1]
    deg = 180 / np.pi
    out["RMS at apogee: position 3D (m)"] = float(np.sqrt(np.mean(np.sum(end[:, 0:3] ** 2, axis=1))))
    out["RMS at apogee: altitude (m)"] = float(np.sqrt(np.mean(end[:, 2] ** 2)))
    out["RMS at apogee: velocity 3D (m/s)"] = float(np.sqrt(np.mean(np.sum(end[:, 3:6] ** 2, axis=1))))
    out["RMS at apogee: tilt (deg)"] = float(np.sqrt(np.mean(np.sum(end[:, 7:9] ** 2, axis=1))) * deg)
    out["RMS at apogee: roll (deg)"] = float(np.sqrt(np.mean(end[:, 6] ** 2)) * deg)
    out["worst RMS altitude error in flight (m)"] = float(np.sqrt(np.mean(E[..., 2] ** 2, axis=0)).max())
    out["worst altitude error, any run (m)"] = float(np.abs(E[..., 2]).max())
    return out
