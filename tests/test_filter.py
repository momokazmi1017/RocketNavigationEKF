"""Closed-loop validation of the navigation filter against the 6-DOF truth."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnc.consistency import anees, anees_bounds, fraction_inside
from gnc.ekf import NavConfig, active_states
from gnc.montecarlo import Case, run_many, run_one, stack, with_outage

N_RUNS = 48


@pytest.fixture(scope="module")
def monte_carlo():
    return run_many(Case(), N_RUNS)


def test_filter_is_consistent(monte_carlo):
    # Average NEES over the flight inside its 95 % chi-square band, for
    # position, velocity, attitude and the full state.
    t = monte_carlo[0].t
    fl = t > 0
    n_full = len(active_states(NavConfig()))
    for block, n in [("position", 3), ("velocity", 3), ("attitude", 3), ("full", n_full)]:
        lo, hi = anees_bounds(n, N_RUNS)
        A = anees(monte_carlo, block)[fl]
        assert lo < A.mean() < hi, block
        assert np.mean((A > lo) & (A < hi)) > 0.8, block


def test_errors_inside_three_sigma(monte_carlo):
    E = stack(monte_carlo, "err")[:, monte_carlo[0].t > 0, :9]
    S = stack(monte_carlo, "sigma")[:, monte_carlo[0].t > 0, :9]
    assert fraction_inside(E, S) > 0.99


def test_liftoff_detected_promptly():
    res = run_one(Case(), 1)
    assert 0.0 < res.t_liftoff < 0.15          # the rocket starts moving about 0.05 s after ignition


def test_recovers_from_gps_dropout():
    case = with_outage(Case(), 2.0, 22.0)
    res = run_one(case, 2)
    t = res.t
    during = (t > 2.0) & (t < 22.0)
    after = t > 25.0
    h_sigma = np.hypot(res.sigma[:, 0], res.sigma[:, 1])
    assert h_sigma[during].max() > 3 * h_sigma[t < 2].max()      # uncertainty grows without GPS...
    assert h_sigma[after].max() < 1.5 * h_sigma[t < 2].max()     # ...and collapses when the fix returns
    assert np.all(np.abs(res.err[:, :6]) < 4 * res.sigma[:, :6])  # errors stay inside the bounds throughout
    assert res.stats["gps"][1] == 0                               # the returning fix is not rejected
