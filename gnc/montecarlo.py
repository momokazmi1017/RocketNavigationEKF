"""Monte Carlo runs of the navigation filter.

Every run flies the same truth trajectory with a fresh set of sensor errors
(new turn-on biases, noise, GPS wander and barometer offset) and a fresh
initial estimation error. Runs are independent and seeded, so a study is
reproducible, and they are spread over all CPU cores.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from .ekf import NavConfig
from .navigation import NavResult, run
from .sensors import SensorSpec, generate
from .truth import load


@dataclass(frozen=True)
class Case:
    truth: str = "wind"
    sensors: SensorSpec = SensorSpec()        # what the real sensors do
    nav: NavConfig = NavConfig()              # what the filter believes and does
    transonic_port_error: bool = True
    pad_time: float = 30.0
    record_every: int = 4


@lru_cache(maxsize=4)
def _truth(name, pad_time):
    return load(name, pad_time)


def run_one(case: Case, seed: int) -> NavResult:
    truth = _truth(case.truth, case.pad_time)
    rng = np.random.default_rng(seed)
    sens = generate(truth, case.sensors, rng, case.transonic_port_error)
    return run(truth, sens, case.nav, rng, record_every=case.record_every)


def _job(args):
    return run_one(*args)


def run_many(case: Case, n_runs: int, seed0: int = 1000, workers: int | None = None) -> list[NavResult]:
    workers = workers or os.cpu_count()
    jobs = [(case, seed0 + i) for i in range(n_runs)]
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_job, jobs, chunksize=max(1, n_runs // (4 * workers))))


def stack(results: list[NavResult], attr: str) -> np.ndarray:
    return np.stack([getattr(r, attr) for r in results])


def with_outage(case: Case, t0: float, t1: float) -> Case:
    gps = replace(case.sensors.gps, outages=((t0, t1),))
    return replace(case, sensors=replace(case.sensors, gps=gps))
