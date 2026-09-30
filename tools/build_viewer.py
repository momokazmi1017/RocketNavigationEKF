"""Build the interactive 3D navigation replay.

Flies the crosswind mission through the navigation filter twice, once
normally and once with a 20 s GPS dropout (same sensor errors, same seed),
packs the truth, the estimate, the filter's uncertainty, the GPS fixes and the
sensor status into compact JSON, and inlines it into viewer/template.html:
    viewer/nav_viewer.html          one self-contained page (opens from disk, or GitHub Pages)
    viewer/_publish/nav_viewer.html the same page without the document wrapper

Run from the GNC folder:
    python tools/build_viewer.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from gnc.atmosphere import air
from gnc.ekf import P_, TH, V_, NavConfig
from gnc.montecarlo import Case, _truth, with_outage
from gnc.navigation import run
from gnc.sensors import generate

VIEWER = ROOT / "viewer"
SEED = 7
T_START = -5.0                  # s: show the last seconds of pad calibration
OUTAGE = (2.0, 22.0)
DEG = 180 / np.pi
R = lambda a, n: [round(float(x), n) for x in a]   # noqa: E731


def pack(case: Case, label: str, blurb: str) -> dict:
    truth = _truth(case.truth, case.pad_time)
    rng = np.random.default_rng(SEED)
    sens = generate(truth, case.sensors, rng, case.transonic_port_error)
    res = run(truth, sens, case.nav, rng, record_every=4)
    gps = case.sensors.gps
    band = case.nav.baro_mach_lockout

    # Engine throttle proxy from the axial specific force during the burn
    fx = truth.f_b[:, 0]
    burning = truth.phase[:-1] == 1
    full = np.median(fx[burning])

    gps_t = truth.t[sens.gps_k]
    rows = []
    for j, k in enumerate(res.k):
        t = truth.t[k]
        if t < T_START:
            continue
        kk = min(k, len(fx) - 1)
        thr = float(np.clip(fx[kk] / full, 0, 1)) if truth.phase[k] == 1 else 0.0
        z = truth.site_altitude + res.p[j, 2]
        m_est = np.linalg.norm(res.v[j]) / air(z)[2]
        pad = t < res.t_liftoff
        baro = 0 if pad or not (band and band[0] < m_est < band[1]) else 1          # 0 used, 1 locked out
        # GPS status at the latest epoch: 0 fix, 1 over the CoCom speed limit, 2 scheduled dropout
        e = max(0, int(np.searchsorted(gps_t, t + 1e-9)) - 1)
        if sens.gps_valid[e]:
            g = 0
        elif np.linalg.norm(truth.v[sens.gps_k[e]]) > gps.speed_limit:
            g = 1
        else:
            g = 2
        rows.append([
            round(float(t), 3),
            *R(truth.r[k], 2), *R(truth.q[k], 5),
            *R(res.p[j], 2), *R(res.q[j], 5),
            *R(3 * res.sigma[j, P_], 2), *R(3 * res.sigma[j, TH] * DEG, 3),
            *R(res.err[j, TH] * DEG, 3), *R(res.err[j, V_], 3), *R(3 * res.sigma[j, V_], 3),
            int(truth.phase[k]), round(float(np.linalg.norm(truth.v[k])), 1), round(float(truth.mach[k]), 3),
            round(thr, 3), int(pad), baro, g,
        ])

    fixes = [[round(float(gps_t[i]), 2), *R(sens.gps_pos[i], 2)]
             for i in range(len(gps_t)) if sens.gps_valid[i] and gps_t[i] >= T_START]
    ev = truth.meta["events"]
    m = truth.t >= 0
    t_m1 = truth.t[m][np.where(np.diff(np.sign(truth.mach[m] - 1.0)))[0]]
    fast = truth.t[np.linalg.norm(truth.v, axis=1) > gps.speed_limit]
    events = [{"name": "Liftoff", "t": 0.0}, {"name": "Mach 1", "t": float(t_m1[0])},
              {"name": "Burnout", "t": ev["burnout"]}, {"name": "Apogee", "t": float(truth.t[-1])}]
    spans = [{"name": "No GPS: over 1,000 kn", "t0": float(fast.min()), "t1": float(fast.max())}]
    for t0, t1 in gps.outages:
        spans.append({"name": "GPS dropout", "t0": t0, "t1": t1})
    fl = res.t > 0
    return {
        "label": label, "blurb": blurb,
        "columns": ["t", "rE", "rN", "rU", "qw", "qx", "qy", "qz", "eE", "eN", "eU", "ew", "ex", "ey", "ez",
                    "s3E", "s3N", "s3U", "s3roll", "s3pitch", "s3yaw", "eroll", "epitch", "eyaw",
                    "evE", "evN", "evU", "s3vE", "s3vN", "s3vU", "phase", "speed", "mach", "throttle", "pad",
                    "baro", "gps"],
        "rows": rows, "gps": fixes, "events": events, "spans": spans,
        "t_liftoff": res.t_liftoff,
        "summary": {"max_err_m": float(np.abs(res.err[fl][:, P_]).max()),
                    "apogee_err_m": float(np.linalg.norm(res.err[-1, P_]))},
    }


def main():
    truth = _truth("wind", 30.0)
    base = Case()
    data = {
        "vehicle": truth.meta["vehicle"],
        "site": {"altitude_m": truth.site_altitude, "rail_elevation_deg": truth.meta["rail_elevation_deg"],
                 "rail_azimuth_deg": truth.meta["rail_azimuth_deg"]},
        "lockout": list(base.nav.baro_mach_lockout),
        "flights": [
            pack(base, "Nominal",
                 "GPS goes blind for 1.7 s above 1,000 knots, right at burnout; the filter coasts on the IMU."),
            pack(with_outage(base, *OUTAGE), "20 s GPS dropout",
                 "Same flight and sensor errors, but GPS is lost from T+2 s to T+22 s. Watch the bubble grow, "
                 "then snap back when the fix returns."),
        ],
    }
    blob = json.dumps(data, separators=(",", ":"))
    page = (VIEWER / "template.html").read_text(encoding="utf-8").replace("/*__NAV_DATA__*/null", blob)
    out = VIEWER / "nav_viewer.html"
    out.write_text('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n' + page + "\n</html>\n",
                   encoding="utf-8")
    (VIEWER / "_publish").mkdir(exist_ok=True)
    (VIEWER / "_publish" / "nav_viewer.html").write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
