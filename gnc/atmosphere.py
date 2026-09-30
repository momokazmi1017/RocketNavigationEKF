"""Standard atmosphere and gravity, as the flight computer knows them.

U.S. Standard Atmosphere 1976, first three layers (0-32 km geopotential, which
covers this rocket's 10.5 km ceiling). The same model (and constants) as the
FlightSim6DOF truth, so the barometer is only wrong by its own errors.
Within a layer with base (Hb, Tb, pb) and lapse rate L:
    T = Tb + L (H - Hb)
    p = pb (T / Tb) ^ (-g0 / (R L))           if L != 0
    p = pb exp(-g0 (H - Hb) / (R Tb))         if L == 0
Geopotential altitude H = Re z / (Re + z) for geometric altitude z.

Hydrostatic balance gives the barometer's sensitivity, used by the filter:
    dp/dz = -rho g(z).
"""
import math

G0 = 9.80665               # m/s^2
R_AIR = 287.05287          # J/kg-K
GAMMA = 1.4
R_EARTH = 6_356_766.0      # m

_LAYERS = [(0.0, 288.15, -0.0065), (11_000.0, 216.65, 0.0), (20_000.0, 216.65, 0.0010)]
_H_TOP = 32_000.0


def _layer_pressure(pb, tb, lapse, dh):
    if lapse == 0.0:
        return pb * math.exp(-G0 * dh / (R_AIR * tb))
    return pb * ((tb + lapse * dh) / tb) ** (-G0 / (R_AIR * lapse))


_P_BASE = [101_325.0]
for (_hb, _tb, _L), (_hn, _, _) in zip(_LAYERS, _LAYERS[1:]):
    _P_BASE.append(_layer_pressure(_P_BASE[-1], _tb, _L, _hn - _hb))


def gravity(z: float) -> float:
    """Gravitational acceleration (m/s^2) at geometric altitude z (m MSL)."""
    return G0 * (R_EARTH / (R_EARTH + z)) ** 2


def air(z: float) -> tuple[float, float, float]:
    """(pressure Pa, density kg/m^3, speed of sound m/s) at geometric altitude z (m MSL)."""
    h = min(max(R_EARTH * z / (R_EARTH + z), -1000.0), _H_TOP)
    i = 2 if h >= _LAYERS[2][0] else 1 if h >= _LAYERS[1][0] else 0
    hb, tb, lapse = _LAYERS[i]
    T = tb + lapse * (h - hb)
    p = _layer_pressure(_P_BASE[i], tb, lapse, h - hb)
    return p, p / (R_AIR * T), math.sqrt(GAMMA * R_AIR * T)


def pressure(z: float) -> float:
    return air(z)[0]
