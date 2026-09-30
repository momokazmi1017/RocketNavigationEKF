"""Quaternion and rotation utilities (same conventions as FlightSim6DOF).

Quaternions are [w, x, y, z] and rotate body vectors into the navigation
(ENU) frame: v_nav = R(q) v_body. Small rotations are handled with the
exponential map: a rotation vector phi (axis times angle, rad) becomes the
quaternion Exp(phi) = [cos(|phi|/2), sin(|phi|/2) phi/|phi|], and Log is its
inverse. The filter's attitude error is such a rotation vector, in body axes:
    q_true = q_est (x) Exp(dtheta).
"""
import math

import numpy as np


def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_dcm(q):
    """Rotation matrix (body -> nav) of a unit quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_exp(phi):
    """Quaternion of the rotation vector phi (rad)."""
    phi = np.asarray(phi, float)
    angle = math.sqrt(phi @ phi)
    if angle < 1e-8:
        q = np.array([1.0, *(0.5 * phi)])          # second-order accurate
        return q / np.linalg.norm(q)
    s = math.sin(0.5 * angle) / angle
    return np.array([math.cos(0.5 * angle), *(s * phi)])


def quat_log(q):
    """Rotation vector (rad) of a unit quaternion, taking the short way round."""
    q = np.asarray(q, float)
    if q[0] < 0:
        q = -q
    v = q[1:]
    s = math.sqrt(v @ v)
    if s < 1e-12:
        return 2.0 * v
    return 2.0 * math.atan2(s, q[0]) * v / s


def skew(a):
    """Cross-product matrix: skew(a) @ b = a x b."""
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
