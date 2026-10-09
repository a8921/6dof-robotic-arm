"""
dh_kinematics.py — Forward & Inverse Kinematics for a 6-DOF Robotic Arm
IIT Bhilai | Ashish Devadas

DH Parameters (Modified Craig convention: T = Rx(α)·Tx(a)·Rz(θ)·Tz(d)):
  Joint | a     | alpha  | d     | θ_offset | Role
  ------+-------+--------+-------+----------+------
    1   | 0     |  0     | 0.640 | 0        | Waist
    2   | 0     | +90°   | 0     | +90°     | Shoulder
    3   | 0.800 |  0     | 0     | 0        | Elbow (upper arm)
    4   | 0.200 | +90°   | 0     | 0        | Forearm
    5   | 0     | -90°   | 0     | 0        | Wrist pitch
    6   | 0     |  0     | 0     | 0        | Tool flange

At q=0 the arm stands vertically (home = [0, 0, 1.64] m above base).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

# ── DH Parameters (metres, radians) ──────────────────────────────────────────
@dataclass
class DHParams:
    """Modified DH parameters for one joint."""
    a:     float   # link length (m)
    alpha: float   # link twist (rad)
    d:     float   # link offset (m)
    theta_offset: float = 0.0   # fixed joint offset (rad)

# 6-DOF arm DH parameters — upright candle at q=0
# Modified Craig convention: T = Rx(alpha) * Tx(a) * Rz(theta+offset) * Tz(d)
ARM_DH = [
    DHParams(a=0.0,    alpha=0.0,          d=0.640, theta_offset=0.0),        # J1 waist
    DHParams(a=0.0,    alpha=np.pi/2,      d=0.0,   theta_offset=np.pi/2),    # J2 shoulder
    DHParams(a=0.800,  alpha=0.0,          d=0.0,   theta_offset=0.0),        # J3 elbow
    DHParams(a=0.200,  alpha=np.pi/2,      d=0.0,   theta_offset=0.0),        # J4 forearm
    DHParams(a=0.0,    alpha=-np.pi/2,     d=0.0,   theta_offset=0.0),        # J5 wrist pitch
    DHParams(a=0.0,    alpha=0.0,          d=0.0,   theta_offset=0.0),        # J6 tool flange
]

# Joint limits (rad) — conservative industrial limits
JOINT_LIMITS = np.array([
    [-np.pi,       np.pi      ],   # J1
    [-np.pi/2,     np.pi/2    ],   # J2
    [-np.pi*0.75,  np.pi*0.75 ],   # J3
    [-np.pi,       np.pi      ],   # J4
    [-np.pi/2,     np.pi/2    ],   # J5
    [-np.pi,       np.pi      ],   # J6
])

N_JOINTS = 6


# ── Homogeneous transform helpers ─────────────────────────────────────────────
def rot_x(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0],
                     [0, ca,-sa, 0],
                     [0, sa, ca, 0],
                     [0, 0,  0,  1]], dtype=float)

def rot_z(t: float) -> np.ndarray:
    ct, st = np.cos(t), np.sin(t)
    return np.array([[ct,-st, 0, 0],
                     [st, ct, 0, 0],
                     [0,   0, 1, 0],
                     [0,   0, 0, 1]], dtype=float)

def trans_x(a: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = a
    return T

def trans_z(d: float) -> np.ndarray:
    T = np.eye(4)
    T[2, 3] = d
    return T


# ── Modified DH transform for joint i ────────────────────────────────────────
def dh_transform(p: DHParams, theta: float) -> np.ndarray:
    """
    T = Rx(alpha) * Tx(a) * Rz(theta + offset) * Tz(d)
    Modified (Craig) DH convention.
    """
    return rot_x(p.alpha) @ trans_x(p.a) @ rot_z(theta + p.theta_offset) @ trans_z(p.d)


# ── Forward Kinematics ────────────────────────────────────────────────────────
def forward_kinematics(q: np.ndarray,
                       dh: list = ARM_DH) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Compute end-effector pose and all joint frames.

    Args:
        q:  joint angles (rad), shape (6,)
        dh: DH parameter list

    Returns:
        T_ee:   4×4 end-effector transform (base frame)
        frames: list of 4×4 transforms for each joint (base frame)
    """
    T = np.eye(4)
    frames = []
    for i, (p, qi) in enumerate(zip(dh, q)):
        T = T @ dh_transform(p, qi)
        frames.append(T.copy())
    return T, frames


def ee_position(q: np.ndarray) -> np.ndarray:
    """Return end-effector XYZ position (m)."""
    T, _ = forward_kinematics(q)
    return T[:3, 3]


def ee_pose(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (position [3,], rotation_matrix [3,3])."""
    T, _ = forward_kinematics(q)
    return T[:3, 3], T[:3, :3]


# ── Geometric Jacobian ────────────────────────────────────────────────────────
def jacobian(q: np.ndarray, dh: list = ARM_DH) -> np.ndarray:
    """
    Compute the 6×6 geometric Jacobian.
    J = [J_v; J_w]  where J_v (3×6) is linear velocity, J_w (3×6) angular.

    For Modified Craig DH: T_i = Rx(alpha) * Tx(a) * Rz(theta) * Tz(d)
    Joint i rotates about z-axis of the frame AFTER applying Rx(alpha)*Tx(a)
    but BEFORE Rz(theta)*Tz(d).  We compute that intermediate frame explicitly.
    """
    _, frames = forward_kinematics(q, dh)
    T_ee = frames[-1]
    p_ee = T_ee[:3, 3]

    J = np.zeros((6, N_JOINTS))
    T_prev = np.eye(4)

    for i in range(N_JOINTS):
        p = dh[i]
        # Frame after Rx(alpha)*Tx(a) — this is where the joint z-axis lives
        T_pre_joint = T_prev @ rot_x(p.alpha) @ trans_x(p.a)
        z_i = T_pre_joint[:3, 2]     # z-axis joint i rotates about (world frame)
        p_i = T_pre_joint[:3, 3]     # origin of that intermediate frame
        J[:3, i] = np.cross(z_i, p_ee - p_i)
        J[3:, i] = z_i
        T_prev = frames[i]

    return J


# ── Inverse Kinematics (Jacobian Pseudoinverse + Damped LS) ──────────────────
def inverse_kinematics(
    target_pos:    np.ndarray,
    target_rot:    Optional[np.ndarray] = None,
    q_init:        Optional[np.ndarray] = None,
    max_iter:      int   = 500,
    pos_tol:       float = 1e-4,
    orient_weight: float = 0.3,
    damping:       float = 0.05,
    step_size:     float = 0.8,
) -> tuple[np.ndarray, bool, float]:
    """
    Damped Least Squares (Levenberg-Marquardt style) IK.

    Args:
        target_pos:    desired EE position [3]
        target_rot:    desired EE rotation matrix [3,3] (None = position only)
        q_init:        starting joint config [6] (None = random)
        max_iter:      iteration limit
        pos_tol:       position error threshold (m)
        orient_weight: weight of orientation error vs position
        damping:       DLS damping factor (λ)
        step_size:     gradient step scale

    Returns:
        q:        solution joint angles [6]
        success:  converged within tolerance
        pos_err:  final position error (m)
    """
    if q_init is None:
        q = np.random.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    else:
        q = q_init.copy()

    for _ in range(max_iter):
        T_cur, _ = forward_kinematics(q)
        pos_cur = T_cur[:3, 3]
        rot_cur = T_cur[:3, :3]

        # Position error
        e_pos = target_pos - pos_cur
        pos_err = np.linalg.norm(e_pos)

        # Orientation error (axis-angle from rotation matrix)
        if target_rot is not None:
            R_err = target_rot @ rot_cur.T
            # Rodrigues formula for rotation vector
            angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
            if abs(angle) < 1e-8:
                e_rot = np.zeros(3)
            else:
                e_rot = angle / (2 * np.sin(angle)) * np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ])
            e_task = np.concatenate([e_pos, orient_weight * e_rot])
        else:
            e_task = e_pos

        if pos_err < pos_tol:
            return q, True, pos_err

        # Jacobian (3×6 for pos-only, 6×6 with orientation)
        J_full = jacobian(q)
        J = J_full if target_rot is not None else J_full[:3, :]

        # Damped least squares: dq = J^T (J J^T + λ²I)^{-1} e
        JJT = J @ J.T
        lam2 = damping ** 2
        dq = J.T @ np.linalg.solve(JJT + lam2 * np.eye(JJT.shape[0]), e_task)
        q = q + step_size * dq

        # Clamp to joint limits
        q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    return q, False, np.linalg.norm(target_pos - ee_position(q))


def ik_with_restarts(
    target_pos: np.ndarray,
    target_rot: Optional[np.ndarray] = None,
    n_restarts: int = 8,
    **ik_kwargs,
) -> tuple[np.ndarray, bool, float]:
    """
    Run IK with multiple random restarts; return best solution.
    Improves reliability for targets with multiple IK solutions.
    """
    best_q, best_success, best_err = None, False, np.inf

    for _ in range(n_restarts):
        q, success, err = inverse_kinematics(target_pos, target_rot, **ik_kwargs)
        if err < best_err:
            best_q, best_success, best_err = q, success, err
        if success:
            break

    return best_q, best_success, best_err


# ── Workspace Analysis ────────────────────────────────────────────────────────
def sample_workspace(n_samples: int = 5000) -> np.ndarray:
    """
    Sample the reachable workspace by random joint configurations.
    Returns array of EE positions (n_samples × 3).
    """
    positions = np.zeros((n_samples, 3))
    for i in range(n_samples):
        q = np.random.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        positions[i] = ee_position(q)
    return positions


if __name__ == "__main__":
    # Quick sanity check
    q_home = np.zeros(6)
    T, frames = forward_kinematics(q_home)
    print("Home config EE position:", T[:3, 3].round(4), "m")
    print("Reach from base:        ", np.linalg.norm(T[:3, 3]).round(3), "m")

    target = np.array([1.2, 0.3, 0.8])
    q_sol, ok, err = ik_with_restarts(target)
    print(f"\nIK target: {target}")
    print(f"Converged: {ok}  |  position error: {err*1000:.2f} mm")
    print(f"Solution q (deg): {np.degrees(q_sol).round(2)}")
