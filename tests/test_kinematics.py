"""
tests/test_kinematics.py — Unit tests for 6-DOF arm kinematics
IIT Bhilai | Ashish Devadas
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src import (
    forward_kinematics, inverse_kinematics, ik_with_restarts,
    jacobian, ee_position,
    JOINT_LIMITS, N_JOINTS,
)


# ── Forward kinematics ────────────────────────────────────────────────────────

def test_fk_home_position():
    """Home config (all zeros) should produce a valid 4×4 transform."""
    q = np.zeros(N_JOINTS)
    T, frames = forward_kinematics(q)
    assert T.shape == (4, 4)
    assert len(frames) == N_JOINTS
    # Bottom row of homogeneous matrix
    np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-9)


def test_fk_rotation_orthogonal():
    """Rotation part of FK result must be orthogonal (R @ R^T ≈ I)."""
    q = np.radians([30, -45, 60, 0, 30, 0])
    T, _ = forward_kinematics(q)
    R = T[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_fk_determinant():
    """Rotation determinant must be +1 (proper rotation, no reflection)."""
    for trial in range(5):
        q = np.random.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        T, _ = forward_kinematics(q)
        R = T[:3, :3]
        assert abs(np.linalg.det(R) - 1.0) < 1e-9


def test_ee_position_consistent_with_fk():
    """ee_position() must match the FK translation vector."""
    q = np.radians([-60, 20, -30, 45, -20, 60])
    T, _ = forward_kinematics(q)
    pos_fk = T[:3, 3]
    pos_ee = ee_position(q)
    np.testing.assert_allclose(pos_fk, pos_ee, atol=1e-12)


# ── Jacobian ─────────────────────────────────────────────────────────────────

def test_jacobian_shape():
    q = np.zeros(N_JOINTS)
    J = jacobian(q)
    assert J.shape == (6, N_JOINTS)


def test_jacobian_numerical_consistency():
    """Analytical Jacobian top 3 rows (linear velocity) vs finite differences."""
    q = np.radians([15, -20, 35, 10, -15, 5])
    J_analytical = jacobian(q)[:3, :]  # linear velocity part

    eps = 1e-5
    J_num = np.zeros((3, N_JOINTS))
    p0 = ee_position(q)
    for i in range(N_JOINTS):
        dq = np.zeros(N_JOINTS)
        dq[i] = eps
        J_num[:, i] = (ee_position(q + dq) - p0) / eps

    np.testing.assert_allclose(J_analytical, J_num, atol=1e-4)


# ── Inverse kinematics ────────────────────────────────────────────────────────

def test_ik_reachable_target():
    """IK should converge on a point we know the arm can reach (sample FK)."""
    q_true = np.radians([20, -30, 40, 0, 20, -10])
    target  = ee_position(q_true)

    q_sol, ok, err = ik_with_restarts(target, n_restarts=10)
    assert ok, f"IK failed to converge, err={err*1000:.1f} mm"
    assert err < 5e-3, f"IK error too large: {err*1000:.1f} mm"

    # Verify the solution actually reaches the target
    pos_check = ee_position(q_sol)
    np.testing.assert_allclose(pos_check, target, atol=5e-3)


def test_ik_joint_limits_respected():
    """IK solution must stay within joint limits."""
    q_true = np.radians([45, -20, 50, 30, -15, 20])
    target  = ee_position(q_true)
    q_sol, ok, err = ik_with_restarts(target, n_restarts=8)

    if ok:
        assert np.all(q_sol >= JOINT_LIMITS[:, 0] - 1e-6)
        assert np.all(q_sol <= JOINT_LIMITS[:, 1] + 1e-6)


def test_ik_home_position():
    """Home position EE should be solvable."""
    # FK at zero config
    T, _ = forward_kinematics(np.zeros(N_JOINTS))
    target = T[:3, 3]
    q_sol, ok, err = ik_with_restarts(target, n_restarts=5)
    assert ok or err < 0.01, f"IK for home position failed, err={err*1000:.1f} mm"


# ── Workspace ─────────────────────────────────────────────────────────────────

def test_workspace_sample_shape():
    from src import sample_workspace
    pts = sample_workspace(100)
    assert pts.shape == (100, 3)


def test_workspace_points_reachable():
    """All sampled workspace points must be within the arm's max reach."""
    from src import sample_workspace
    pts = sample_workspace(200)
    # Arm link lengths: 0 + 0.8 + 0.2 + 0 + 0 ≈ 1.0 m links + d offsets
    # Max theoretical reach well under 2.5 m
    norms = np.linalg.norm(pts, axis=1)
    assert np.all(norms < 2.5), f"Point outside expected reach: {norms.max():.3f} m"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
