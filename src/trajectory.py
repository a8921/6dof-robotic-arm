"""
trajectory.py — Trajectory Planning for 6-DOF Robotic Arm
IIT Bhilai | Ashish Devadas

Provides:
  - Joint-space trajectories: cubic & quintic splines, trapezoidal velocity
  - Cartesian-space trajectories: linear interpolation with IK at each waypoint
  - Multi-segment path planning
  - Velocity / acceleration profile generation
"""

import numpy as np
from typing import Optional
from .dh_kinematics import (
    inverse_kinematics, ik_with_restarts, forward_kinematics,
    ee_position, JOINT_LIMITS, N_JOINTS
)


# ── Time scaling profiles ─────────────────────────────────────────────────────
def cubic_time_scale(t: np.ndarray, T: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cubic (3-4-5 polynomial) time scaling: s(t), sd(t), sdd(t).
    Boundary conditions: s(0)=0, s(T)=1, sd(0)=sd(T)=0.
    """
    tau = np.clip(t / T, 0, 1)
    s   =  3*tau**2 - 2*tau**3
    sd  = (6*tau   - 6*tau**2) / T
    sdd = (6       -12*tau   ) / T**2
    return s, sd, sdd


def quintic_time_scale(t: np.ndarray, T: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quintic time scaling — zero vel/accel at both endpoints.
    """
    tau = np.clip(t / T, 0, 1)
    s   =  10*tau**3 - 15*tau**4 +  6*tau**5
    sd  = (30*tau**2 - 60*tau**3 + 30*tau**4) / T
    sdd = (60*tau    -180*tau**2 +120*tau**3) / T**2
    return s, sd, sdd


def trapezoidal_time_scale(
    t: np.ndarray, T: float, accel_frac: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trapezoidal velocity profile.
    accel_frac: fraction of T spent accelerating (and decelerating).
    """
    ta = accel_frac * T       # acceleration phase duration
    s  = np.zeros_like(t)
    sd = np.zeros_like(t)
    sdd = np.zeros_like(t)

    a = 1.0 / (T * ta - ta**2)   # constant acceleration

    for i, ti in enumerate(t):
        if ti < 0:
            pass
        elif ti <= ta:
            s[i]   = 0.5 * a * ti**2
            sd[i]  = a * ti
            sdd[i] = a
        elif ti <= T - ta:
            s[i]   = a * ta * (ti - ta/2)
            sd[i]  = a * ta
            sdd[i] = 0
        elif ti <= T:
            dt     = T - ti
            s[i]   = 1 - 0.5 * a * dt**2
            sd[i]  = a * dt
            sdd[i] = -a
        else:
            s[i] = 1.0

    return s, sd, sdd


# ── Joint-space trajectory ─────────────────────────────────────────────────────
class JointTrajectory:
    """
    Plan a smooth joint-space trajectory between two configurations.

    Args:
        q_start:   start joint angles [6] (rad)
        q_end:     end   joint angles [6] (rad)
        duration:  motion duration (s)
        dt:        time step (s)
        profile:   'cubic' | 'quintic' | 'trapezoid'
    """

    def __init__(
        self,
        q_start:  np.ndarray,
        q_end:    np.ndarray,
        duration: float = 3.0,
        dt:       float = 0.02,
        profile:  str   = 'quintic',
    ):
        self.q_start  = np.asarray(q_start, dtype=float)
        self.q_end    = np.asarray(q_end,   dtype=float)
        self.duration = duration
        self.dt       = dt
        self.profile  = profile

        self.t = np.arange(0, duration + dt, dt)
        self._compute()

    def _compute(self):
        T = self.duration
        if self.profile == 'cubic':
            s, sd, sdd = cubic_time_scale(self.t, T)
        elif self.profile == 'trapezoid':
            s, sd, sdd = trapezoidal_time_scale(self.t, T)
        else:
            s, sd, sdd = quintic_time_scale(self.t, T)

        dq = self.q_end - self.q_start
        self.q   = self.q_start + np.outer(s,   dq)   # (N, 6)
        self.qd  = np.outer(sd,  dq)                   # (N, 6) joint vel
        self.qdd = np.outer(sdd, dq)                   # (N, 6) joint accel

    @property
    def positions(self) -> np.ndarray:
        return self.q

    @property
    def velocities(self) -> np.ndarray:
        return self.qd

    @property
    def accelerations(self) -> np.ndarray:
        return self.qdd

    def ee_path(self) -> np.ndarray:
        """Compute EE Cartesian path (N×3)."""
        return np.array([ee_position(q) for q in self.q])

    def max_joint_velocity(self) -> np.ndarray:
        """Max velocity per joint (rad/s)."""
        return np.max(np.abs(self.qd), axis=0)

    def max_joint_acceleration(self) -> np.ndarray:
        """Max acceleration per joint (rad/s²)."""
        return np.max(np.abs(self.qdd), axis=0)


# ── Cartesian-space (linear) trajectory ───────────────────────────────────────
class CartesianTrajectory:
    """
    Straight-line Cartesian trajectory between two EE poses.
    IK solved at each time step (position-only by default).

    Args:
        pos_start:  start EE position [3] (m)
        pos_end:    end   EE position [3] (m)
        rot_start:  start EE rotation matrix [3,3] (None = no orientation)
        rot_end:    end   EE rotation matrix [3,3]
        duration:   motion duration (s)
        dt:         time step (s)
        q_init:     initial joint config for IK seeding [6]
        profile:    time scaling profile
    """

    def __init__(
        self,
        pos_start:  np.ndarray,
        pos_end:    np.ndarray,
        rot_start:  Optional[np.ndarray] = None,
        rot_end:    Optional[np.ndarray] = None,
        duration:   float = 3.0,
        dt:         float = 0.02,
        q_init:     Optional[np.ndarray] = None,
        profile:    str   = 'quintic',
    ):
        self.pos_start = np.asarray(pos_start, dtype=float)
        self.pos_end   = np.asarray(pos_end,   dtype=float)
        self.rot_start = rot_start
        self.rot_end   = rot_end
        self.duration  = duration
        self.dt        = dt
        self.q_init    = q_init if q_init is not None else np.zeros(N_JOINTS)
        self.profile   = profile

        self.t = np.arange(0, duration + dt, dt)
        self._compute()

    def _compute(self):
        T = self.duration
        if self.profile == 'cubic':
            s, _, _ = cubic_time_scale(self.t, T)
        elif self.profile == 'trapezoid':
            s, _, _ = trapezoidal_time_scale(self.t, T)
        else:
            s, _, _ = quintic_time_scale(self.t, T)

        n = len(self.t)
        self.ee_positions = np.zeros((n, 3))
        self.q_traj       = np.zeros((n, N_JOINTS))
        self.ik_success   = np.ones(n, dtype=bool)

        q_prev = self.q_init.copy()

        for i, si in enumerate(s):
            # Interpolate position
            pos_i = self.pos_start + si * (self.pos_end - self.pos_start)
            self.ee_positions[i] = pos_i

            # Interpolate orientation (SLERP-like via weighted rotation)
            rot_i = None
            if self.rot_start is not None and self.rot_end is not None:
                # Simple linear interpolation of rotation (not SLERP but sufficient for small rotations)
                rot_i = (1 - si) * self.rot_start + si * self.rot_end
                # Re-orthogonalise via SVD
                U, _, Vt = np.linalg.svd(rot_i)
                rot_i = U @ Vt

            q_i, ok, _ = inverse_kinematics(
                pos_i, rot_i, q_init=q_prev, max_iter=200, damping=0.05
            )
            self.q_traj[i]     = q_i
            self.ik_success[i] = ok
            q_prev = q_i

    @property
    def positions(self) -> np.ndarray:
        return self.q_traj

    def success_rate(self) -> float:
        return self.ik_success.mean()


# ── Multi-segment path ─────────────────────────────────────────────────────────
class MultiSegmentPath:
    """
    Chain multiple Cartesian waypoints into a smooth path.
    Each segment is a separate CartesianTrajectory.

    Args:
        waypoints: list of EE positions [[x,y,z], ...]
        durations: time for each segment (s), or single float for all
        dt:        time step (s)
        q_init:    initial joint config
    """

    def __init__(
        self,
        waypoints: list,
        durations: float | list = 2.0,
        dt:        float = 0.02,
        q_init:    Optional[np.ndarray] = None,
    ):
        self.waypoints = [np.asarray(w, dtype=float) for w in waypoints]
        n_seg = len(waypoints) - 1

        if isinstance(durations, (int, float)):
            self.durations = [float(durations)] * n_seg
        else:
            self.durations = list(durations)

        self.dt     = dt
        self.q_init = q_init if q_init is not None else np.zeros(N_JOINTS)

        self.segments: list[CartesianTrajectory] = []
        self._plan()

    def _plan(self):
        q_cur = self.q_init.copy()
        for i in range(len(self.waypoints) - 1):
            seg = CartesianTrajectory(
                pos_start = self.waypoints[i],
                pos_end   = self.waypoints[i + 1],
                duration  = self.durations[i],
                dt        = self.dt,
                q_init    = q_cur,
            )
            self.segments.append(seg)
            q_cur = seg.q_traj[-1]

    @property
    def full_q_traj(self) -> np.ndarray:
        """Concatenated joint trajectory across all segments."""
        return np.vstack([s.q_traj for s in self.segments])

    @property
    def full_ee_positions(self) -> np.ndarray:
        """Concatenated EE positions across all segments."""
        return np.vstack([s.ee_positions for s in self.segments])

    @property
    def full_time(self) -> np.ndarray:
        """Absolute time vector."""
        chunks = []
        offset = 0.0
        for seg in self.segments:
            chunks.append(seg.t + offset)
            offset += seg.duration
        return np.concatenate(chunks)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from src.dh_kinematics import forward_kinematics
    import numpy as np

    q0 = np.zeros(6)
    q1 = np.array([0.5, -0.3, 0.4, 0.0, 0.2, -0.5])

    traj = JointTrajectory(q0, q1, duration=3.0, dt=0.02, profile='quintic')
    print(f"Joint trajectory: {len(traj.t)} steps  ({traj.duration}s @ {1/traj.dt:.0f}Hz)")
    print(f"Max joint vel (rad/s): {traj.max_joint_velocity().round(3)}")
    print(f"Max joint accel (rad/s²): {traj.max_joint_acceleration().round(3)}")

    p0 = np.array([0.8, 0.0, 0.8])
    p1 = np.array([0.8, 0.5, 0.5])
    ctraj = CartesianTrajectory(p0, p1, duration=3.0, dt=0.05)
    print(f"\nCartesian trajectory IK success rate: {ctraj.success_rate()*100:.1f}%")
