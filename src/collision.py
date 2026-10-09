"""
collision.py — Obstacle Avoidance for 6-DOF Robotic Arm
IIT Bhilai | Ashish Devadas

Implements:
  - Sphere and box obstacle primitives
  - Link capsule collision model
  - Potential field obstacle avoidance (repulsive gradient)
  - Configuration-space validity checking
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .dh_kinematics import forward_kinematics, jacobian, JOINT_LIMITS, N_JOINTS


# ── Obstacle primitives ───────────────────────────────────────────────────────
@dataclass
class SphereObstacle:
    center: np.ndarray   # [x, y, z] (m)
    radius: float        # (m)
    name:   str = "sphere"

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)

    def distance_to_point(self, p: np.ndarray) -> float:
        return float(np.linalg.norm(p - self.center) - self.radius)

    def closest_point(self, p: np.ndarray) -> np.ndarray:
        d = p - self.center
        n = np.linalg.norm(d)
        if n < 1e-9:
            return self.center + np.array([self.radius, 0, 0])
        return self.center + (d / n) * self.radius


@dataclass
class BoxObstacle:
    center:     np.ndarray   # [x, y, z] (m)
    half_size:  np.ndarray   # [hx, hy, hz] (m)
    name:       str = "box"

    def __post_init__(self):
        self.center    = np.asarray(self.center,    dtype=float)
        self.half_size = np.asarray(self.half_size, dtype=float)

    def distance_to_point(self, p: np.ndarray) -> float:
        """Signed distance — negative inside."""
        d = np.abs(p - self.center) - self.half_size
        return float(np.linalg.norm(np.maximum(d, 0)) + min(np.max(d), 0))

    def closest_point(self, p: np.ndarray) -> np.ndarray:
        lo = self.center - self.half_size
        hi = self.center + self.half_size
        return np.clip(p, lo, hi)


# ── Link capsule model ────────────────────────────────────────────────────────
def get_link_capsules(q: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """
    Model each arm link as a capsule (line segment + radius).
    Returns list of (p_start, p_end, radius) tuples in base frame.
    """
    _, frames = forward_kinematics(q)
    base = np.zeros(3)
    origins = [base] + [f[:3, 3] for f in frames]

    # Link radii (rough estimate for arm structural members)
    radii = [0.08, 0.07, 0.06, 0.05, 0.04, 0.04]

    capsules = []
    for i in range(N_JOINTS):
        capsules.append((origins[i], origins[i+1], radii[i]))
    return capsules


def segment_to_point_distance(p1: np.ndarray, p2: np.ndarray, p: np.ndarray) -> float:
    """Minimum distance from point p to line segment p1–p2."""
    d = p2 - p1
    t = np.dot(p - p1, d) / (np.dot(d, d) + 1e-12)
    t = np.clip(t, 0, 1)
    closest = p1 + t * d
    return float(np.linalg.norm(p - closest))


def segment_to_sphere_distance(
    p1: np.ndarray, p2: np.ndarray, center: np.ndarray, radius: float
) -> float:
    """Distance from segment p1–p2 to sphere surface (negative if penetrating)."""
    return segment_to_point_distance(p1, p2, center) - radius


# ── Collision checker ─────────────────────────────────────────────────────────
class CollisionChecker:
    """
    Check arm configurations for collision with a scene of obstacles.
    Uses link capsule model for arm geometry.
    """

    def __init__(self, obstacles: list, safety_margin: float = 0.05):
        """
        Args:
            obstacles:     list of SphereObstacle / BoxObstacle
            safety_margin: minimum clearance (m)
        """
        self.obstacles     = obstacles
        self.safety_margin = safety_margin

    def is_valid(self, q: np.ndarray) -> bool:
        """Return True if configuration q is collision-free."""
        if np.any(q < JOINT_LIMITS[:, 0]) or np.any(q > JOINT_LIMITS[:, 1]):
            return False

        capsules = get_link_capsules(q)
        for (p1, p2, r_link) in capsules:
            for obs in self.obstacles:
                if isinstance(obs, SphereObstacle):
                    d = segment_to_sphere_distance(p1, p2, obs.center, obs.radius)
                elif isinstance(obs, BoxObstacle):
                    # Sample along capsule and check each point
                    pts = np.linspace(p1, p2, 8)
                    d = min(obs.distance_to_point(pt) for pt in pts)
                else:
                    continue
                if d < r_link + self.safety_margin:
                    return False
        return True

    def min_clearance(self, q: np.ndarray) -> float:
        """Return minimum clearance to all obstacles (m). Negative = collision."""
        capsules = get_link_capsules(q)
        min_d = np.inf
        for (p1, p2, r_link) in capsules:
            for obs in self.obstacles:
                if isinstance(obs, SphereObstacle):
                    d = segment_to_sphere_distance(p1, p2, obs.center, obs.radius) - r_link
                elif isinstance(obs, BoxObstacle):
                    pts = np.linspace(p1, p2, 8)
                    d = min(obs.distance_to_point(pt) for pt in pts) - r_link
                else:
                    continue
                min_d = min(min_d, d)
        return float(min_d)


# ── Potential field obstacle avoidance ───────────────────────────────────────
class PotentialFieldPlanner:
    """
    Artificial Potential Field planner in joint space.

    Attractive potential: F_att = k_att * (q_goal - q)
    Repulsive potential:  F_rep = k_rep * (1/d - 1/d0) * (1/d²) * grad_d
      where d = min clearance, d0 = influence radius

    Suitable for local, short-range motion. Not globally complete.
    """

    def __init__(
        self,
        checker:        CollisionChecker,
        k_att:          float = 1.0,
        k_rep:          float = 0.5,
        influence_dist: float = 0.3,
        step_size:      float = 0.02,
        max_steps:      int   = 2000,
        goal_tol:       float = 0.05,
    ):
        self.checker        = checker
        self.k_att          = k_att
        self.k_rep          = k_rep
        self.influence_dist = influence_dist
        self.step_size      = step_size
        self.max_steps      = max_steps
        self.goal_tol       = goal_tol

    def _repulsive_gradient(self, q: np.ndarray) -> np.ndarray:
        """Numerical gradient of repulsive potential in joint space."""
        grad = np.zeros(N_JOINTS)
        d0   = self.influence_dist
        d    = self.checker.min_clearance(q)

        if d >= d0 or d < 1e-6:
            return grad

        eps = 1e-3
        for i in range(N_JOINTS):
            dq = np.zeros(N_JOINTS)
            dq[i] = eps
            d_plus  = self.checker.min_clearance(q + dq)
            d_minus = self.checker.min_clearance(q - dq)
            dd_dqi  = (d_plus - d_minus) / (2 * eps)

            # Repulsive force magnitude
            f_rep = self.k_rep * (1/d - 1/d0) * (1/d**2)
            grad[i] = -f_rep * dd_dqi

        return grad

    def plan(
        self,
        q_start: np.ndarray,
        q_goal:  np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """
        Run potential field planning from q_start to q_goal.

        Returns:
            path:    (N × 6) joint trajectory
            success: reached goal within tolerance
        """
        q     = q_start.copy()
        path  = [q.copy()]
        stuck = 0

        for step in range(self.max_steps):
            # Attractive force toward goal
            f_att = self.k_att * (q_goal - q)
            f_att_norm = np.linalg.norm(f_att)
            if f_att_norm > 1.0:
                f_att /= f_att_norm

            # Repulsive force from obstacles
            f_rep = self._repulsive_gradient(q)

            # Total force → step
            f_total = f_att + f_rep
            q_new   = q + self.step_size * f_total
            q_new   = np.clip(q_new, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

            if not self.checker.is_valid(q_new):
                # Take smaller step or perturb
                q_new = q + 0.1 * self.step_size * f_total
                q_new = np.clip(q_new, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
                if not self.checker.is_valid(q_new):
                    stuck += 1
                    if stuck > 50:
                        break
                    continue

            stuck = 0
            q = q_new
            path.append(q.copy())

            if np.linalg.norm(q - q_goal) < self.goal_tol:
                return np.array(path), True

        return np.array(path), False


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    obs = [
        SphereObstacle(center=[0.7, 0.15, 0.9], radius=0.18, name="sphere_1"),
        BoxObstacle(center=[0.4, -0.2, 0.8], half_size=[0.12, 0.12, 0.15], name="box_1"),
    ]
    checker = CollisionChecker(obs, safety_margin=0.04)

    q_test = np.zeros(6)
    print("Config valid:", checker.is_valid(q_test))
    print("Min clearance:", f"{checker.min_clearance(q_test)*100:.1f} cm")

    q_goal = np.radians([50, -20, 40, 0, 25, -30])
    planner = PotentialFieldPlanner(checker, k_att=1.5, k_rep=0.3,
                                    influence_dist=0.4, step_size=0.03, max_steps=2000,
                                    goal_tol=0.08)
    path, ok = planner.plan(q_test, q_goal)
    print(f"\nPotential field path: {len(path)} steps  success={ok}")
