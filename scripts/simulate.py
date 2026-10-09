#!/usr/bin/env python3
"""
simulate.py — Full 6-DOF Arm Simulation with Visualization
IIT Bhilai | Ashish Devadas

Run:
    python3 scripts/simulate.py

Generates:
    media/arm_fk.png           — home config + FK joint frames
    media/workspace.png        — 3D reachable workspace cloud
    media/trajectory.png       — joint + Cartesian trajectory plots
    media/obstacle_avoidance.png — potential field path around obstacles
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from src import (
    forward_kinematics, inverse_kinematics, ik_with_restarts,
    jacobian, ee_position, sample_workspace,
    JointTrajectory, CartesianTrajectory, MultiSegmentPath,
    SphereObstacle, BoxObstacle, CollisionChecker, PotentialFieldPlanner,
    JOINT_LIMITS, N_JOINTS,
)

os.makedirs("media", exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────────────
C_LINK   = "#2563eb"
C_JOINT  = "#ef4444"
C_EE     = "#16a34a"
C_TRAJ   = "#7c3aed"
C_OBS    = "#f97316"
C_PATH   = "#0891b2"
BG       = "#0f1117"
GRID_C   = "#1e293b"
TEXT_C   = "#e2e8f0"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    GRID_C,
    "axes.labelcolor":   TEXT_C,
    "axes.titlecolor":   TEXT_C,
    "xtick.color":       TEXT_C,
    "ytick.color":       TEXT_C,
    "text.color":        TEXT_C,
    "grid.color":        GRID_C,
    "grid.linewidth":    0.6,
    "lines.linewidth":   2.0,
    "font.family":       "sans-serif",
})


def draw_arm(ax, q, color=C_LINK, label_joints=True, alpha=1.0):
    """Draw the arm links and joints in 3D."""
    _, frames = forward_kinematics(q)
    origins = [np.zeros(3)] + [f[:3, 3] for f in frames]

    for i in range(len(origins) - 1):
        xs = [origins[i][0], origins[i+1][0]]
        ys = [origins[i][1], origins[i+1][1]]
        zs = [origins[i][2], origins[i+1][2]]
        ax.plot(xs, ys, zs, color=color, linewidth=3, alpha=alpha)

    for i, o in enumerate(origins[:-1]):
        ax.scatter(*o, s=60, color=C_JOINT, zorder=5, alpha=alpha)

    # EE
    ax.scatter(*origins[-1], s=120, color=C_EE, marker='*', zorder=6, alpha=alpha)

    # EE frame axes
    T_ee = frames[-1]
    ax.quiver(*T_ee[:3, 3], *T_ee[:3, 0]*0.08, color='red',   linewidth=1.5, alpha=alpha)
    ax.quiver(*T_ee[:3, 3], *T_ee[:3, 1]*0.08, color='lime',  linewidth=1.5, alpha=alpha)
    ax.quiver(*T_ee[:3, 3], *T_ee[:3, 2]*0.08, color='cyan',  linewidth=1.5, alpha=alpha)


def set_ax3d(ax, title="", xlim=(-1.5,1.5), ylim=(-1.5,1.5), zlim=(0,1.8)):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(title, pad=10)
    ax.grid(True)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False


# ── 1. Forward Kinematics Visualisation ──────────────────────────────────────
print("1/4  Forward kinematics...")
fig = plt.figure(figsize=(14, 6))
fig.suptitle("6-DOF Robotic Arm — Forward Kinematics", fontsize=14, y=0.98)

configs = {
    "Home  q = [0, 0, 0, 0, 0, 0]":          np.zeros(6),
    "Config q = [30°, -45°, 60°, 0°, 30°, 0°]": np.radians([30, -45, 60, 0, 30, 0]),
    "Config q = [-60°, 20°, -30°, 45°, -20°, 60°]": np.radians([-60, 20, -30, 45, -20, 60]),
}

for idx, (title, q) in enumerate(configs.items()):
    ax = fig.add_subplot(1, 3, idx+1, projection='3d')
    draw_arm(ax, q)
    T, _ = forward_kinematics(q)
    pos  = T[:3, 3]
    ax.text2D(0.05, 0.92, f"EE: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m",
              transform=ax.transAxes, fontsize=7, color=TEXT_C)
    set_ax3d(ax, title, xlim=(-1.6,1.6), ylim=(-1.6,1.6), zlim=(0,2.0))

plt.tight_layout()
plt.savefig("media/arm_fk.png", dpi=150, bbox_inches='tight')
plt.close()
print("    → media/arm_fk.png")


# ── 2. Workspace Cloud ────────────────────────────────────────────────────────
print("2/4  Sampling workspace (5000 pts)...")
pts = sample_workspace(5000)

fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
           c=pts[:, 2], cmap='plasma', s=1.5, alpha=0.4)
ax.set_title("Reachable Workspace — 5000 random configs", pad=10)
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
ax.grid(True)
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False

max_reach = np.max(np.linalg.norm(pts, axis=1))
ax.text2D(0.05, 0.95, f"Max reach: {max_reach:.3f} m", transform=ax.transAxes,
          fontsize=10, color=TEXT_C)

plt.savefig("media/workspace.png", dpi=150, bbox_inches='tight')
plt.close()
print("    → media/workspace.png")


# ── 3. Trajectory Planning ────────────────────────────────────────────────────
print("3/4  Trajectory planning...")
q_start = np.radians([0,   0,  0, 0,  0,  0])
q_end   = np.radians([60, -30, 45, 0, 30, -45])

traj_cubic    = JointTrajectory(q_start, q_end, 4.0, 0.02, 'cubic')
traj_quintic  = JointTrajectory(q_start, q_end, 4.0, 0.02, 'quintic')
traj_trapezoid= JointTrajectory(q_start, q_end, 4.0, 0.02, 'trapezoid')

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle("Trajectory Planning — Joint Space (Profiles Comparison)", fontsize=13)

joint_names = ["J1 Waist", "J2 Shoulder", "J3 Elbow", "J4 Wrist-Roll", "J5 Wrist-Pitch", "J6 Wrist-Yaw"]
colors_traj = ["#3b82f6", "#22c55e", "#f59e0b"]
labels = ["Cubic", "Quintic", "Trapezoidal"]

for j in range(6):
    ax = axes[j // 2][j % 2]
    for traj, c, lbl in zip([traj_cubic, traj_quintic, traj_trapezoid], colors_traj, labels):
        ax.plot(traj.t, np.degrees(traj.q[:, j]), color=c, label=lbl)
    ax.set_title(joint_names[j], fontsize=9)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (°)")
    ax.legend(fontsize=7); ax.grid(True)

plt.tight_layout()
plt.savefig("media/trajectory.png", dpi=150, bbox_inches='tight')
plt.close()
print("    → media/trajectory.png")


# ── 4. Obstacle Avoidance ─────────────────────────────────────────────────────
print("4/4  Obstacle avoidance (potential field)...")
obstacles = [
    SphereObstacle(center=[0.7, 0.15, 0.9], radius=0.18, name="sphere"),
    BoxObstacle(center=[0.4, -0.2, 0.8], half_size=[0.12, 0.12, 0.15], name="box"),
]
checker = CollisionChecker(obstacles, safety_margin=0.04)
planner = PotentialFieldPlanner(checker, k_att=1.5, k_rep=0.3,
                                 influence_dist=0.4, step_size=0.03, max_steps=2000,
                                 goal_tol=0.08)

q_s = np.radians([0, 0, 0, 0, 0, 0])
q_g = np.radians([50, -20, 40, 0, 25, -30])
path, success = planner.plan(q_s, q_g)
print(f"    Path steps: {len(path)}  success: {success}")

fig = plt.figure(figsize=(12, 8))
ax  = fig.add_subplot(111, projection='3d')
ax.set_title(f"Obstacle Avoidance — Potential Field  (success={success})", pad=10)

# Draw obstacles
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
for obs in obstacles:
    if isinstance(obs, SphereObstacle):
        xs = obs.center[0] + obs.radius * np.cos(u) * np.sin(v)
        ys = obs.center[1] + obs.radius * np.sin(u) * np.sin(v)
        zs = obs.center[2] + obs.radius * np.cos(v)
        ax.plot_surface(xs, ys, zs, alpha=0.25, color=C_OBS)

# Draw start and goal arm
draw_arm(ax, q_s, color="#3b82f6", alpha=0.6)
draw_arm(ax, q_g, color="#22c55e", alpha=0.6)

# Draw EE path
ee_pts = np.array([ee_position(q) for q in path[::5]])
ax.plot(ee_pts[:, 0], ee_pts[:, 1], ee_pts[:, 2],
        color=C_PATH, linewidth=2, linestyle='--', label="EE path")

# Draw a few intermediate arm configs
for qi in path[::len(path)//5 or 1]:
    draw_arm(ax, qi, color="#94a3b8", alpha=0.2)

ax.legend()
set_ax3d(ax, xlim=(-1.5,1.5), ylim=(-1.5,1.5), zlim=(0,1.8))
plt.savefig("media/obstacle_avoidance.png", dpi=150, bbox_inches='tight')
plt.close()
print("    → media/obstacle_avoidance.png")

print("\n✓ All plots saved to media/")
print(f"  Max arm reach: {np.max(np.linalg.norm(pts, axis=1)):.3f} m")
q_home = np.zeros(6)
T_home, _ = forward_kinematics(q_home)
print(f"  Home EE position: {T_home[:3,3].round(3)} m")
