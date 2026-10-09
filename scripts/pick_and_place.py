#!/usr/bin/env python3
"""
pick_and_place.py — Pick-and-Place Demo for 6-DOF Arm
IIT Bhilai | Ashish Devadas

Simulates a complete pick-and-place cycle:
  1. Home → approach above pick
  2. Approach → pick (descend)
  3. Pick → retreat (ascend)
  4. Retreat → approach above place
  5. Approach → place (descend)
  6. Place → home

Run:
    python3 scripts/pick_and_place.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src import (
    ik_with_restarts, forward_kinematics, ee_position,
    MultiSegmentPath, JointTrajectory,
    JOINT_LIMITS, N_JOINTS,
)

os.makedirs("media", exist_ok=True)

# ── Task definition ───────────────────────────────────────────────────────────
APPROACH_HEIGHT = 0.12   # m above pick/place

PICK_POS  = np.array([0.75,  0.20, 0.70])
PLACE_POS = np.array([0.60, -0.45, 0.70])
HOME_POS  = np.array([0.0,   0.0,  1.60])   # upright above base

PICK_APPROACH  = PICK_POS  + np.array([0, 0, APPROACH_HEIGHT])
PLACE_APPROACH = PLACE_POS + np.array([0, 0, APPROACH_HEIGHT])

waypoints = [
    HOME_POS,
    PICK_APPROACH,
    PICK_POS,
    PICK_APPROACH,
    PLACE_APPROACH,
    PLACE_POS,
    PLACE_APPROACH,
    HOME_POS,
]

durations = [2.5, 1.0, 1.0, 1.0, 2.5, 1.0, 1.0, 2.5]

segment_labels = [
    "Home → Pick approach",
    "Descend to pick",
    "Ascend from pick",
    "Pick approach → Place approach",
    "Descend to place",
    "Ascend from place",
    "Return to home",
]

# ── Solve IK for home config ──────────────────────────────────────────────────
print("Solving IK for home position...")
q_home, ok, err = ik_with_restarts(HOME_POS, n_restarts=10)
print(f"  Home IK: ok={ok}  err={err*1000:.1f} mm")

# ── Plan multi-segment path ───────────────────────────────────────────────────
print("Planning pick-and-place trajectory...")
path = MultiSegmentPath(waypoints, durations=durations, dt=0.02, q_init=q_home)

total_time = sum(durations)
n_pts = len(path.full_time)
print(f"  Total time: {total_time:.1f}s  |  {n_pts} waypoints")

for i, seg in enumerate(path.segments):
    sr = seg.success_rate() * 100
    print(f"  Seg {i+1} ({segment_labels[i]}): IK success {sr:.0f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
BG = "#0f1117"; TEXT_C = "#e2e8f0"; GRID_C = "#1e293b"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": GRID_C, "axes.labelcolor": TEXT_C,
    "axes.titlecolor": TEXT_C, "xtick.color": TEXT_C,
    "ytick.color": TEXT_C, "text.color": TEXT_C,
    "grid.color": GRID_C, "grid.linewidth": 0.6, "lines.linewidth": 1.8,
})

fig = plt.figure(figsize=(16, 10))
fig.suptitle("6-DOF Arm — Pick-and-Place Trajectory", fontsize=14)

# ── 3D path ──
ax3d = fig.add_subplot(2, 2, (1, 3), projection='3d')
ee_path = path.full_ee_positions

# Color by segment
seg_colors = ["#3b82f6", "#22c55e", "#f59e0b", "#a855f7",
              "#ef4444", "#06b6d4", "#84cc16", "#f43f5e"]
offset = 0
for i, seg in enumerate(path.segments):
    pts = seg.ee_positions
    n   = len(pts)
    ax3d.plot(pts[:, 0], pts[:, 1], pts[:, 2],
              color=seg_colors[i % len(seg_colors)], linewidth=2,
              label=segment_labels[i] if i < len(segment_labels) else f"Seg {i+1}")

# Mark pick and place
ax3d.scatter(*PICK_POS,  s=150, color="#ef4444", marker='D', zorder=6, label="Pick")
ax3d.scatter(*PLACE_POS, s=150, color="#22c55e", marker='s', zorder=6, label="Place")
ax3d.scatter(*HOME_POS,  s=120, color="#facc15", marker='^', zorder=6, label="Home")

ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
ax3d.set_xlim(-1.2, 1.2); ax3d.set_ylim(-1.2, 1.2); ax3d.set_zlim(0, 1.8)
ax3d.set_title("End-Effector Path", pad=8)
ax3d.legend(fontsize=7, loc='upper right')
ax3d.grid(True)
ax3d.xaxis.pane.fill = ax3d.yaxis.pane.fill = ax3d.zaxis.pane.fill = False

# ── Joint trajectories ──
ax_j = fig.add_subplot(2, 2, 2)
t_full = path.full_time
q_full = path.full_q_traj
joint_names = ["J1", "J2", "J3", "J4", "J5", "J6"]
cmap = plt.cm.tab10
for j in range(N_JOINTS):
    ax_j.plot(t_full, np.degrees(q_full[:, j]), color=cmap(j/N_JOINTS), label=joint_names[j])
ax_j.set_xlabel("Time (s)"); ax_j.set_ylabel("Joint Angle (°)")
ax_j.set_title("Joint Angles over Time")
ax_j.legend(fontsize=8, ncol=3); ax_j.grid(True)

# ── EE height profile ──
ax_z = fig.add_subplot(2, 2, 4)
ax_z.plot(t_full, ee_path[:, 2], color="#3b82f6", label="Z")
ax_z.plot(t_full, ee_path[:, 0], color="#ef4444", label="X")
ax_z.plot(t_full, ee_path[:, 1], color="#22c55e", label="Y")
ax_z.axhline(PICK_POS[2],  color="#ef4444", linestyle=':', linewidth=1, alpha=0.6)
ax_z.axhline(PLACE_POS[2], color="#22c55e", linestyle=':', linewidth=1, alpha=0.6)
ax_z.set_xlabel("Time (s)"); ax_z.set_ylabel("Position (m)")
ax_z.set_title("EE Position over Time")
ax_z.legend(fontsize=8); ax_z.grid(True)

plt.tight_layout()
plt.savefig("media/pick_and_place.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Saved media/pick_and_place.png")
