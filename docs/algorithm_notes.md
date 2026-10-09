# Algorithm Notes — 6-DOF Robotic Arm

**IIT Bhilai | Ashish Devadas**

---

## Modified Craig DH Convention

Each joint's homogeneous transform is computed as:

```
T_i = Rx(alpha_{i}) * Tx(a_{i}) * Rz(theta_i + offset_i) * Tz(d_i)
```

The overall end-effector transform:

```
T_ee = T_1 * T_2 * T_3 * T_4 * T_5 * T_6
```

---

## Geometric Jacobian

For a revolute joint i, the joint axis in world frame passes through the intermediate frame **after** applying Rx(α)·Tx(a) but **before** Rz(θ)·Tz(d):

```
T_pre_i = T_{i-1} * Rx(alpha_i) * Tx(a_i)

z_i     = T_pre_i[:3, 2]    # joint axis in world frame
p_i     = T_pre_i[:3, 3]    # joint origin in world frame

J_v[:,i] = z_i × (p_ee - p_i)   # linear velocity contribution
J_w[:,i] = z_i                   # angular velocity contribution
```

---

## Damped Least Squares IK

Iterative solver using the DLS / Levenberg–Marquardt update:

```
e      = target_pos - current_pos      # 3-vector position error
J      = J_v (3×6 linear Jacobian)
dq     = J^T (J J^T + λ²I)^{-1} e
q_new  = q + α * dq                    # α = step_size, λ = damping
```

- **λ (damping):** 0.05 — prevents instability near singularities
- **α (step):** 0.8 — controls convergence rate  
- **Restarts:** 10 random restarts improve global coverage

---

## Trajectory Time Scaling

All profiles map elapsed time `t ∈ [0,T]` to a normalized progress `s ∈ [0,1]`:

| Profile | s(t) | Boundary conditions |
|---------|------|---------------------|
| Cubic | 3τ² - 2τ³ | s=0 at t=0, s=1 at t=T, ṡ=0 at both ends |
| Quintic | 10τ³ - 15τ⁴ + 6τ⁵ | + s̈=0 at both ends |
| Trapezoidal | piecewise linear velocity | constant accel/decel phases |

Joint motion: `q(t) = q_start + s(t) * (q_end - q_start)`

---

## Cartesian Trajectory (Linear Interpolation)

Straight-line path between two EE positions, with IK solved at each timestep:

```
pos(t) = pos_start + s(t) * (pos_end - pos_start)
q(t)   = IK(pos(t), q_seed = q(t-1))      # warm-started
```

Note: straight Cartesian paths can pass through workspace boundary regions,
causing IK failure for some intermediate points. This is normal; the trajectory
handles it by keeping the previous valid joint config.

---

## Potential Field Obstacle Avoidance

Configuration-space planner with attractive and repulsive forces:

```python
F_att = k_att * (q_goal - q)                           # toward goal
F_rep = k_rep * (1/d - 1/d0) * (1/d²) * ∇d  if d < d0, else 0
F_tot = F_att + F_rep
q_new = q + step * F_tot
```

The repulsive gradient ∇d is computed numerically by finite differences
of the minimum capsule–obstacle clearance.

**Limitations:** Local planner — can get trapped in local minima.
For complex environments, global planners (RRT, PRM) are preferred.

---

## Link Capsule Collision Model

Each arm link is approximated as a capsule (infinite cylinder capped at both ends):
- **Capsule i:** line segment `[origin_i, origin_{i+1}]` with radius `r_i`
- Collision test: minimum distance from capsule axis to obstacle < r_link + safety_margin

```
r = [0.08, 0.07, 0.06, 0.05, 0.04, 0.04]  # m, J1..J6
```
