"""
6-DOF Robotic Arm Simulation — IIT Bhilai
Ashish Devadas | Project Engineer, Dept. of Mechanical Engineering
"""
from .dh_kinematics import (
    forward_kinematics, inverse_kinematics, ik_with_restarts,
    jacobian, ee_position, ee_pose, sample_workspace,
    ARM_DH, JOINT_LIMITS, N_JOINTS,
)
from .trajectory import JointTrajectory, CartesianTrajectory, MultiSegmentPath
from .collision import (
    SphereObstacle, BoxObstacle, CollisionChecker, PotentialFieldPlanner,
)

__version__ = "1.0.0"
__author__  = "Ashish Devadas"
__affil__   = "IIT Bhilai — Department of Mechanical Engineering"
