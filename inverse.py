#!/usr/bin/env python3
"""
3-DOF Robotic Arm Kinematics

This script provides forward and inverse kinematics functions for a
3-DOF robotic arm with three revolute joints (θ1, θ2, θ3) and link
offsets L1, L2, L3, L4. You can import these functions into your
simulation or run this file directly to test them.
"""

import numpy as np


def forward_kinematics(theta1, theta2, theta3, L1, L2, L3, L4):
    """
    Compute the end-effector position (x, y, z) of a 3-DOF arm.

    Parameters
    ----------
    theta1, theta2, theta3 : float
        Joint angles in radians.
    L1, L2, L3, L4 : float
        Link lengths as defined in your model.
        - L1: vertical offset from base to joint 2
        - L2: length of link 2
        - L3: length of link 3
        - L4: x-offset from link 3 to end-effector

    Returns
    -------
    x, y, z : float
        Cartesian coordinates of the end effector.
    """
    # Precompute sines and cosines
    c1, s1 = np.cos(theta1), np.sin(theta1)
    # Combined angle for joints 2 & 3
    phi23 = theta2 + theta3
    c23, s23 = np.cos(phi23), np.sin(phi23)

    # Compute planar projection (in the θ1=0 plane)
    plane_x = L2 * np.cos(theta2) + L3 * c23 + L4
    plane_z = L2 * np.sin(theta2) + L3 * s23

    # Rotate into world frame
    x = c1 * plane_x
    y = s1 * plane_x
    z = L1 + plane_z

    return x, y, z


def inverse_kinematics(x, y, z, L1, L2, L3, L4):
    """
    Solve for (θ1, θ2, θ3) given end-effector (x, y, z).

    Parameters
    ----------
    x, y, z : float
        Desired end-effector coordinates.
    L1, L2, L3, L4 : float
        Link lengths (same definitions as in forward_kinematics).

    Returns
    -------
    theta1, theta2, theta3 : float
        Joint angles in radians.
    """
    # Base rotation
    theta1 = np.arctan2(y, x)

    # Project into the plane of joints 2 & 3
    r = np.hypot(x, y) - L4
    s = z - L1

    # Distance from joint2 to the "wrist" point
    D = np.hypot(r, s)

    # Law of Cosines for theta3
    cos_q3 = (D**2 - L2**2 - L3**2) / (2 * L2 * L3)
    cos_q3 = np.clip(cos_q3, -1.0, 1.0)  # numerical safety
    theta3 = np.arccos(cos_q3)

    # Compute intermediate angles for theta2
    phi = np.arctan2(s, r)
    psi = np.arctan2(L3 * np.sin(theta3), L2 + L3 * np.cos(theta3))

    theta2 = phi - psi

    return theta1, theta2, theta3


def main():
    # Example link lengths
    L1, L2, L3, L4 = 0.067, 0.213, 0.210, 0.094 

    # Example joint angles
    t1, t2, t3 = 0.2, -0.5, 0.8

    # Forward kinematics
    x, y, z = forward_kinematics(t1, t2, t3, L1, L2, L3, L4)
    print(f"Forward Kinematics → x: {x:.4f}, y: {y:.4f}, z: {z:.4f}")

    # Inverse kinematics (should recover the original angles)
    q1, q2, q3 = inverse_kinematics(x, y, z, L1, L2, L3, L4)
    print(f"Inverse Kinematics → θ1: {q1:.4f}, θ2: {q2:.4f}, θ3: {q3:.4f}")


if __name__ == "__main__":
    main()
