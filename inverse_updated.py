import numpy as np

def forward_kinematics(theta1, theta2, theta3, L1, L2, L3, L4):
    """
    Compute the forward kinematics of a 3-DOF arm with an extra translation (L4).
    All angles in radians, lengths in consistent units.
    Returns (x, y, z).
    """
    # Precompute sines/cosines
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c2, s2 = np.cos(theta2), np.sin(theta2)
    c3, s3 = np.cos(theta3), np.sin(theta3)
    # φ = θ2 + θ3
    c23, s23 = np.cos(theta2 + theta3), np.sin(theta2 + theta3)
    
    # Using your original transformation chain:
    # T01 * T12 * T23 * T34, then multiply by [0,0,0,1]^T.
    x = c1*(L4 + L3*c23) + s1*(L1 + L2*s2 + L3*s23)
    y = s1*(L4 + L3*c23) - c1*(L1 + L2*s2 + L3*s23)
    z = L2*c2 + L3*c23 + L1
    return x, y, z

def inverse_kinematics(x, y, z, L1, L2, L3, L4):
    """
    Analytic inverse kinematics for the above arm.
    Returns (theta1, theta2, theta3).
    May have two solutions for elbow-up/elbow-down (we return the 'elbow-down' by default).
    """
    # θ1 from projection onto XY-plane
    theta1 = np.arctan2(y, x)
    
    # Compute wrist position vector in the plane of θ2/θ3:
    # subtract the offset L4 along the rotated x-y
    r = np.hypot(x, y) - L4
    s = z - L1
    
    # Law of cosines for θ3:
    D = (r**2 + s**2 - L2**2 - L3**2) / (2 * L2 * L3)
    # clamp for numerical safety
    D = np.clip(D, -1.0, 1.0)
    # elbow-down:
    theta3 = np.arctan2(-np.sqrt(1 - D**2), D)
    # (if you want the other elbow-up solution, use +sqrt)
    # theta3_alt = np.arctan2( np.sqrt(1 - D**2), D)
    
    # θ2 from geometry
    phi = np.arctan2(s, r)
    psi = np.arctan2(L3 * np.sin(theta3), L2 + L3 * np.cos(theta3))
    theta2 = phi - psi
    
    return theta1, theta2, theta3

# Example usage:
if __name__ == "__main__":
    # link lengths
    L1, L2, L3, L4 = 0.067, 0.213, 0.210, 0.094  
    # pick some joint angles
    t1, t2, t3 = 0.2, 0.5, -0.3
    
    # FK
    x, y, z = forward_kinematics(t1, t2, t3, L1, L2, L3, L4)
    print(f"FK → x={x:.3f}, y={y:.3f}, z={z:.3f}")
    
    # IK (should recover the original angles)
    sol1, sol2, sol3 = inverse_kinematics(x, y, z, L1, L2, L3, L4)
    print(f"IK → θ1={sol1:.3f}, θ2={sol2:.3f}, θ3={sol3:.3f}")
