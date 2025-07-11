import numpy as np

def leg_IK(x, y, z, L_haa=0.0955, L_thigh=0.213, L_shin=0.213):
    """
    Computes 3DOF inverse kinematics for Unitree Go2 leg.
    """
    q1 = np.arctan2(y, z)
    y_proj = np.sqrt(y ** 2 + z ** 2) - L_haa
    D = np.sqrt(x ** 2 + y_proj ** 2)
    cos_q3 = (L_thigh ** 2 + L_shin ** 2 - D ** 2) / (2 * L_thigh * L_shin)
    cos_q3 = np.clip(cos_q3, -1.0, 1.0)
    q3 = np.pi - np.arccos(cos_q3)
    alpha = np.arctan2(y_proj, x)
    cos_beta = (L_thigh ** 2 + D ** 2 - L_shin ** 2) / (2 * L_thigh * D)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)
    q2 = alpha - beta
    return q1, q2, q3

if __name__ == "__main__":
    # Target foot positions for all four legs (x, y, z)
    foot_targets = {
        "FL": [0.25, 0.05, -0.25],  # Front Left
        "FR": [0.25, -0.05, -0.25], # Front Right
        "RL": [-0.25, 0.05, -0.25], # Rear Left
        "RR": [-0.25, -0.05, -0.25],# Rear Right
    }

    print("Unitree Go2 Leg IK Results:")
    for leg_name, (x, y, z) in foot_targets.items():
        q1, q2, q3 = leg_IK(x, y, z)
        print(f"{leg_name} | HAA: {np.degrees(q1):6.2f}° | HFE: {np.degrees(q2):6.2f}° | KFE: {np.degrees(q3):6.2f}°")