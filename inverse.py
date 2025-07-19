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

def go2_leg_ik(x, y, z, L_hip=0.0955, L_thigh=0.213, L_shin=0.213):
    """
    Inverse kinematics for one Unitree Go2 leg.

    Parameters
    ----------
    x, y, z : float
        Desired foot position in the hip-yaw joint frame:
        x = forward, y = lateral (to the left), z = upwards.
    L_hip : float
        Distance from hip-yaw axis to hip-pitch axis.
    L_thigh : float
        Length of the thigh link.
    L_shin : float
        Length of the shin link.

    Returns
    -------
    q_yaw, q_hip, q_knee : tuple of floats
        Joint angles (radians):
          q_yaw  – rotation about vertical (hip yaw)
          q_hip  – hip pitch
          q_knee – knee pitch
    """

    # 1) Hip yaw: project foot into horizontal plane
    q_yaw = np.arctan2(y, x)

    # 2) Project into sagittal plane of the hip-pitch joint
    r = np.hypot(x, y) - L_hip
    d = np.hypot(r, z)

    # 3) Law of cosines for knee angle
    cos_knee = (L_thigh**2 + L_shin**2 - d**2) / (2 * L_thigh * L_shin)
    cos_knee = np.clip(cos_knee, -1.0, 1.0)
    q_knee = np.pi - np.arccos(cos_knee)

    # 4) Hip pitch: angle to target plus offset from triangle
    alpha = np.arctan2(-z, r)                # downwards is positive hip-pitch
    cos_beta = (L_thigh**2 + d**2 - L_shin**2) / (2 * L_thigh * d)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)
    q_hip = alpha + beta

    return q_yaw, q_hip, q_knee

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