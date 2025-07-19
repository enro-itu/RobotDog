import time
import mujoco
import mujoco.viewer
import numpy as np
import gait_updated as gait
from inverse_updated import inverse_kinematics as leg_IK

# Load model
# Change this according to your path
m = mujoco.MjModel.from_xml_path(
    'C:/Main/enro/robodog/git/gym-quadruped-master/gym_quadruped/robot_model/go2/go21.xml'
)
d = mujoco.MjData(m)

# P gain for position control
Kp = 20.0

foot_traj = gait.gait(0, 0, 100, 0)
each_step = []

for foot_point in foot_traj:
    each = ()
    for coord in foot_point:
        each = each + ((leg_IK(coord[0], coord[1], coord[2], 0.067, 0.213, 0.210, 0.094)),)
    each_step.append(each)

"""
# Target wheel joint positions (degrees)
target_pos_list_deg = np.array([
    [0, 0, 0, 0],
    [90, -90, -90, 90],
    [-30, -30, -30, -30],
    [-90, 90, 90, -90]
])
"""
target_pos_list_deg = []

for step in each_step:
    a = []
    for k in step:
        for l in k:
            a.append(l)
    target_pos_list_deg.append(a)

# Convert target degrees → radians
target_pos_list = np.deg2rad(target_pos_list_deg)

# Current joint positions (radians)
mobile_pos = np.zeros(12)

# Controller: P control for position
def P_controller(target_pos, measured_pos):
    error = target_pos - measured_pos
    return Kp * error

t = 0
delay = 100

with mujoco.viewer.launch_passive(m, d) as viewer:
    time.sleep(5)
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(m, d)

        # Get joint positions (check indices!)
        # FL
        mobile_pos[0] = d.qpos[6]   
        mobile_pos[1] = d.qpos[7]
        mobile_pos[2] = d.qpos[8]
        # FR
        mobile_pos[3] = d.qpos[9]   
        mobile_pos[4] = d.qpos[10]
        mobile_pos[5] = d.qpos[11]
        # RL
        mobile_pos[6] = d.qpos[12]  
        mobile_pos[7] = d.qpos[13]
        mobile_pos[8] = d.qpos[14]
        # RR
        mobile_pos[9] = d.qpos[15]  
        mobile_pos[10] = d.qpos[16]
        mobile_pos[11] = d.qpos[17]

        """
        # Pick target
        if t < delay:
            target_pos = np.zeros(4)
        elif t < delay + 1000:
            target_pos = target_pos_list[0]
        elif t < delay + 2000:
            target_pos = target_pos_list[1]
        elif t < delay + 3000:
            target_pos = target_pos_list[2]
        elif t < delay + 4000:
            target_pos = target_pos_list[3]
        else:
            target_pos = np.zeros(4)
        """
        # Example: suppose you have 200 different target_pos_list entries
        # So: target_pos_list[0], target_pos_list[1], ..., target_pos_list[199]

        # Pick target
        if t < delay:
            target_pos = np.zeros(12)
        else:
            index = (t - delay) // 10  # or whatever step you want
            if index < 12:
                target_pos = target_pos_list[index]
            else:
                target_pos = np.zeros(12)  # or whatever fallback

            
        # Compute P control command (torque/force)
        command = P_controller(target_pos, mobile_pos)

        # Apply to actuators — check mapping!
        d.ctrl[0] = command[0]   # FR hip
        d.ctrl[1] = command[1]
        d.ctrl[2] = command[2]
        d.ctrl[3] = command[6]   # FL hip
        d.ctrl[4] = command[7]
        d.ctrl[5] = command[8]
        d.ctrl[6] = command[9]   # RR hip
        d.ctrl[7] = command[10]
        d.ctrl[8] = command[11]
        d.ctrl[9] = command[3]   # RL hip
        d.ctrl[10] = command[4]
        d.ctrl[11] = command[5]

        viewer.sync()

        # Keep step time consistent
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        # Debug
        if t % 100 == 0:
            print("Current joint positions (deg):", np.rad2deg(mobile_pos))
            print("Control output:", command)
            print()

        t += 1
