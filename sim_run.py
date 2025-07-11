import time
import mujoco
import mujoco.viewer
import numpy as np

# Load model    
m = mujoco.MjModel.from_xml_path('C:/Main/enro/robodog/git/gym-quadruped-master/gym_quadruped/robot_model/go2/go21.xml')
d = mujoco.MjData(m)

# # P gain
# Kp = 20.0

# # Target wheel velocities
# target_vel_list = np.array([
#     [1, 1, 1, 1],
#     [1.5, -1.5, -1.5, 1.5],
#     [-1, -1, -1, -1],
#     [-1.5, 1.5, 1.5, -1.5]
# ])

# # Velocity feedback
# mobile_dot = np.zeros(4)
# command = np.zeros(4)

# # Controller
# def P_controller(target_vel, measured_vel):
#     error = target_vel - measured_vel
#     return Kp * error

# t = 0
# delay = 1000

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        # step_start = time.time()
        # mujoco.mj_step(m, d)

        # # Get wheel joint velocities
        # mobile_dot[0] = d.qvel[6]   # FL
        # mobile_dot[1] = d.qvel[9]   # FR
        # mobile_dot[2] = d.qvel[12]  # RL
        # mobile_dot[3] = d.qvel[15]  # RR

        # # Pick target
        # if t < delay:
        #     target_vel = np.zeros(4)
        # elif t < delay + 4000:
        #     target_vel = target_vel_list[0]
        # elif t < delay + 8000:
        #     target_vel = target_vel_list[1]
        # elif t < delay + 12000:
        #     target_vel = target_vel_list[2]
        # elif t < delay + 16000:
        #     target_vel = target_vel_list[3]
        # else:
        #     target_vel = np.zeros(4)

        # # Compute P controller
        # command = P_controller(target_vel, mobile_dot)

        # # Send control commands → match to actuators
        # d.ctrl[0] = command[1]   # FR hip
        # d.ctrl[3] = command[0]   # FL hip
        # d.ctrl[6] = command[3]   # RR hip
        # d.ctrl[9] = command[2]   # RL hip

        # viewer.sync()

        # # Keep step time consistent
        # time_until_next_step = m.opt.timestep - (time.time() - step_start)
        # if time_until_next_step > 0:
        #     time.sleep(time_until_next_step)

        # if t % 100 == 0:
        #     print("current velocity:", mobile_dot)
        #     print("control force:", command)
        #     print()

        # t += 1
        pass