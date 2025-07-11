import numpy as np
import matplotlib.pyplot as plt

Dog_Width = 0.11
Dog_Length = 0.21
Base_Height = 0.02
Total_Steps = 200
Leg_Height = 0.05
Step_Length = 0.03
Max_Height = 0.05
Between_Legs = 0.16


def main():

    com_start = np.array([0.0, 0.0, Leg_Height + Base_Height / 2])
    com_end = np.array([0.0, 0.1, Leg_Height + Base_Height / 2])

    delta = com_end[:2] - com_start[:2]
    target_angle = np.rad2deg(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta[:2]) > 1e-6 else 0.0

    foot_traj, com_traj = [], []

    if abs(target_angle) > 1e-3:
        turn_traj, turn_com = generate_turn_gait(target_angle)
        foot_traj += turn_traj
        com_traj += turn_com

    walk_traj, walk_com = generate_gait_with_com(com_start, com_end)
    foot_traj += walk_traj
    com_traj += walk_com

    com_graph(com_traj)
    foot_graph(foot_traj)
    #plot_global_foot_x_motion(foot_traj, com_traj)
    plot_3d_trajectory(foot_traj, com_traj)

def interpolate(start, end, steps):
    return np.linspace(start, end, steps)


def generate_foot_trajectory(start_pos, step_length, swing_height, swing_steps, is_turning, step_direction=None):
    traj = []
    if step_direction is None:
        step_direction = np.array([1, 0])  # Varsayılan: x yönünde

    # Adım yönünü normalize et
    step_direction = np.array(step_direction)
    if np.linalg.norm(step_direction) > 1e-6:
        step_direction = step_direction / np.linalg.norm(step_direction)

    for i in range(swing_steps):
        phase = i / (swing_steps - 1)
        # Adım vektörünü yönlendir
        step_vec = step_length * np.sin(np.pi * phase) * step_direction

        x = start_pos[0] + step_vec[0]
        y = start_pos[1] + step_vec[1]

        if is_turning:
            z = swing_height*0.5 * (0.7 * np.sin(np.pi * phase)**2)
        else:
            z = swing_height * np.sin(np.pi * phase)

        if phase > 0.8:  # Yumuşak iniş
            z *= 0.5 * (1 - phase)

        traj.append((x, y, max(0, z)))
    return traj


def generate_gait_with_com(com_start, com_end):
    base_feet_pos = np.array([
        [Between_Legs / 2, -Dog_Width / 2, 0],   # Leg 0: Front Right
        [Between_Legs / 2, Dog_Width / 2, 0],    # Leg 1: Front Left
        [-Between_Legs / 2, -Dog_Width / 2, 0],   # Leg 2: Rear Right
        [-Between_Legs / 2, Dog_Width / 2, 0]     # Leg 3: Rear Left
    ])

    # Ağırlık merkezi trajektorisi
    com_traj = interpolate(com_start, com_end, Total_Steps)

    # Adım yönünü hesapla (başlangıç ve bitiş arasındaki vektör)
    step_direction = (com_end[:2] - com_start[:2])
    if np.linalg.norm(step_direction) > 1e-6:
        step_direction = step_direction / np.linalg.norm(step_direction)

    # Walk gait sırası
    gait_order = [0, 3, 1, 2]
    swing_steps = 25

    trajectory = []
    com_positions = []

    current_foot_states = base_feet_pos.copy()

    for t in range(Total_Steps):
        current_phase = t // swing_steps
        swing_leg = gait_order[current_phase % 4]
        feet_positions = base_feet_pos.copy()

        # CoM pozisyonu (x, y, z)
        com = com_traj[t]
        com_positions.append(com.tolist())

        # Stance bacaklar adım yönünün tersine hareket eder
        for i in range(4):
            if i != swing_leg:
                feet_positions[i][:2] -= (Step_Length / swing_steps) * step_direction

        # Swing yapan bacak için trajektori
        local_step_index = t % swing_steps

        if local_step_index < swing_steps:
            traj = generate_foot_trajectory(
                base_feet_pos[swing_leg],
                Step_Length,
                Max_Height,
                swing_steps,
                is_turning=False,
                step_direction=step_direction
            )
            feet_positions[swing_leg] = traj[local_step_index]

            if local_step_index == swing_steps - 1:
                current_foot_states[swing_leg] = traj[-1]

        else:
            current_foot_states = feet_positions.copy()

        trajectory.append(feet_positions.tolist())

    return trajectory, com_positions


def generate_turn_gait(angle_deg):
    swing_steps = 25
    angle_rad = np.deg2rad(angle_deg)

    base_feet_pos = np.array([
        [Between_Legs / 2, -Dog_Width / 2, 0],
        [Between_Legs / 2, Dog_Width / 2, 0],
        [-Between_Legs / 2, -Dog_Width / 2, 0],
        [-Between_Legs / 2, Dog_Width / 2, 0]
    ])

    gait_order = [0, 3, 1, 2]
    trajectory = []

    radius = 0.02
    theta_vals = np.linspace(0, angle_rad, Total_Steps)
    """
    com_traj = [
        np.array([
            radius * np.sin(theta),
            radius * (1 - np.cos(theta)),
            Leg_Height + Base_Height / 2
        ]) for theta in theta_vals
    ]
    """
    com_traj = []
    current_theta = 0

    for idx, theta in enumerate(theta_vals):
        noise = 0.002 * np.sin(4 * np.pi * idx / Total_Steps)
        x = radius * np.sin(theta) + noise
        y = radius * (1 - np.cos(theta)) + noise
        com_traj.append(np.array([x, y, Leg_Height + Base_Height / 2]))
        current_theta = theta

    for t in range(Total_Steps):
        current_phase = t // swing_steps
        swing_leg = gait_order[current_phase % 4]
        feet_positions = base_feet_pos.copy()

        theta_step = theta_vals[t]
        for i in range(4):
            pos = base_feet_pos[i, :2]
            #rotated = rotate(pos, (angle_rad / 4) * (current_phase if i != swing_leg else 0))
            rotated = rotate(pos, theta_step)
            feet_positions[i, 0] = rotated[0]
            feet_positions[i, 1] = rotated[1]

        local_step_index = t % swing_steps

        if local_step_index < swing_steps:
            traj = generate_foot_trajectory(
                base_feet_pos[swing_leg],
                0.0,
                Max_Height,
                swing_steps,
                is_turning=True
            )
            feet_positions[swing_leg] = traj[local_step_index]

        trajectory.append(feet_positions.tolist())

    return trajectory, [c.tolist() for c in com_traj]

def rotate(p, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R @ p


def com_graph(com_traj):
    com_traj = np.array(com_traj)
    plt.figure(figsize=(8, 4))
    plt.plot(com_traj[:, 0], label='x')
    plt.plot(com_traj[:, 1], label='y')
    plt.plot(com_traj[:, 2], label='z')
    plt.title('Ağırlık Merkezi (CoM) Zamanla Değişimi')
    plt.xlabel('Zaman Adımı')
    plt.ylabel('Konum (m)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def foot_graph(foot_traj):
    foot_traj_array = np.array(foot_traj)  # boyut: (100, 4, 3)
    timesteps = foot_traj_array.shape[0]

    # Bacak etiketleri
    leg_labels = ["Front Right (0)", "Front Left (1)", "Rear Right (2)", "Rear Left (3)"]

    # --- Z Koordinatı (Yukarı-Aşağı) Grafiği ---
    plt.figure(figsize=(10, 4))
    for leg in range(4):
        z_values = foot_traj_array[:, leg, 2]
        plt.plot(range(timesteps), z_values, label=leg_labels[leg])

    plt.title("Bacakların Z Koordinatları (Swing Yüksekliği)")
    plt.xlabel("Zaman Adımı")
    plt.ylabel("Z (Yükseklik) [m]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- X Koordinatı (İleri-Geri) Grafiği ---
    plt.figure(figsize=(10, 4))
    for leg in range(4):
        x_values = foot_traj_array[:, leg, 0]
        plt.plot(range(timesteps), x_values, label=leg_labels[leg])

    plt.title("Bacakların X Koordinatları (Adım İlerlemesi)")
    plt.xlabel("Zaman Adımı")
    plt.ylabel("X (İleri-Geri) [m]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Y Koordinatı (Yan) Grafiği ---
    plt.figure(figsize=(10, 4))
    for leg in range(4):
        y_values = foot_traj_array[:, leg, 1]
        plt.plot(range(timesteps), y_values, label=leg_labels[leg])

    plt.title("Bacakların Y Koordinatları (Yan Konum)")
    plt.xlabel("Zaman Adımı")
    plt.ylabel("Y (Sağ-Sol) [m]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_global_foot_x_motion(foot_traj, com_traj):
    """
    Ayakların dünya koordinat sistemindeki (global) X eksenindeki konumlarını çizer.
    """
    foot_traj = np.array(foot_traj)  # boyut: (Total_Steps, 4, 3)
    com_traj = np.array(com_traj)    # boyut: (Total_Steps, 3)

    timesteps = foot_traj.shape[0]
    leg_labels = ["Front Right (0)", "Front Left (1)", "Rear Right (2)", "Rear Left (3)"]

    plt.figure(figsize=(10, 4))
    for leg in range(4):
        global_x = foot_traj[:, leg, 0] + com_traj[:, 0]
        plt.plot(range(timesteps), global_x, label=leg_labels[leg])

    plt.title("Ayakların Global X Koordinatları (Dünya Koordinatında Adım İlerlemesi)")
    plt.xlabel("Zaman Adımı")
    plt.ylabel("X (Global) [m]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_3d_trajectory(foot_traj, com_traj):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    foot_traj = np.array(foot_traj)
    com_traj = np.array(com_traj)

    # COM yolunu çiz (kırmızı kalın çizgi)
    ax.plot(com_traj[:,0], com_traj[:,1], com_traj[:,2],
            'r-', linewidth=3, label='COM Trajectory')

    # Ayak uçlarının son pozisyonlarını çiz (büyük noktalar)
    colors = ['blue', 'green', 'cyan', 'magenta']
    leg_labels = ['FR', 'FL', 'RR', 'RL']

    for leg in range(4):
        # Global ayak pozisyonları (COM + local foot position)
        x = foot_traj[:,leg,0] + com_traj[:,0]
        y = foot_traj[:,leg,1] + com_traj[:,1]
        z = foot_traj[:,leg,2] + com_traj[:,2]

        # Sadece son pozisyonu göster
        ax.scatter(x[-1], y[-1], z[-1],
                  color=colors[leg], s=100, label=f'{leg_labels[leg]} Foot Final Position')

        # Ayak izlerini göster (ince çizgiler)
        ax.plot(x, y, z, color=colors[leg], alpha=0.3, linewidth=1)

    # Başlangıç ve bitiş noktalarını işaretle
    ax.scatter(com_traj[0,0], com_traj[0,1], com_traj[0,2],
              color='black', s=150, marker='*', label='Start')
    ax.scatter(com_traj[-1,0], com_traj[-1,1], com_traj[-1,2],
              color='gold', s=150, marker='*', label='End')

    # Eksenleri ve görünümü ayarla
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('COM and Foot Tip Trajectories')
    ax.legend(loc='upper right')

    # Eşit ölçekli eksenler
    max_range = np.array([x.max()-x.min() for x in [com_traj[:,0], com_traj[:,1], com_traj[:,2]]]).max() * 0.5
    mid_x = (com_traj[:,0].max()+com_traj[:,0].min()) * 0.5
    mid_y = (com_traj[:,1].max()+com_traj[:,1].min()) * 0.5
    mid_z = (com_traj[:,2].max()+com_traj[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Görünüm açısını ayarla
    ax.view_init(elev=25, azim=-45)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

