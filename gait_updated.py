import numpy as np
import matplotlib.pyplot as plt


Dog_Width = 0.2
Dog_Length = 0.5
Base_Height = 0.25
Total_Steps = 200
Leg_Height = 0.426
Step_Length = 0.1
Max_Height = 0.45
Between_Legs = 0.35


def gait(start_x, start_y, end_x, end_y):

    com_start = np.array([start_x, start_y, Leg_Height + Base_Height / 2])
    com_end = np.array([end_x, end_y, Leg_Height + Base_Height / 2])

    foot_traj, com_traj = generate_gait_with_com(com_start, com_end)

    return foot_traj

    #com_graph(com_traj)
    #foot_graph(foot_traj)
    #plot_global_foot_x_motion(foot_traj, com_traj)

def interpolate(start, end, steps):
    return np.linspace(start, end, steps)


def generate_foot_trajectory(start_pos, step_length, swing_height, swing_steps):
    traj = []
    for i in range(swing_steps):
        phase = i / (swing_steps - 1)
        x = start_pos[0] + step_length * np.sin(np.pi * phase)
        y = start_pos[1]
        z = swing_height * np.sin(np.pi * phase)

        if phase > 0.8:  # Yumuşak iniş
            z *= 0.5 * (1 - phase)

        traj.append((x, y, max(0, z)))
    return traj


def generate_gait_with_com(com_start, com_end):

    base_feet_pos = np.array([
        [Between_Legs / 2, -Dog_Width / 2, 0],   # Leg 0: Front Right
        [Between_Legs / 2, Dog_Width / 2, 0],   # Leg 1: Front Left
        [-Between_Legs / 2, -Dog_Width / 2, 0],  # Leg 2: Rear Right
        [-Between_Legs / 2, Dog_Width / 2, 0]   # Leg 3: Rear Left
    ])

    # Ağırlık merkezinin başlangıç ve bitiş konumları
    com_traj = interpolate(com_start, com_end, Total_Steps)

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

        # Stance bacaklar CoM'ye göre geriye doğru hareket eder
        for i in range(4):
            if i != swing_leg:
                feet_positions[i][0] -= Step_Length / swing_steps

        # Swing yapan bacak için z-x-y pozisyonu
        local_step_index = t % swing_steps

        if (local_step_index) < swing_steps:
            traj = generate_foot_trajectory(
                base_feet_pos[swing_leg],
                Step_Length,
                Max_Height,
                swing_steps)

            feet_positions[swing_leg] = traj[local_step_index]

            if local_step_index == swing_steps - 1:
                current_foot_states[swing_leg] = traj[-1]

        else:
            # Swing yapılmıyorsa stance bacak pozisyonları güncellenir
            current_foot_states = feet_positions.copy()

        trajectory.append(feet_positions.tolist())

    return trajectory, com_positions


def com_graph(com_traj):
    com_traj = np.array(com_traj)
    plt.figure(figsize=(6, 4))
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


if __name__ == "__main__":
    gait()

