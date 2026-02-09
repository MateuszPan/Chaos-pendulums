import numpy as np  
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import seaborn as sns
import matplotlib

g = 9.81
L1 = L2 = 1.0
m1 = m2 = 1.0

def derivs(state):
    theta1, omega1, theta2, omega2 = state
    delta = theta2 - theta1
    denom1 = (m1 + m2)*L1 - m2*L1*np.cos(delta)**2
    denom2 = (L2/L1)*denom1

    domega1 = (m2*L1*omega1**2*np.sin(delta)*np.cos(delta) +
               m2*g*np.sin(theta2)*np.cos(delta) +
               m2*L2*omega2**2*np.sin(delta) -
               (m1 + m2)*g*np.sin(theta1)) / denom1

    domega2 = (-m2*L2*omega2**2*np.sin(delta)*np.cos(delta) +
               (m1 + m2)*g*np.sin(theta1)*np.cos(delta) -
               (m1 + m2)*L1*omega1**2*np.sin(delta) -
               (m1 + m2)*g*np.sin(theta2)) / denom2

    return np.array([omega1, domega1, omega2, domega2])

def runge_kutta(state, dt):
    k1 = derivs(state)
    k2 = derivs(state + 0.5 * dt * k1)
    k3 = derivs(state + 0.5 * dt * k2)
    k4 = derivs(state + dt * k3)
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

dt = 0.02
num_pendulums = 1000
trail_length = 150

cmap = matplotlib.colormaps["turbo"]

states = [np.array([np.pi / 2, 0, np.pi / 2 + i * 1e-4, 0]) for i in range(num_pendulums)]
trails = [[] for _ in range(num_pendulums)]

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.axis('off')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

trail_lines = [ax.plot([], [], lw=0.8, color=cmap(i / num_pendulums), alpha=0.5)[0]
               for i in range(num_pendulums)]
rod_lines_1 = [ax.plot([], [], lw=1.2, color=cmap(i / num_pendulums))[0] for i in range(num_pendulums)]
rod_lines_2 = [ax.plot([], [], lw=1.2, color=cmap(i / num_pendulums))[0] for i in range(num_pendulums)]

def update(frame):
    for i in range(num_pendulums):
        states[i] = runge_kutta(states[i], dt)
        theta1, _, theta2, _ = states[i]

        x1 = L1 * np.sin(theta1)
        y1 = -L1 * np.cos(theta1)

        x2 = x1 + L2 * np.sin(theta2)
        y2 = y1 - L2 * np.cos(theta2)

        trails[i].append((x2, y2))
        if len(trails[i]) > trail_length:
            trails[i].pop(0)

        xs, ys = zip(*trails[i])
        trail_lines[i].set_data(xs, ys)
        trail_lines[i].set_alpha(0.3 + 0.7 * len(trails[i]) / trail_length)

        rod_lines_1[i].set_data([0, x1], [0, y1])
        rod_lines_2[i].set_data([x1, x2], [y1, y2])

    return trail_lines + rod_lines_1 + rod_lines_2

ani = FuncAnimation(fig, update, frames=2000, interval=20, blit=True)
ani.save('z4_symulacja_1000_wahadel_podwojnych.mp4', writer='ffmpeg', fps=50, dpi=150)
plt.show()

T = 20  
steps = int(T / dt)

state_0 = np.array([np.pi / 2, 0, np.pi / 2, 0])
state_1 = np.array([np.pi / 2, 0, np.pi / 2 + 1e-4, 0])

states_0 = []
states_1 = []

for _ in range(steps):
    states_0.append([state_0[2], state_0[3]])  
    states_1.append([state_1[2], state_1[3]])

    state_0 = runge_kutta(state_0, dt)
    state_1 = runge_kutta(state_1, dt)

states_0 = np.array(states_0)
states_1 = np.array(states_1)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(states_0[:, 0], states_0[:, 1], color='blue', lw=1)
axs[0].set_title("Wahadło 0: θ₂ vs ω₂")
axs[0].set_xlabel("θ₂ [rad]")
axs[0].set_ylabel("ω₂ [rad/s]")

axs[1].plot(states_1[:, 0], states_1[:, 1], color='red', lw=1)
axs[1].set_title("Wahadło 1: θ₂ vs ω₂")
axs[1].set_xlabel("θ₂ [rad]")
axs[1].set_ylabel("ω₂ [rad/s]")

fig.suptitle("Przestrzeń fazowa: kąt vs prędkość kątowa zewnętrznego segmentu dla dwóch wahadeł", fontsize=14, y=1.60, fontweight='bold', color='navy')
plt.tight_layout()
plt.show()

theta2_diff = np.abs(states_0[:, 0] - states_1[:, 0])
time = np.linspace(0, T, steps)

plt.figure(figsize=(8, 4))
plt.plot(time, theta2_diff, color='purple')
plt.title("Różnica kątów zewnętrznych segmentów Δθ₂(t)")
plt.xlabel("Czas [s]")
plt.ylabel("Δθ₂ [rad]")
plt.yscale('log')  
plt.grid(True)
plt.tight_layout()
plt.show()

states_array = np.array(states)
df = pd.DataFrame(states_array, columns=["theta1", "omega1", "theta2", "omega2"])
corr_matrix = df.corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Macierz korelacji stanu 1000 wahadeł podwójnych")
plt.tight_layout()
plt.show()

num_to_test = 5
test_indices = np.linspace(0, num_pendulums - 1, num_to_test, dtype=int)

theta2_data = np.zeros((num_to_test, steps))
test_states = [np.array([np.pi / 2, 0, np.pi / 2 + i * 1e-4, 0]) for i in test_indices]

for step in range(steps):
    for idx, s in enumerate(test_states):
        theta2_data[idx, step] = s[2]
        test_states[idx] = runge_kutta(s, dt)

def autocorrelation(x):
    n = len(x)
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:] / (np.var(x) * n)

time_axis = np.arange(steps) * dt

fig, axs = plt.subplots(num_to_test, 2, figsize=(10, 2.8 * num_to_test))
fig.suptitle("Analiza θ₂: histogram i autokorelacja", fontsize=12, y=1.02)

for i in range(num_to_test):
    axs[i, 0].hist(theta2_data[i], bins=50, density=True, color='skyblue')
    axs[i, 0].set_title(f"Wahadło {test_indices[i]}: Histogram θ₂", fontsize=8)
    axs[i, 0].set_xlabel("θ₂ [rad]", fontsize=8)
    axs[i, 0].set_ylabel("Gęstość", fontsize=8)
    axs[i, 0].tick_params(axis='both', labelsize=7)

    acorr = autocorrelation(theta2_data[i])
    axs[i, 1].plot(time_axis[:200], acorr[:200], color='orange')
    axs[i, 1].set_title(f"Wahadło {test_indices[i]}: Autokorelacja θ₂", fontsize=8)
    axs[i, 1].set_xlabel("Opóźnienie [s]", fontsize=8)
    axs[i, 1].set_ylabel("Autokorelacja", fontsize=8)
    axs[i, 1].tick_params(axis='both', labelsize=7)
    axs[i, 1].grid(True)

plt.tight_layout(pad=2.0)
plt.show()

from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

selected_idx = 0
traj_single = np.vstack((theta2_data[selected_idx], np.gradient(theta2_data[selected_idx], dt))).T
distance_matrix_single = squareform(pdist(traj_single))
epsilon_single = 0.1
recurrence_matrix_single = distance_matrix_single < epsilon_single

plt.figure(figsize=(6, 6))
plt.imshow(recurrence_matrix_single, cmap='binary', origin='lower')
plt.title(f"Wykres powtarzalności: wahadło {test_indices[selected_idx]}")
plt.xlabel("Czas")
plt.ylabel("Czas")
plt.tight_layout()
plt.show()

steps_for_all = int(T / dt)
all_states = [np.array([np.pi / 2, 0, np.pi / 2 + i * 1e-4, 0]) for i in range(num_pendulums)]
trajectory_all = np.zeros((steps_for_all, num_pendulums * 4))

for step in range(steps_for_all):
    for i, s in enumerate(all_states):
        trajectory_all[step, 4*i:4*i+4] = s
        all_states[i] = runge_kutta(s, dt)

scaler = StandardScaler()
trajectory_all = scaler.fit_transform(trajectory_all)

distance_matrix_all = squareform(pdist(trajectory_all))
epsilon_all = 15.0
recurrence_matrix_all = distance_matrix_all < epsilon_all

plt.figure(figsize=(6, 6))
plt.imshow(recurrence_matrix_all, cmap='binary', origin='lower')
plt.title("Wykres powtarzalności: cały układ 1000 wahadeł")
plt.xlabel("Czas")
plt.ylabel("Czas")
plt.tight_layout()
plt.show()

def simulate_double_pendulum(theta1_init, theta2_init, T=20, dt=0.02):
    steps = int(T / dt)
    state = np.array([theta1_init, 0.0, theta2_init, 0.0])
    traj = []

    for _ in range(steps):
        traj.append([state[0], state[1], state[2], state[3]])
        state = runge_kutta(state, dt)

    traj = np.array(traj)
    theta2 = traj[:, 2]
    omega2 = traj[:, 3]
    return np.vstack((theta2, omega2)).T

def generate_recurrence_plot(traj, epsilon=0.1, title=""):
    dist_matrix = squareform(pdist(traj))
    rec_matrix = dist_matrix < epsilon

    plt.figure(figsize=(6, 6))
    plt.imshow(rec_matrix, cmap='binary', origin='lower')
    plt.title(f"Wykres powtarzalności: {title}")
    plt.xlabel("Czas")
    plt.ylabel("Czas")
    plt.tight_layout()
    plt.show()

traj1 = simulate_double_pendulum(theta1_init=0.1*np.pi, theta2_init=0.1*np.pi)
traj2 = simulate_double_pendulum(theta1_init=0.55*np.pi, theta2_init=0.55*np.pi)

generate_recurrence_plot(traj1, epsilon=0.1, title="θ1 = θ2 = 0.1π")
generate_recurrence_plot(traj2, epsilon=0.1, title="θ1 = θ2 = 0.55π")

state = np.array([0.55 * np.pi, 0, 0.55 * np.pi, 0])
T = 20
steps = int(T / dt)

PE = []
KE = []
TE = []

for _ in range(steps):
    theta1, omega1, theta2, omega2 = state

    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    vx1 = L1 * omega1 * np.cos(theta1)
    vy1 = L1 * omega1 * np.sin(theta1)
    vx2 = vx1 + L2 * omega2 * np.cos(theta2)
    vy2 = vy1 + L2 * omega2 * np.sin(theta2)

    pe = -(m1 + m2) * g * L1 * np.cos(theta1) - m2 * g * L2 * np.cos(theta2)
    ke = 0.5 * m1 * (vx1**2 + vy1**2) + 0.5 * m2 * (vx2**2 + vy2**2)

    PE.append(pe)
    KE.append(ke)
    TE.append(pe + ke)

    state = runge_kutta(state, dt)

time = np.linspace(0, T, steps)
plt.figure(figsize=(10, 5))
plt.plot(time, PE, label="Energia potencjalna", color='orange')
plt.plot(time, KE, label="Energia kinetyczna", color='dodgerblue')
plt.plot(time, TE, label="Energia całkowita", color='green')
plt.xlabel("Czas [s]")
plt.ylabel("Energia [J]")
plt.title("Ewolucja energii dla wahadła podwójnego (0.55π, 0.55π)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
