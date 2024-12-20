from tqdm import tqdm

import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np

def obs_to_state(obs):
    last_pipe = obs[:3]
    l_pipe_h_bin = np.digitize(last_pipe[0], h_bins) - 1

    player = obs[9:]
    # player vertical position - last bottom pipe vertical position
    player_velocity_bin = np.digitize(player[1], v_bins) - 1

    height_diff_bin = np.digitize(player[0] - last_pipe[2], v_bins) - 1

    return (l_pipe_h_bin, player_velocity_bin, height_diff_bin)

# env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
env = gym.make("FlappyBird-v0", use_lidar=False)

# n bins here for horizontal distance, still tweaking this number
h_bins = np.linspace(-1, 1, num=5) 

# may change later to be different than horizontal bins
v_bins = h_bins 

rng = np.random.default_rng(seed=1234)
q_table = rng.uniform(low=0, high=0.1, size=(5, 5, 5, 2))

df = 0.95
lr = 0.01

n_episodes = 100_000

for episode in tqdm(range(n_episodes)):
    obs, _ = env.reset()
    while True:
        # Next action:
        # (feed the observation to your agent here)
        q_value = q_table[obs_to_state(obs)]
        action = np.argmax(q_value)

        # Processing:
        obs_old = obs
        obs, reward, terminated, _, info = env.step(action)
        if not episode % 5000:
            np.save(f'{episode}-qtable.npy', q_table)

        q_next_value = q_table[obs_to_state(obs)]
        q_new = ((1-lr) * q_value[action]) + (lr * (reward + (df * np.max(q_next_value))))
        q_table[obs_to_state(obs_old),action] = q_new
        
        # Checking if the player is still alive
        if terminated:
            break

env.close()