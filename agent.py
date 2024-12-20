from tqdm import tqdm

import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np

def obs_to_qval(obs):
    last_pipe = obs[:3]
    l_pipe_h_bin = np.digitize(last_pipe[0], h_bins) - 1
    l_top_pipe_v_bin = np.digitize(last_pipe[1], v_bins) - 1
    l_bottom_pipe_v_bin = np.digitize(last_pipe[2], v_bins) - 1

    player = obs[9:]
    player_v_bin = np.digitize(player[0], v_bins) - 1
    player_velocity_bin = np.digitize(player[1], v_bins) - 1
    player_rotation_bin = np.digitize(player[1], v_bins) - 1

    return q_table[l_pipe_h_bin,l_top_pipe_v_bin,l_bottom_pipe_v_bin,player_v_bin,player_velocity_bin,player_rotation_bin,:]

def obs_to_qtable(obs, action, q_value):
    last_pipe = obs[:3]
    l_pipe_h_bin = np.digitize(last_pipe[0], h_bins) - 1
    l_top_pipe_v_bin = np.digitize(last_pipe[1], v_bins) - 1
    l_bottom_pipe_v_bin = np.digitize(last_pipe[2], v_bins) - 1

    player = obs[9:]
    player_v_bin = np.digitize(player[0], v_bins) - 1
    player_velocity_bin = np.digitize(player[1], v_bins) - 1
    player_rotation_bin = np.digitize(player[1], v_bins) - 1

    q_table[l_pipe_h_bin,l_top_pipe_v_bin,l_bottom_pipe_v_bin,player_v_bin,player_velocity_bin,player_rotation_bin,action] = q_value


env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)

# n bins here for horizontal distance, still tweaking this number
h_bins = np.linspace(-1, 1, num=5) 

# may change later to be different than horizontal bins
v_bins = h_bins 

rng = np.random.default_rng(seed=1234)
q_table = rng.uniform(low=0, high=0.1, size=(5, 5, 5, 5, 5, 5, 2))

df = 0.95
lr = 0.1

n_episodes = 1000

for episode in tqdm(range(n_episodes)):
    obs, _ = env.reset()
    while True:
        # Next action:
        # (feed the observation to your agent here)
        # action = env.action_space.sample()
        q_value = obs_to_qval(obs)
        action = np.argmax(q_value)
        # print(np.argmax(q_value))

        # Processing:
        obs_old = obs
        obs, reward, terminated, _, info = env.step(action)

        q_next_value = obs_to_qval(obs)
        q_new = ((1-lr) * q_value[action]) + (lr * (reward + (df * np.max(q_next_value))))
        obs_to_qtable(obs_old, action, q_new)
        
        # Checking if the player is still alive
        if terminated:
            break

env.close()