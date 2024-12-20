from tqdm import tqdm

import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np

def obs_to_state(obs):
    last_pipe = obs[:3]
    l_pipe_h_bin = np.digitize(last_pipe[0], h_bins) - 1

    player = obs[9:]
    player_v_bin = np.digitize(player[0], v_bins) - 1

    # player vertical position - last bottom pipe vertical position
    height_diff_bin = np.digitize(player[0] - last_pipe[2], v_bins) - 1

    return (l_pipe_h_bin, player_v_bin, height_diff_bin)

env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
# env = gym.make("FlappyBird-v0", use_lidar=False)

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
        q_state = q_table[obs_to_state(obs)]
        action = np.argmax(q_state)

        # Processing:
        old_state = obs_to_state(obs)
        obs, reward, terminated, _, info = env.step(action)


        last_pipe_top_v = obs[1]
        last_pipe_bottom_v = obs[2]
        player_v = obs[9]

        # penalize being if below bottom pipe
        if player_v < last_pipe_bottom_v:
            reward -= .25 * (last_pipe_bottom_v - player_v)
        # or above top pipe
        if player_v > last_pipe_top_v:
            reward -= .25 * (player_v - last_pipe_top_v)

        q_next_state = q_table[obs_to_state(obs)]
        q_table[old_state][action] = (1-lr) * q_state[action] + lr * (reward + df * np.max(q_next_state))
        
        # Checking if the player is still alive
        if terminated:
            break

    if not episode % 5000:
        np.save(f'{episode}-qtable.npy', q_table)

env.close()