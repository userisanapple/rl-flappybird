from tqdm import tqdm

import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np

env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
# env = gym.make("FlappyBird-v0", use_lidar=False)

# n bins here for horizontal distance, still tweaking this number
h_bins = np.linspace(-1, 1, num=5) 

# may change later to be different than horizontal bins
v_bins = np.linspace(-1, 1, num=15)

rng = np.random.default_rng(seed=1234)
# q_table = np.zeros((v_bins.size, h_bins.size, v_bins.size, env.action_space.n))
q_table = rng.uniform(low=0, high=0.1, size=(v_bins.size, h_bins.size, v_bins.size, env.action_space.n))

def obs_to_state(obs):
    # get vertical distance between two pipes
    last_pipe_v = obs[2]
    next_pipe_v = obs[5]
    pipe_v_distance = last_pipe_v - next_pipe_v

    # get horizontal and vertical distance to next pipe
    last_pipe_h = obs[0]
    next_pipe_h = obs[3]
    player_v = obs[9]

    next_pipe_distance = next_pipe_h
    player_pipe_v_distance = player_v - next_pipe_v
    if last_pipe_h >= 0:
        next_pipe_distance = last_pipe_h
        player_pipe_v_distance = player_v - last_pipe_v

    b1 = np.digitize(player_pipe_v_distance, v_bins) - 1
    b2 = np.digitize(next_pipe_distance, h_bins) - 1
    b3 = np.digitize(pipe_v_distance, v_bins) - 1

    return (b1, b2, b3)

df = 0.95
lr = 0.05

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

        q_next_state = q_table[obs_to_state(obs)]
        q_table[old_state][action] = (1-lr) * q_state[action] + lr * (reward + df * np.max(q_next_state))
        
        # Checking if the player is still alive
        if terminated:
            break

    if not episode % 5000:
        np.save(f'{episode}-qtable.npy', q_table)

env.close()