from agent import FlappyBirdAgent
from gymnasium.wrappers import RecordVideo
from tqdm import tqdm

import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import os

if __name__ == '__main__':
    n_episodes = 100_000
    save_interval = 5000

    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    env = RecordVideo(env, video_folder="flappybird-agent", name_prefix="training", episode_trigger=lambda x: x % save_interval == 0)

    agent = FlappyBirdAgent(env)

    for episode in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        while True:
            # Next action:
            # (feed the observation to your agent here)
            action = agent.get_action(obs)

            # Processing:
            old_state = agent.obs_to_state(obs)
            obs, reward, terminated, _, info = env.step(action)

            agent.update(old_state, action, reward, obs)
            
            # Checking if the player is still alive
            if terminated:
                break

        if not episode % save_interval:
            if not os.path.exists('episodes'):
                os.mkdir('episodes')
            episode_path = os.path.join('episodes', f'{episode}-qtable.npy')
            np.save(episode_path, agent.q_table)

    env.close()