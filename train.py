from agent import FlappyBirdAgent
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from tqdm import tqdm

import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import os


if __name__ == '__main__':
    n_episodes = 500_000
    save_interval = 5000

    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    env = RecordVideo(env, video_folder="episodes", name_prefix="training", episode_trigger=lambda x: x % save_interval == 0)
    env = RecordEpisodeStatistics(env)

    agent = FlappyBirdAgent(env=env, start_epsilon=1.0, epsilon_decay=1.0/(n_episodes/2), end_epsilon=0.1)

    for episode in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        while True:
            # Next action:
            # (feed the observation to your agent here)
            state = agent.obs_to_state(obs)
            action = agent.get_action(obs)

            # Processing:
            obs, reward, terminated, _, info = env.step(action)

            agent.update(state, action, reward, terminated, obs)
            
            # Checking if the player is still alive
            if terminated:
                break
            agent.decay_epsilon()

        if not episode % save_interval:
            if not os.path.exists('episodes'):
                os.mkdir('episodes')
            print(f'Episode statistics: {info['episode']}')
            episode_path = os.path.join('episodes', f'{episode}-qtable.npy')
            np.save(episode_path, agent.q_table)

    env.close()