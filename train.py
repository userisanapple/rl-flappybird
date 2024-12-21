from agent import FlappyBirdAgent
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from tqdm import tqdm

import flappy_bird_gymnasium
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    n_episodes = 100_000
    save_interval = 10000

    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    env = RecordVideo(env, video_folder="episodes", name_prefix="training", episode_trigger=lambda x: x % save_interval == 0)
    env = RecordEpisodeStatistics(env)

    agent = FlappyBirdAgent(
        env=env,
        start_lr=0.1,
        lr_decay=1.0/(n_episodes*2),
        start_epsilon=1.5,
        epsilon_decay=1.0/(n_episodes/2),
        end_epsilon=0.1,
    )

    ep_min_scores = []
    ep_cumulative_scores = []
    ep_max_scores = []
    ep_lengths = []

    interval_total_reward = 0
    interval_total_len = 0
    for episode in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        
        max_score = None
        min_score = None
        while True:
            # Next action:
            # (feed the observation to your agent here)
            state = agent.obs_to_state(obs)
            action = agent.get_action(obs)

            # Processing:
            obs, reward, terminated, _, info = env.step(action)
            if max_score is None or reward > max_score:
                max_score = reward
            if min_score is None or reward < min_score:
                min_score = reward

            agent.update(state, action, reward, terminated, obs)
            
            # Checking if the player is still alive
            if terminated:
                if 'episode' in info:
                    interval_total_reward += info['episode']['r']
                    interval_total_len += info['episode']['l']

                    ep_lengths.append(info['episode']['l'])
                    ep_min_scores.append(min_score)
                    ep_cumulative_scores.append(info['episode']['r'])
                    ep_max_scores.append(max_score)
                break
            agent.decay()

        if not episode % save_interval:
            if not os.path.exists('episodes'):
                os.mkdir('episodes')
            print(f'avg score: {interval_total_reward/save_interval}', end='\t')
            if 'episode' in info:
                print(f'avg len: {interval_total_len/save_interval}', end='\t')
            print(f'lr: {agent.lr}', end='\t')
            print(f'epsilon: {agent.epsilon}', end='\t')
            interval_total_reward = 0
            interval_total_len = 0
            episode_path = os.path.join('episodes', f'{episode}-qtable.npy')
            np.save(episode_path, agent.q_table)
        
    env.close()

    fig, ax = plt.subplots(1, 2, figsize=(15,5), sharex=True)
    linspace_x = np.linspace(0, n_episodes, num=100)
    scores_ax = ax[0]
    scores_ax.plot(linspace_x, ep_min_scores, label='Min episode score')
    scores_ax.plot(linspace_x, ep_cumulative_scores, label='Cumulative episode score')
    scores_ax.plot(linspace_x, ep_max_scores, label='Max episode score')
    scores_ax.set_ylabel('Score')
    scores_ax.legend()

    length_ax = ax[1]
    length_ax.plot(linspace_x, ep_lengths, label='Episode length')
    length_ax.set_ylabel('Length')
    length_ax.legend()
    
    fig.supxlabel('Episode #')
    plt.tight_layout()
    plt.show()