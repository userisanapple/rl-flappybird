from agent import FlappyBirdAgent
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from tqdm import tqdm

import flappy_bird_gymnasium
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_data(n_episodes: int, ep_min_scores: list, ep_average_scores: list, ep_cumulative_scores: list, ep_max_scores: list, ep_lengths: list) -> tuple:
    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(15,5), sharex=True)
    linspace_x = np.linspace(0, n_episodes, num=n_episodes, dtype=int)
    scores_ax = ax[0]
    scores_ax.plot(linspace_x, ep_cumulative_scores, label='Cumulative episode reward')
    scores_ax.plot(linspace_x, ep_average_scores, label='Average episode reward')
    scores_ax.plot(linspace_x, ep_min_scores, label='Min episode reward')
    scores_ax.plot(linspace_x, ep_max_scores, label='Max episode reward')
    scores_ax.set_ylabel('Reward')
    scores_ax.legend()

    length_ax = ax[1]
    length_ax.plot(linspace_x, ep_lengths, label='Episode length')
    length_ax.set_ylabel('Length')
    length_ax.legend()
    
    fig.supxlabel('Episode #')
    plt.tight_layout()
    return (fig, ax)

if __name__ == '__main__':
    n_episodes = 100_000
    save_interval = 5000

    env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    env = RecordVideo(env, video_folder="episodes", name_prefix="training", episode_trigger=lambda x: x % save_interval == 0)
    env = RecordEpisodeStatistics(env)

    agent = FlappyBirdAgent(
        env=env,
        start_lr=0.1,
        lr_decay=1.0/(n_episodes*2),
        start_epsilon=1.5,
        epsilon_decay=1.0/(n_episodes/1000),
        end_epsilon=0.01,
    )

    ep_min_scores = []
    ep_avg_scores = []
    ep_cumulative_scores = []
    ep_max_scores = []
    ep_lengths = []

    interval_total_reward = 0
    interval_total_len = 0
    for episode in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        
        max_score = None
        min_score = None
        cumulative_score = 0
        ep_steps = 0
        while True:
            # Next action:
            # (feed the observation to your agent here)
            state = agent.obs_to_state(obs)
            action = agent.get_action(obs)

            # Processing:
            obs, reward, terminated, _, info = env.step(action)
            shaped_reward = agent.shape_reward(obs, reward)
            if max_score is None or shaped_reward > max_score:
                max_score = shaped_reward
            if min_score is None or shaped_reward < min_score:
                min_score = shaped_reward
            cumulative_score += shaped_reward

            agent.update(state, action, reward, terminated, obs)
            
            # Checking if the player is still alive
            if terminated:
                if 'episode' in info:
                    interval_total_reward += cumulative_score
                    interval_total_len += info['episode']['l']

                    ep_lengths.append(info['episode']['l'])
                    ep_min_scores.append(min_score)
                    ep_avg_scores.append(cumulative_score/ep_steps)
                    ep_cumulative_scores.append(cumulative_score)
                    ep_max_scores.append(max_score)
                break
            ep_steps += 1

        if not episode % save_interval and episode > 0:
            print(f'avg score: {interval_total_reward/save_interval}', end='\t')
            if 'episode' in info:
                print(f'avg len: {interval_total_len/save_interval}', end='\t')
            print(f'lr: {agent.lr}', end='\t')
            print(f'epsilon: {agent.epsilon}')
            interval_total_reward = 0
            interval_total_len = 0

            # make episode dir
            if not os.path.exists('episodes'):
                os.mkdir('episodes')

            # save qtable
            episode_path = os.path.join('episodes', f'{episode}-qtable.npy')
            np.save(episode_path, agent.q_table)

            # save plot
            fig, ax = plot_data(episode+1, ep_min_scores, ep_avg_scores, ep_cumulative_scores, ep_max_scores, ep_lengths)
            plt.savefig(os.path.join('episodes', f'{episode}-plots.png'))
        agent.decay()
        
    env.close()

    fig, ax = plot_data(n_episodes, ep_min_scores, ep_avg_scores, ep_cumulative_scores, ep_max_scores, ep_lengths)
    plt.show()