from dqn_agent import FlappyBirdDQN
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from tqdm import tqdm

import flappy_bird_gymnasium
import gymnasium as gym
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import wandb

def get_pyplot_from_data(n_episodes: int, ep_min_scores: list, ep_average_scores: list, ep_cumulative_scores: list, ep_max_scores: list, ep_lengths: list) -> tuple:
    plt.close()

    x_resolution = 75

    # the np.reshape we do below requires that x_resolution is a multiple of the size of our input array, so we ignore the last index_truncate values to satisfy this requirement
    index_truncate = 0
    while (n_episodes - index_truncate) % x_resolution:
        index_truncate += 1

    fig, ax = plt.subplots(1, 2, figsize=(15,5), sharex=True)
    linspace_x = np.linspace(0, n_episodes, num=x_resolution, dtype=int)
    new_shape = (-1, (n_episodes - index_truncate) // x_resolution)
    scores_ax = ax[0]
    scores_ax.plot(linspace_x, np.average(np.array(ep_cumulative_scores[:-index_truncate]).reshape(new_shape), axis=1), label='Cumulative episode reward')
    scores_ax.plot(linspace_x, np.average(np.array(ep_average_scores[:-index_truncate]).reshape(new_shape), axis=1), label='Average episode reward')
    scores_ax.plot(linspace_x, np.average(np.array(ep_min_scores[:-index_truncate]).reshape(new_shape), axis=1), label='Min episode reward')
    scores_ax.plot(linspace_x, np.average(np.array(ep_max_scores[:-index_truncate]).reshape(new_shape), axis=1), label='Max episode reward')
    scores_ax.set_ylabel('Reward')
    scores_ax.legend()

    length_ax = ax[1]
    length_ax.plot(linspace_x, np.average(np.array(ep_lengths[:-index_truncate]).reshape(new_shape), axis=1), label='Episode length')
    length_ax.set_ylabel('Length')
    length_ax.legend()
    
    fig.supxlabel('Episode #')
    plt.tight_layout()
    return (fig, ax)

if __name__ == '__main__':
    save_interval = 250
    save = True

    try:
        with open('model_conf.json', 'r') as conf_file:
            config = json.load(conf_file)
    except:
        print("Couldn't load config")
        exit()

    wandb.init(
        project="rl-flappybird",
        config=config
    )

    if save:
        env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
        env = RecordVideo(env, video_folder="episodes", name_prefix="training", episode_trigger=lambda x: x % save_interval == 0)
    else:
        env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    env = RecordEpisodeStatistics(env)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)

    agent = FlappyBirdDQN(
        env=env,
        torch_device=device,
        memory_size=config['replay_memory_size'],
        optim_lr=config['learning_rate'],
        batch_size=config['batch_size'],
        start_epsilon=config['start_epsilon'],
        epsilon_decay=config['epsilon_decay'],
        end_epsilon=config['end_epsilon'],
        discount_factor=config['discount_factor'],
    )

    ep_min_scores = []
    ep_avg_scores = []
    ep_cumulative_scores = []
    ep_max_scores = []
    ep_lengths = []

    interval_total_reward = 0
    interval_total_len = 0
    for episode in tqdm(range(config['n_episodes'])):
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

            agent.memory_push(state, action, agent.obs_to_state(obs), shaped_reward, terminated)

            if not ep_steps % config['policy_update_interval']:
                agent.train()
            if not ep_steps % config['target_update_interval']:
                agent.update_target_net()
            
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
            print(f'epsilon: {agent.epsilon}')
            interval_total_reward = 0
            interval_total_len = 0

            if save:
                # make episode dir
                if not os.path.exists('episodes'):
                    os.mkdir('episodes')

                # save dqn model checkpoint
                agent.serialize('episodes', episode)

                # save plot
                plot_data = np.array([ep_min_scores, ep_avg_scores, ep_cumulative_scores, ep_max_scores, ep_lengths]) 
                np.save(os.path.join('episodes', f'plot-{episode}.npy'), plot_data)

                fig, ax = get_pyplot_from_data(episode+1, ep_min_scores, ep_avg_scores, ep_cumulative_scores, ep_max_scores, ep_lengths)
                plt.savefig(os.path.join('episodes', f'plot-{episode}.png'))
        agent.decay()

        wandb.log({"cumulative-score": cumulative_score, "episode-length": info['episode']['l'], "epsilon": agent.epsilon})

    env.close()

    fig, ax = get_pyplot_from_data(config['n_episodes'], ep_min_scores, ep_avg_scores, ep_cumulative_scores, ep_max_scores, ep_lengths)
    if save:
        agent.serialize('episodes', episode)

        plot_data = np.array([ep_min_scores, ep_avg_scores, ep_cumulative_scores, ep_max_scores, ep_lengths]) 
        np.save(os.path.join('episodes', f'plotdata-{episode}.npy'), plot_data)

        plt.savefig(os.path.join('episodes', f'plot-{episode}.png'))
    plt.show()