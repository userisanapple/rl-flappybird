import gymnasium as gym
import numpy as np
from os import PathLike

class FlappyBirdAgent:
    def __init__(
        self,
        *,
        env: gym.Env,
        start_lr: float = 0.1,
        lr_decay: float = 0.0005,
        end_lr: float = 0.01,
        start_epsilon: float = 1.0,
        epsilon_decay: float = 0.005,
        end_epsilon: float = 0.1,
        discount_factor: float = 0.95,
        seed: int = 1234,
        start_episode: PathLike = None,
    ):
        self.env = env
        self.df = discount_factor
        if start_episode:
            self.lr = end_lr
            self.epsilon = end_epsilon
        else:
            self.lr = start_lr
            self.epsilon = start_epsilon
        self.lr_decay = lr_decay
        self.end_lr = end_lr
        self.epsilon_decay = epsilon_decay
        self.end_epsilon = end_epsilon
        self.h_bins = np.linspace(-1, 1, num=30) 
        self.v_bins = np.linspace(-1, 1, num=50) 
        self.rng = np.random.default_rng(seed=seed)
        q_size = (self.v_bins.size, self.h_bins.size, self.v_bins.size, env.action_space.n)
        if start_episode:
            self.q_table = np.load(start_episode)
        else:
            # self.q_table = np.full(q_size, fill_value=10.0)
            self.q_table = np.zeros(q_size)
            # self.q_table = rng.uniform(low=0, high=0.1, shape=q_size)

    def obs_to_state(self, obs: gym.spaces.Space):
        # get vertical distance between two pipes
        last_pipe_v = obs[2]
        next_pipe_v = obs[5]

        # get horizontal and vertical distance to next pipe
        last_pipe_h = obs[0]
        next_pipe_h = obs[3]
        player_v = obs[9]

        next_pipe_distance = next_pipe_h
        player_pipe_v_distance = player_v - next_pipe_v
        if last_pipe_h >= -0.15:
            next_pipe_distance = last_pipe_h
            player_pipe_v_distance = player_v - last_pipe_v

        player_velocity = obs[10]

        b1 = np.digitize(player_pipe_v_distance, self.v_bins) - 1
        b2 = np.digitize(next_pipe_distance, self.h_bins) - 1
        b3 = np.digitize(player_velocity, self.v_bins) - 1

        return (b1, b2, b3)
    
    def get_action(self, obs: gym.spaces.Space):
        if self.rng.random() < self.epsilon:
            return self.env.action_space.sample()
        q_state = self.q_table[self.obs_to_state(obs)]
        return np.argmax(q_state)
    
    def shape_reward(self, obs: gym.spaces.Space, reward: float):
        shaped_reward = reward
        if np.isclose(reward, 1.0):
            shaped_reward = 25.0
        elif np.isclose(reward, -1.0):
            shaped_reward = -10.0

        player_v = obs[9]
        next_pipe_v = obs[5]
        last_pipe_h = obs[0]
        if last_pipe_h >= -0.15:
            next_pipe_v = obs[2]
        shaped_reward -= abs((player_v-next_pipe_v) * 2)

        return shaped_reward

    def update(self, old_state: tuple, action: int, reward: float, terminated: bool, obs: gym.spaces.Space):
        reward = self.shape_reward(obs, reward)

        q_sa = self.q_table[old_state][action]
        next_action = (not terminated) * np.max(self.q_table[self.obs_to_state(obs)])
        q_sa_next = self.df * next_action
        self.q_table[old_state][action] = q_sa + (self.lr * (reward + q_sa_next - q_sa))
    
    def decay(self):
        self.lr = max(self.end_lr, self.lr - self.lr_decay)
        self.epsilon = max(self.end_epsilon, self.epsilon - self.epsilon_decay)