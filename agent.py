import gymnasium as gym
import numpy as np

class FlappyBirdAgent:
    def __init__(
        self,
        *,
        env: gym.Env,
        start_epsilon: float,
        epsilon_decay: float,
        end_epsilon: float,
        learning_rate: float = 0.01,
        discount_factor: float = 0.95,
        seed: int = 1234
    ):
        self.env = env
        self.lr = learning_rate
        self.df = discount_factor
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.end_epsilon = end_epsilon
        self.h_bins = np.linspace(-1, 1, num=5) 
        self.v_bins = np.linspace(-1, 1, num=15) 
        self.rng = np.random.default_rng(seed=seed)
        self.q_table = self.rng.uniform(low=0, high=0.1, size=(self.v_bins.size, self.h_bins.size, self.v_bins.size, env.action_space.n))

    def obs_to_state(self, obs: gym.spaces.Space):
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

        b1 = np.digitize(player_pipe_v_distance, self.v_bins) - 1
        b2 = np.digitize(next_pipe_distance, self.h_bins) - 1
        b3 = np.digitize(pipe_v_distance, self.v_bins) - 1

        return (b1, b2, b3)
    
    def get_action(self, obs: gym.spaces.Space):
        if self.rng.random() < self.epsilon:
            return self.env.action_space.sample()
        q_state = self.q_table[self.obs_to_state(obs)]
        return np.argmax(q_state)

    def update(self, old_state: tuple, action: int, reward: float, obs: gym.spaces.Space):
        q_sa = (1-self.lr) * self.q_table[old_state][action]
        r = self.lr * reward
        new_state = self.obs_to_state(obs)
        new_action = np.max(self.q_table[new_state])
        q_sa_new = self.lr * self.df * new_action
        self.q_table[old_state][action] = q_sa + r + q_sa_new
    
    def decay_epsilon(self):
        self.epsilon = max(self.end_epsilon, self.epsilon - self.epsilon_decay)