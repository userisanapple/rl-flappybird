import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class FlappyBirdDQN:
    def __init__(
        self,
        *,
        env: gym.Env,
        torch_device: torch.device,
        optim_lr: float = 0.01,
        batch_size: int = 64,
        memory_size: int = 1_000_000,
        optimizer: optim.Optimizer = None,
        start_epsilon: float = 1.0,
        epsilon_decay: float = 0.005,
        end_epsilon: float = 0.1,
        discount_factor: float = 0.95,
        seed: int = 1234,
        model_checkpoint: os.PathLike = None
    ):
        self.env = env
        self.torch_device = torch_device
        self.batch_size = batch_size
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.end_epsilon = end_epsilon
        self.df = discount_factor

        self.dqn = DQN().to(self.torch_device)
        self.target_net = DQN().to(self.torch_device)
        if model_checkpoint:
            self.load(model_checkpoint)
        self.update_target_net()
        self.loss_fn = nn.SmoothL1Loss()

        # replay structure:
        # state (5 floats returned by obs_to_state), action, next state (5 floats, like state), shaped reward, terminated
        # will be 13 values per replay

        self.memory = torch.tensor(np.empty((memory_size, 13)), dtype=torch.float32, device=self.torch_device)
        self.memory_len = 0
        self.oldest_mem = 0
        self.max_memory_size = memory_size
        self.rng = np.random.default_rng(seed=seed)

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(self.dqn.parameters(), lr=optim_lr, weight_decay=0)

    def obs_to_state(self, obs: gym.spaces.Space):
        last_pipe_h = obs[0]
        last_pipe_v = obs[2]
        next_pipe_v = obs[5]
        player_vpos = obs[9]
        player_velocity = obs[10]

        return (last_pipe_h, last_pipe_v, next_pipe_v, player_vpos, player_velocity)

    def shape_reward(self, obs: gym.spaces.Space, reward: float):
        shaped_reward = reward
        # if np.isclose(reward, 1.0):
        #     shaped_reward = 25.0
        # elif np.isclose(reward, -1.0):
        #     shaped_reward = -10.0

        player_v = obs[9]
        next_pipe_v_top = obs[4]
        next_pipe_v_bottom = obs[5]
        last_pipe_h = obs[0]
        if last_pipe_h >= -0.15:
            next_pipe_v_top = obs[1]
            next_pipe_v_bottom = obs[2]

        pipe_center = next_pipe_v_bottom + ((next_pipe_v_top - next_pipe_v_bottom) / 2)

        # penalize based on distance from center of pipe (offset slightly down to account for jumping)
        player_center_delta = abs(player_v - (pipe_center-0.05))
        shaped_reward -= player_center_delta * 2

        return shaped_reward
    
    def get_action(self, obs: gym.spaces.Space):
        # if torch.rand(1, dtype=torch.float32) < self.epsilon:
        if self.rng.random() < self.epsilon:
            return self.env.action_space.sample()

        state_tensor = torch.tensor(self.obs_to_state(obs), dtype=torch.float32, device=self.torch_device)

        with torch.no_grad():
            y_pred = self.dqn(state_tensor)
        
        return y_pred.argmax()
    
    def memory_push(self, state: tuple, action: int, next_state: tuple, reward: float, terminated: bool):
        replay_tensor = torch.tensor((*state, action, *next_state, reward, terminated), dtype=torch.float32, device=self.torch_device) 

        if self.memory_len < self.max_memory_size:
            self.memory[self.memory_len] = replay_tensor
            self.memory_len += 1
        else:
            self.memory[self.oldest_mem] = replay_tensor
            self.oldest_mem = (self.oldest_mem + 1) % self.max_memory_size

    def train(self):
        if self.memory_len < self.batch_size:
            return
        
        # sample replays without replacement
        transitions = self.memory[torch.randperm(self.memory_len)][:self.batch_size - 1]

        # combined experience replay, append latest replay to sampled batch
        transitions = torch.cat((transitions, self.memory[(self.oldest_mem - 1) % self.memory_len].reshape(1, -1)), 0)

        batch_state = transitions[:,:5]
        batch_actions = transitions[:,5].to(dtype=torch.int64).unsqueeze(-1)
        batch_next_state = transitions[:,6:11]
        batch_reward = transitions[:,11].unsqueeze(-1)
        batch_terminated = transitions[:,12].unsqueeze(-1)
        
        batch_q_state = self.dqn(batch_state).gather(1, batch_actions)

        with torch.no_grad():
            batch_q_next_action = (1 - batch_terminated) * self.df * self.target_net(batch_next_state).max(dim=1, keepdim=True).values

        expected = batch_reward + batch_q_next_action

        loss = self.loss_fn(batch_q_state, expected)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.dqn.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.dqn.state_dict())

    def decay(self):
        self.epsilon = max(self.end_epsilon, self.epsilon - self.epsilon_decay)

    def serialize(self, episodes_dir: os.PathLike, episode_no: int):
        print(f'serializing to file {os.path.join(episodes_dir, f"nn-{episode_no}.pth")}\n')
        torch.save(self.dqn.state_dict(), os.path.join(episodes_dir, f'nn-{episode_no}.pth'))

    def load(self, checkpoint_path: os.PathLike):
        print(f'loading checkpoint {checkpoint_path}')
        self.dqn.load_state_dict(torch.load(checkpoint_path, weights_only=True))