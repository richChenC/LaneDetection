import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQNNet(nn.Module):
    """
    简单的全连接DQN网络
    """
    def __init__(self, state_dim, action_dim):
        super(DQNNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    """
    DQN智能体，支持训练与推理
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, memory_size=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNet(state_dim, action_dim).to(self.device)
        self.target_net = DQNNet(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.update_count = 0
        self.reward_norm_mean = 0.0
        self.reward_norm_std = 1.0
        self.human_feedback = 0.0

    def select_action(self, state):
        """
        ε-贪婪策略选择动作
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def add_human_feedback(self, value):
        """
        外部接口：添加人类反馈奖励（如人工点击好/坏）
        """
        self.human_feedback += value

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        # 奖励归一化
        rewards_np = np.array(rewards)
        self.reward_norm_mean = rewards_np.mean()
        self.reward_norm_std = rewards_np.std() if rewards_np.std() > 1e-6 else 1.0
        rewards_norm = (rewards_np - self.reward_norm_mean) / self.reward_norm_std
        # 加入人类反馈
        rewards_norm += self.human_feedback
        self.human_feedback = 0.0
        rewards = torch.FloatTensor(rewards_norm).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1, keepdim=True)[0]
            q_target = rewards + self.gamma * q_next * (1 - dones)
        loss = self.loss_fn(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新target网络
        self.update_count += 1
        if self.update_count % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # 衰减epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay 