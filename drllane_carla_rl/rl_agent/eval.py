import yaml
import numpy as np
import torch
from drllane_carla_rl.rl_agent.agent import DQNAgent
from drllane_carla_rl.rl_agent.env import LaneEnv
import os

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def evaluate(model_path):
    config = load_config(os.path.join(os.path.dirname(__file__), '../configs/config.yaml'))
    env = LaneEnv()
    agent = DQNAgent(
        state_dim=config['env']['state_dim'],
        action_dim=config['env']['action_dim']
    )
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.policy_net.eval()
    agent.epsilon = 0.0  # 评估时不探索

    rewards = []
    for episode in range(config['eval']['episodes']):
        state = env.reset()
        total_reward = 0
        for step in range(config['eval']['max_steps']):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                break
        rewards.append(total_reward)
        print(f"评估Episode {episode+1}/{config['eval']['episodes']} 总奖励: {total_reward:.2f}")
    print(f"平均奖励: {np.mean(rewards):.2f}")

if __name__ == '__main__':
    # 默认评估最新模型
    model_path = os.path.join(os.path.dirname(__file__), 'dqn_500.pth')
    evaluate(model_path) 