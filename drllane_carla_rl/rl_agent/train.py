import yaml
import numpy as np
import torch
from drllane_carla_rl.rl_agent.agent import DQNAgent
from drllane_carla_rl.rl_agent.carla_env import CarlaLaneEnv
import os
import logging
import time
import sys
import os
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from drllane_carla_rl.lane_det.ultrafastlane.model import parsingNet
from drllane_carla_rl.utils.visualize_rich import detect_and_draw_lanes
import carla
import cv2

# 屏蔽carla的INFO和WARNING日志
logging.getLogger('carla').setLevel(logging.ERROR)

# 只过滤掉特定字符串的输出
class FilteredStdout:
    def __init__(self, original):
        self.original = original
    def write(self, s):
        if "streaming client: failed to read header: End of file" not in s:
            self.original.write(s)
    def flush(self):
        self.original.flush()

sys.stdout = FilteredStdout(sys.stdout)
sys.stderr = FilteredStdout(sys.stderr)

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'train.log')
    logger = logging.getLogger(f'exp_logger_{log_dir}')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    return logger

def save_log_to_file(logger, target_file):
    """
    将当前日志文件内容复制到目标文件（如log_ep50.txt）。
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
            src = handler.baseFilename
            shutil.copyfile(src, target_file)
            break

def train():
    config = load_config(os.path.join(os.path.dirname(__file__), '../configs/config.yaml'))
    num_episodes = 500
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(r'G:/Graduation_Design/1cursor-Lane_Detection-study-main/drllane_carla_rl/model', f'exp_{timestamp}')
    log_dir = os.path.join(r'G:/Graduation_Design/1cursor-Lane_Detection-study-main/drllane_carla_rl/log', f'exp_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'log_ep0.txt')
    logger = setup_logger(log_dir)
    logger.info('训练开始')

    model_path = r'G:\Graduation_Design\1cursor-Lane_Detection-study-main\车道线检测\my-model\culane_18.pth'
    env = CarlaLaneEnv(model_path=model_path)
    agent = DQNAgent(
        state_dim=4,
        action_dim=env.action_dim,
        lr=config['train']['lr'],
        gamma=config['train']['gamma'],
        epsilon=config['train']['epsilon_start'],
        epsilon_min=config['train']['epsilon_min'],
        epsilon_decay=config['train']['epsilon_decay'],
        memory_size=config['train']['memory_size'],
        batch_size=config['train']['batch_size']
    )
    episode_rewards = []
    start_time = time.time()
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        total_reward = 0
        step_count = 0
        ep_start_time = time.time()
        exit_reason = ''
        for step in range(config['train']['max_steps']):
            if step < 10:
                action = env.action_dim // 2  # 直行
            else:
                action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
            step_count += 1
            if done:
                # 只允许五种退出原因，未达到这些条件时不done
                if hasattr(env, 'lane_invasion') and env.lane_invasion:
                    exit_reason = '车道入侵'
                elif hasattr(env, 'collision') and env.collision:
                    exit_reason = '碰撞'
                elif abs(state[2]) > 1.5:
                    exit_reason = '偏离过大'
                elif hasattr(env, 'total_distance') and env.total_distance >= 50.0:
                    exit_reason = '到达终点'
                elif hasattr(env, 'low_speed_steps') and env.low_speed_steps > 10:
                    exit_reason = '卡死'
                else:
                    # 不done，继续动态调整
                    done = False
                    continue
                break
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else total_reward
        episode_rewards.append(total_reward)
        ep_time = time.time() - ep_start_time
        print(f"Episode {ep}/{num_episodes} 步数:{step_count} 奖励:{total_reward:.2f} 平均奖励(10):{avg_reward:.2f} 用时:{ep_time:.1f}s 退出原因:{exit_reason}")
        logger.info(f"Episode {ep}/{num_episodes} 步数:{step_count} 奖励:{total_reward:.2f} 平均奖励(10):{avg_reward:.2f} 用时:{ep_time:.1f}s 退出原因:{exit_reason}")
        # 每50轮保存模型和日志
        if ep % 50 == 0:
            model_path = os.path.join(model_dir, f'model_ep{ep}.pth')
            torch.save(agent.policy_net.state_dict(), model_path)
            log_file = os.path.join(log_dir, f'log_ep{ep}.txt')
            save_log_to_file(logger, log_file)
    # 最后保存一次
    model_path = os.path.join(model_dir, f'model_ep{num_episodes}.pth')
    torch.save(agent.policy_net.state_dict(), model_path)
    log_file = os.path.join(log_dir, f'log_ep{num_episodes}.txt')
    save_log_to_file(logger, log_file)
    total_time = time.time() - start_time
    logger.info(f'训练结束，总时长: {total_time:.1f}秒，平均奖励: {np.mean(episode_rewards):.2f}')
    print(f'训练结束，总时长: {total_time:.1f}秒，平均奖励: {np.mean(episode_rewards):.2f}')
    env.close()

if __name__ == '__main__':
    print("训练开始")
    train() 
    print("训练结束")