import torch
import numpy as np
import random
import math
import logging
import time
import cv2 # For potential rendering/debugging

from carla_rl_env import CarlaLaneKeepEnv # Assuming the env file is in the same directory
from rl_agent import DQNAgent, ReplayMemory, Transition, discretize_action, get_n_actions

# --- Configuration --- 
BATCH_SIZE = 64          # Number of transitions sampled from the replay buffer
GAMMA = 0.99             # Discount factor for future rewards
EPS_START = 0.9          # Starting value of epsilon (exploration rate)
EPS_END = 0.05           # Minimum value of epsilon
EPS_DECAY = 20000        # Controls the rate of exponential decay of epsilon, higher means slower decay
TARGET_UPDATE = 100      # How often to update the target network (in episodes or steps)
REPLAY_MEMORY_SIZE = 10000 # Size of the replay buffer
LR = 1e-4                # Learning rate for the Adam optimizer
NUM_EPISODES = 500       # Total number of training episodes
MAX_STEPS_PER_EPISODE = 1000 # Maximum steps allowed per episode

# Action discretization parameters (should match rl_agent.py)
STEER_BINS = 5
THROTTLE_BINS = 3
N_ACTIONS = get_n_actions(STEER_BINS, THROTTLE_BINS)

# Environment parameters
CARLA_HOST = 'localhost'
CARLA_PORT = 2000

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Main Training Loop --- 
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    env = None
    try:
        # Initialize environment
        env = CarlaLaneKeepEnv(carla_host=CARLA_HOST, carla_port=CARLA_PORT)
        logger.info("CARLA Environment initialized.")

        # Get state shape from environment
        # Note: Ensure env.reset() returns the correct initial observation shape
        initial_obs = env.reset()
        state_shape = initial_obs.shape
        logger.info(f"Observation space shape: {state_shape}")
        logger.info(f"Action space: {env.action_space}, Discrete actions: {N_ACTIONS}")

        # Initialize agent
        agent = DQNAgent(state_shape=state_shape, n_actions=N_ACTIONS,
                         replay_capacity=REPLAY_MEMORY_SIZE, batch_size=BATCH_SIZE,
                         gamma=GAMMA, lr=LR, target_update=TARGET_UPDATE, device=device)
        logger.info("DQN Agent initialized.")

        episode_rewards = []
        total_steps = 0

        # Training loop
        for i_episode in range(NUM_EPISODES):
            # Initialize the environment and state
            state = env.reset()
            current_episode_reward = 0

            for t in range(MAX_STEPS_PER_EPISODE):
                # Select and perform an action
                epsilon = EPS_END + (EPS_START - EPS_END) * \
                          math.exp(-1. * total_steps / EPS_DECAY)
                action_index_tensor = agent.select_action(state, epsilon)
                action_index = action_index_tensor.item()
                continuous_action = discretize_action(action_index, STEER_BINS, THROTTLE_BINS)

                next_state, reward, done, info = env.step(continuous_action)
                current_episode_reward += reward
                total_steps += 1

                # Store the transition in memory
                # Ensure reward is a tensor
                reward_tensor = torch.tensor([reward], device=device)
                agent.memory.push(state, action_index_tensor, next_state if not done else None, reward_tensor)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                agent.optimize_model()

                # Optional: Render or log step info
                # logger.debug(f"Episode {i_episode+1}, Step {t+1}, Action: {continuous_action}, Reward: {reward:.2f}, Done: {done}")
                # cv2.imshow("Training Observation", state)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     raise KeyboardInterrupt # Allow quitting

                if done:
                    break

            episode_rewards.append(current_episode_reward)
            logger.info(f"Episode {i_episode + 1}/{NUM_EPISODES} finished after {t + 1} steps. Reward: {current_episode_reward:.2f}. Epsilon: {epsilon:.3f}")

            # Update the target network, copying all weights and biases in DQN
            # Update based on episodes instead of steps for simplicity here
            if i_episode % agent.target_update == 0:
                 agent.target_net.load_state_dict(agent.policy_net.state_dict())
                 logger.info(f"Updated target network at episode {i_episode + 1}")

            # Optional: Save model periodically
            # if i_episode % 50 == 0:
            #     torch.save(agent.policy_net.state_dict(), f"dqn_policy_net_episode_{i_episode}.pth")
            #     logger.info(f"Saved model checkpoint at episode {i_episode}")

        logger.info('Training complete')
        # TODO: Add plotting of rewards

    except ConnectionError as e:
        logger.critical(f"Connection Error: {e}")
    except RuntimeError as e:
        logger.critical(f"CARLA Runtime Error: {e}")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during training: {e}")
    finally:
        if env:
            env.close()
            logger.info("CARLA Environment closed.")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()