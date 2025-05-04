import numpy as np

class LaneEnv:
    """
    车道保持自定义环境（伪代码，需与CARLA集成）
    """
    def __init__(self):
        # 状态空间：如[横向偏移, 航向角, 速度]
        self.state_dim = 3
        # 动作空间：左转、直行、右转
        self.action_dim = 3
        self.reset()

    def reset(self):
        # 初始化状态
        self.state = np.zeros(self.state_dim)
        return self.state

    def step(self, action):
        """
        执行动作，返回新状态、奖励、done、info
        """
        # 这里应与CARLA通信，获取新状态
        # 伪代码如下
        offset, angle, speed = self.state
        if action == 0:  # 左
            offset -= 0.1
        elif action == 2:  # 右
            offset += 0.1
        # 状态更新
        self.state = np.array([offset, angle, speed])
        # 奖励函数
        reward = -abs(offset)  # 偏离越大惩罚越大
        done = abs(offset) > 1.5  # 保留核心done判据
        info = {}
        return self.state, reward, done, info 