import carla
import cv2
import numpy as np
import time
import os
from drllane_carla_rl.lane_det.detector import LaneDetector
from drllane_carla_rl.rl_agent.agent import DQNAgent
from drllane_carla_rl.rl_agent.env import LaneEnv

def get_camera_image(sensor_data):
    """
    将CARLA相机数据转为OpenCV格式
    """
    img = np.frombuffer(sensor_data.raw_data, dtype=np.uint8)
    img = img.reshape((sensor_data.height, sensor_data.width, 4))
    img = img[:, :, :3][:, :, ::-1]  # BGRA->BGR
    return img

def compute_offset(lanes, img_shape):
    """
    计算车辆与车道中心的横向偏移（简化版）
    """
    if len(lanes) < 2:
        return 0.0
    left_x = np.mean(lanes[0][:, 0])
    right_x = np.mean(lanes[-1][:, 0])
    lane_center = (left_x + right_x) / 2
    img_center = img_shape[1] / 2
    offset = (img_center - lane_center) / (img_shape[1] / 2)  # 归一化
    return offset

def main():
    # 初始化CARLA客户端
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 选择车辆和相机传感器
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '288')
    camera_bp.set_attribute('fov', '110')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # 初始化检测器和RL智能体
    model_path = '../../车道线检测/my-model/culane_18.pth'
    detector = LaneDetector(model_path)
    env = LaneEnv()
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    # 注册回调获取图像
    image_data = {'frame': None}
    def image_callback(image):
        image_data['frame'] = image
    camera.listen(image_callback)

    try:
        state = env.reset()
        while True:
            if image_data['frame'] is None:
                time.sleep(0.01)
                continue
            img = get_camera_image(image_data['frame'])
            lanes, lane_mask = detector.detect(img)
            offset = compute_offset(lanes, img.shape)
            # 这里可加入航向角、速度等信息
            state = np.array([offset, 0.0, 0.0])
            action = agent.select_action(state)
            # 控制车辆
            control = carla.VehicleControl()
            if action == 0:  # 左
                control.steer = -0.3
            elif action == 1:  # 直行
                control.steer = 0.0
            else:  # 右
                control.steer = 0.3
            control.throttle = 0.4
            vehicle.apply_control(control)
            # 可视化
            vis = detector.visualize(img, lanes, lane_mask)
            cv2.imshow('Lane Detection RL', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.stop()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 