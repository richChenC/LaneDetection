import carla
import numpy as np
import cv2
import time
from drllane_carla_rl.lane_det.detector import LaneDetector
import threading
from drllane_carla_rl.utils.visualize_rich import detect_and_draw_lanes

class CarlaLaneEnv:
    """
    真实端到端的CARLA环境，支持RL训练。
    每一步都用摄像头图像做车道线检测，偏移量作为RL输入。
    """
    def __init__(self, model_path='../../车道线检测/my-model/culane_18.pth', host='localhost', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        try:
            self.world = self.client.get_world()
        except RuntimeError:
            print("[错误] 无法连接到CARLA Simulator，请先启动CARLA并加载地图！")
            exit(1)
        self.map = self.world.get_map()
        self.map_name = self.map.name
        # 自动切换到Town02
        if 'Town02' not in self.map_name:
            print(f"[地图切换] 当前地图为{self.map_name}，正在切换到Town02...")
            self.client.load_world('Town02')
            time.sleep(5)
            self.world = self.client.get_world()
            self.map = self.world.get_map()
            self.map_name = self.map.name
            print(f"[地图切换] 已切换到{self.map_name}")
        else:
            print(f"[地图检测] 当前地图为{self.map_name}")
        # 初始化时清理所有旧资源
        self.cleanup_all_actors()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.image_data = {'frame': None}
        self.lane_detector = LaneDetector(model_path)
        self.state_dim = 4  # [model_offset, angle, speed, official_offset]
        # 优化动作空间设计
        self.steer_choices = [-0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3]
        self.throttle_choices = [0.3, 0.4, 0.5, 0.6, 0.7]
        self.brake_choices = [0.0, 0.5, 1.0]
        self.action_dim = len(self.steer_choices) * len(self.throttle_choices) * len(self.brake_choices)
        self.spawn_idx = 11  # Town02的第12号点，直线路段
        self._setup_vehicle_and_camera()
        self._setup_sensors()
        self.collision = False
        self.lane_invasion = False
        self.start_location = None
        self.episode_start_time = None
        self.offset_history = []  # 新增：记录最近偏移量

    def _setup_vehicle_and_camera(self):
        # 清理旧车辆
        for actor in self.world.get_actors().filter('vehicle.*'):
            actor.destroy()
        # 生成新车辆
        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        spawn_point = self.world.get_map().get_spawn_points()[self.spawn_idx]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        # 车头前无遮挡摄像头（与UI/录制一致）
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '640')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(
            carla.Location(x=0.5, y=0.0, z=1.4),
            carla.Rotation(pitch=-10, yaw=0, roll=0)
        )
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(self._image_callback)
        while self.image_data['frame'] is None:
            time.sleep(0.01)

    def _image_callback(self, image):
        self.image_data['frame'] = image

    def _setup_sensors(self):
        # 碰撞传感器
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        # 车道偏离传感器
        lane_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(lane_bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))

    def _on_collision(self, event):
        self.collision = True

    def _on_lane_invasion(self, event):
        self.lane_invasion = True

    def reset(self):
        self.cleanup_all_actors()
        self._setup_vehicle_and_camera()
        self._setup_sensors()
        self.collision = False
        self.lane_invasion = False
        self.official_invasion_history = []
        self.total_distance = 0.0
        self.last_location = None
        self.start_location = self.vehicle.get_location()
        self.episode_start_time = time.time()
        self.step_count = 0
        self.low_speed_steps = 0
        self.offset_history = []
        img = self._get_camera_image()
        from drllane_carla_rl.utils.visualize_rich import detect_and_draw_lanes
        vis_img, lanes = detect_and_draw_lanes(
            img, self.lane_detector.model, self.lane_detector.img_transforms, self.lane_detector.row_anchor, self.lane_detector.cls_num_per_lane,
            show_green_mask=True, show_hline=False, show_red_lane=True, show_car_center=True
        )
        cv2.namedWindow('Driver View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Driver View', 1280, 640)
        cv2.imshow('Driver View', vis_img)
        cv2.waitKey(1)
        print(f"[Reset] 地图={self.map_name}, spawn={self.spawn_idx}, 车辆位置={self.vehicle.get_location()}, 车道线数={len(lanes)}")
        state = self._get_state(img)
        return state

    def step(self, action):
        self.step_count += 1
        steer_idx = action // (len(self.throttle_choices) * len(self.brake_choices))
        throttle_idx = (action // len(self.brake_choices)) % len(self.throttle_choices)
        brake_idx = action % len(self.brake_choices)
        steer = self.steer_choices[steer_idx]
        throttle = self.throttle_choices[throttle_idx]
        brake = self.brake_choices[brake_idx]
        if self.step_count <= 5:
            throttle = 0.7
            brake = 0.0
        control = carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
        self.vehicle.apply_control(control)
        time.sleep(0.1)
        img = self._get_camera_image()
        from drllane_carla_rl.utils.visualize_rich import detect_and_draw_lanes
        vis_img, lanes = detect_and_draw_lanes(
            img, self.lane_detector.model, self.lane_detector.img_transforms, self.lane_detector.row_anchor, self.lane_detector.cls_num_per_lane,
            show_green_mask=True, show_hline=False, show_red_lane=True, show_car_center=True
        )
        cv2.namedWindow('Driver View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Driver View', 1280, 640)
        cv2.imshow('Driver View', vis_img)
        cv2.waitKey(1)
        cur_location = self.vehicle.get_location()
        if self.last_location is not None:
            self.total_distance += self._calc_distance(self.last_location, cur_location)
        self.last_location = cur_location
        dist = self._calc_distance(self.start_location, cur_location)
        elapsed = time.time() - self.episode_start_time
        official_info = self.get_official_lane_info()
        official_offset = official_info['offset']
        if not hasattr(self, 'official_invasion_history'):
            self.official_invasion_history = []
        self.official_invasion_history.append(official_offset > 1.5)
        if len(self.official_invasion_history) > 3:
            self.official_invasion_history.pop(0)
        state = self._get_state(img)
        reward, done, info = self._get_reward_done(state, steer, throttle, brake)
        if self.total_distance >= 50.0:
            info['reason'] += '|行驶距离达标+2'
            reward += 2.0
            done = True
        if elapsed > 15.0:
            info['reason'] += '|超时15秒自动复位'
            done = True
        if all(self.official_invasion_history[-3:]):
            info['reason'] += '|连续3步官方车道入侵'
            reward -= 5
            done = True
        if self.step_count < 20:
            done = False
        if len(lanes) == 0:
            reward -= 0.1
            info['reason'] = '未检测到车道线，允许继续尝试'
            done = False
        if info.get('collision', False):
            if not hasattr(self, 'try_recovering') or not self.try_recovering:
                self.try_recovering = True
                self.recover_start_time = time.time()
            if state[0] < -0.1:
                action = 2
            elif state[0] > 0.1:
                action = 0
            else:
                action = 1
            if time.time() - self.recover_start_time > 5:
                self.try_recovering = False
                return self.reset(), 0, True, {'reason': '卡死自动复位'}
        else:
            self.try_recovering = False
        # print(f"车辆位置: {self.vehicle.get_location()}, 官方中心: {official_info['center']}, 官方偏移: {official_info['offset']:.3f}, 模型偏移: {state[0]:.3f}")
        return state, reward, done, info

    def _get_camera_image(self):
        while self.image_data['frame'] is None:
            time.sleep(0.01)
        image = self.image_data['frame']
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((image.height, image.width, 4))
        img = img[:, :, :3]  # BGRA->BGR
        return img

    def _get_state(self, img):
        try:
            lanes = self.lane_detector.get_lanes(img)
        except Exception:
            lanes = []
        # 检测不到车道线时直接返回0偏移
        if not lanes or not isinstance(lanes, list):
            model_offset = 0.0
        else:
            model_offset = self._compute_offset(lanes, img.shape)
        speed = self._get_speed()
        # 官方辅助
        official_info = self.get_official_lane_info()
        official_offset = official_info['offset']
        angle = 0.0
        return np.array([model_offset, speed, official_offset, angle], dtype=np.float32)

    def _compute_offset(self, lanes, img_shape):
        # lanes: List[List[Tuple[x, y]]]
        # 去除偏差过大的点
        def filter_outliers(points, threshold=50):
            if not points:
                return points
            xs = np.array([pt[0] for pt in points])
            median = np.median(xs)
            filtered = [pt for pt in points if abs(pt[0] - median) < threshold]
            return filtered
        if not isinstance(lanes, list) or len(lanes) < 2:
            return 0.0
        left = filter_outliers(lanes[0])
        right = filter_outliers(lanes[-1])
        if not left or not right:
            return 0.0
        left_x = np.mean([pt[0] for pt in left])
        right_x = np.mean([pt[0] for pt in right])
        lane_center = (left_x + right_x) / 2
        img_center = img_shape[1] / 2
        offset = (img_center - lane_center) / (img_shape[1] / 2)
        return offset

    def _get_speed(self):
        v = self.vehicle.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2 + v.z**2)
        return speed

    def get_official_lane_info(self):
        # 获取车辆最近的官方车道中心线、左右边界线坐标
        location = self.vehicle.get_location()
        waypoint = self.map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_center = waypoint.transform.location
        lane_id = waypoint.lane_id
        lane_width = waypoint.lane_width
        # 计算车辆中心到官方车道中心线的横向偏移
        dx = location.x - lane_center.x
        dy = location.y - lane_center.y
        offset = np.sqrt(dx*dx + dy*dy)
        # 计算左右边界点
        right_vec = waypoint.transform.get_right_vector()
        left_edge = carla.Location(lane_center.x - right_vec.x * lane_width/2, lane_center.y - right_vec.y * lane_width/2, lane_center.z)
        right_edge = carla.Location(lane_center.x + right_vec.x * lane_width/2, lane_center.y + right_vec.y * lane_width/2, lane_center.z)
        return {
            'center': (lane_center.x, lane_center.y),
            'left': (left_edge.x, left_edge.y),
            'right': (right_edge.x, right_edge.y),
            'lane_id': lane_id,
            'lane_width': lane_width,
            'offset': offset
        }

    def _get_reward_done(self, state, steer=0, throttle=0, brake=0):
        done = False
        model_offset = state[0]
        speed = state[1]
        official_offset = state[2]
        reward = 0.0
        reason = []
        # 1. 基础居中奖励(使用指数函数)
        center_reward = np.exp(-2 * abs(official_offset)) 
        reward += center_reward
        reason.append(f"居中奖励:{center_reward:.2f}")
        # 2. 速度奖励(使用分段函数)
        if 3 < speed < 8:  # 扩大理想速度范围
            speed_reward = 0.3 * (1 - abs(speed - 5.5) / 2.5)  # 以5.5m/s为最佳速度
            reward += speed_reward
            reason.append(f"速度奖励:{speed_reward:.2f}")
        elif speed <= 1.0:
            reward -= 0.3
            reason.append("速度过低-0.3")
        elif speed > 10.0:
            reward -= 0.3
            reason.append("速度过快-0.3")
        # 3. 平滑操作奖励
        if abs(steer) < 0.1:  # 鼓励小角度转向
            reward += 0.1
            reason.append("平稳驾驶+0.1")
        elif abs(steer) > 0.3:  # 惩罚大角度转向
            reward -= 0.1
            reason.append("转向过大-0.1")
        # 4. 距离奖励(使用指数增长)
        if not hasattr(self, 'last_total_distance'):
            self.last_total_distance = self.total_distance
        distance_delta = self.total_distance - self.last_total_distance
        if distance_delta > 0:
            distance_reward = 0.1 * np.exp(distance_delta - 1)
            reward += distance_reward
            reason.append(f"距离奖励:{distance_reward:.2f}")
        self.last_total_distance = self.total_distance
        # 5. 持续行驶奖励
        if distance_delta > 0.1:
            reward += 0.2
            reason.append("持续前进+0.2")
        elif distance_delta < -0.01:
            reward -= 0.3
            reason.append("倒退-0.3")
        # 6. 完成任务大奖励
        if hasattr(self, 'total_distance') and self.total_distance >= 50.0:
            reward += 10.0  # 增加完成奖励
            reason.append("任务完成+10.0")
            done = True
        # 7. 失败惩罚
        if hasattr(self, 'lane_invasion') and self.lane_invasion:
            reward -= 2.0  # 减小惩罚
            reason.append("车道入侵-2.0")
            done = True
        elif hasattr(self, 'collision') and self.collision:
            reward -= 5.0  # 碰撞惩罚保持较大
            reason.append("碰撞-5.0")
            done = True
        elif abs(official_offset) > 2.0:  # 放宽偏离限制
            reward -= 2.0
            reason.append("偏离过大-2.0")
            done = True
        elif hasattr(self, 'low_speed_steps') and self.low_speed_steps > 15:  # 增加容忍步数
            reward -= 1.0
            reason.append("持续低速-1.0")
            done = True
        info = {'reason': '|'.join(reason)}
        return reward, done, info

    def _calc_distance(self, loc1, loc2):
        if loc1 is None or loc2 is None:
            return 0.0
        dx = loc1.x - loc2.x
        dy = loc1.y - loc2.y
        dz = loc1.z - loc2.z
        return np.sqrt(dx*dx + dy*dy + dz*dz)

    def cleanup_all_actors(self):
        # 清理所有车辆和传感器，sensor先stop再destroy，避免stream残留
        actors = self.world.get_actors()
        for actor in actors:
            if actor.type_id.startswith('sensor.'):
                try:
                    if actor.is_listening:
                        actor.stop()
                except Exception:
                    pass
        for actor in actors:
            if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('sensor.'):
                try:
                    actor.destroy()
                except Exception:
                    pass
        # 清空本地引用，防止"sensor object went out of the scope"
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None

    def close(self):
        cv2.destroyAllWindows()
        self.cleanup_all_actors() 