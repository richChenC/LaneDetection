import os
import sys
import time
import queue
import threading
import numpy as np
import cv2
import torch
import carla
from PIL import Image
import torchvision.transforms as transforms
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import logging
import urllib.request
import zipfile
import shutil
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """系统配置类"""
    # 视频参数
    WIDTH: int = 1280
    HEIGHT: int = 720
    FPS: int = 30
    RECORD_SECONDS: int = 30
    
    # CARLA参数
    CARLA_HOST: str = 'localhost'
    CARLA_PORT: int = 2000
    CARLA_TIMEOUT: float = 10.0
    
    # 输出路径
    OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "my-video", "carla-vedio", "output")
    
    # 模型参数
    MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "model", "culane_18.pth")
    ROW_ANCHOR: np.ndarray = np.array([121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287])

    # 编码器设置
    CODEC_PRIORITY: List[Dict] = [
        {'codec': 'avc1', 'ext': '.mp4'},
        {'codec': 'h264', 'ext': '.mp4'},
        {'codec': 'XVID', 'ext': '.avi'},
        {'codec': 'MJPG', 'ext': '.avi'},
        {'codec': 'DIV3', 'ext': '.avi'},
        {'codec': 'MP42', 'ext': '.avi'}
    ]

def setup_openh264():
    """设置OpenH264库"""
    try:
        openh264_dir = os.path.join(os.path.expanduser("~"), ".openh264")
        os.makedirs(openh264_dir, exist_ok=True)
        
        dll_path = os.path.join(openh264_dir, "openh264-1.8.0-win64.dll")
        if not os.path.exists(dll_path):
            logger.info("下载OpenH264库...")
            url = "https://github.com/cisco/openh264/releases/download/v1.8.0/openh264-1.8.0-win64.dll.bz2"
            urllib.request.urlretrieve(url, dll_path + ".bz2")
            
            import bz2
            with bz2.open(dll_path + ".bz2", "rb") as f_in:
                with open(dll_path, "wb") as f_out:
                    f_out.write(f_in.read())
            
            os.remove(dll_path + ".bz2")
            
        os.environ["PATH"] = openh264_dir + os.pathsep + os.environ["PATH"]
        return True
    except Exception as e:
        logger.warning(f"OpenH264设置失败: {str(e)}")
        return False

class FrameProcessor:
    """帧处理类"""
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self._load_model()
        self.transforms = self._setup_transforms()
        
    def _load_model(self):
        """加载车道线检测模型"""
        try:
            logger.info(f'加载模型: {self.config.MODEL_PATH}')
            
            state_dict = torch.load(self.config.MODEL_PATH, map_location=self.device)
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            
            # 获取模型维度
            cls_dim = None
            for k, v in state_dict.items():
                if 'cls.2.weight' in k:
                    cls_dim = v.shape[0]
                    break
            
            if cls_dim is None:
                raise RuntimeError("无法获取模型维度")
            
            from model.model import parsingNet
            net = parsingNet(pretrained=False, backbone='18', cls_dim=cls_dim, use_aux=False).to(self.device)
            net.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
            net.eval()
            return net
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
            
    def _setup_transforms(self):
        """设置图像预处理"""
        return transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """处理单帧图像"""
        try:
            # 确保帧是可写的
            frame = frame.copy()
            
            # 转换图像格式
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = self.transforms(img_pil).unsqueeze(0).to(self.device)
            
            # 模型推理
            with torch.no_grad():
                out = self.net(img_tensor)
            
            # 处理输出
            out_j = out[0].data.cpu().numpy()
            out_j = out_j[:, ::-1, :]
            prob = torch.nn.functional.softmax(torch.tensor(out_j[:-1, :, :]), dim=0).numpy()
            idx = np.arange(out_j.shape[0]) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == out_j.shape[0]] = 0
            
            # 可视化结果
            result_img = frame.copy()
            lanes = []
            lane_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)] # BGR for OpenCV
            
            # 提取车道线点
            for i in range(out_j.shape[1]):
                lane_points = []
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            # 原始坐标计算 (基于800x288)
                            raw_x = out_j[k, i]
                            raw_y = self.config.ROW_ANCHOR[out_j.shape[0]-1-k]
                            # 映射到当前帧尺寸
                            x = int(raw_x * frame.shape[1] / 800)
                            y = int(frame.shape[0] * (raw_y / 288))
                            # 确保坐标在图像范围内
                            x = max(0, min(frame.shape[1] - 1, x))
                            y = max(0, min(frame.shape[0] - 1, y))
                            lane_points.append((x, y))
                if lane_points:
                    lanes.append(np.array(lane_points, dtype=np.int32))

            # --- 开始集成可视化逻辑 ---
            overlay_mask = result_img.copy() # 用于绿色薄膜
            
            # 确定当前车道 (简化逻辑：假设中间两条是当前车道)
            # 注意：这个逻辑可能需要根据实际情况调整
            center_x = result_img.shape[1] // 2
            lane_index = 1 # 默认第一条车道
            if len(lanes) >= 2:
                # 简单的基于x坐标的排序，找到靠近中心的两条线
                lanes.sort(key=lambda lane: np.mean(lane[:, 0]))
                min_dist_idx = -1
                min_dist = float('inf')
                for i in range(len(lanes) - 1):
                    dist_left = abs(np.mean(lanes[i][:, 0]) - center_x)
                    dist_right = abs(np.mean(lanes[i+1][:, 0]) - center_x)
                    if lanes[i][-1][0] < center_x < lanes[i+1][-1][0]: # 确保车辆在两条线之间
                         if dist_left + dist_right < min_dist:
                             min_dist = dist_left + dist_right
                             min_dist_idx = i
                if min_dist_idx != -1:
                    lane_index = min_dist_idx + 1 # lane_index 从1开始

            # 绘制绿色可通行区域
            if len(lanes) >= 2 and lane_index <= len(lanes) - 1:
                left_lane = lanes[lane_index - 1]
                right_lane = lanes[lane_index]
                
                if left_lane.size > 0 and right_lane.size > 0:
                    # 合并左右车道点以创建多边形
                    poly_pts = np.concatenate((left_lane, right_lane[::-1]), axis=0)
                    if poly_pts.shape[0] >= 3:
                        cv2.fillPoly(overlay_mask, [poly_pts.reshape((-1, 1, 2))], (0, 255, 0)) # 绿色
                        # 叠加绿色薄膜
                        result_img = cv2.addWeighted(result_img, 0.7, overlay_mask, 0.3, 0)

            # 绘制红色车道线点
            for i, lane in enumerate(lanes):
                color = lane_colors[i % len(lane_colors)]
                for point in lane:
                    cv2.circle(result_img, tuple(point), 5, (0, 0, 255), -1) # 红色点
                # 可选：绘制车道线轮廓
                # if len(lane) > 1:
                #     cv2.polylines(result_img, [lane], False, color, 2)
            # --- 结束集成可视化逻辑 ---

            return result_img, lanes
            
        except Exception as e:
            logger.error(f"帧处理失败: {str(e)}")
            return frame.copy(), []

class VideoRecorder:
    """视频录制类"""
    def __init__(self, config: Config):
        self.config = config
        self.frame_queue = queue.Queue(maxsize=120)
        self.writer = None
        self.recording = False
        self._setup_output()
        
    def _setup_output(self):
        """设置输出目录和文件"""
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.video_path = None
        self.timestamp = timestamp
        
    def _try_create_writer(self, codec_info):
        """尝试创建视频写入器"""
        try:
            path = os.path.join(self.config.OUTPUT_DIR, f"carla_record_{self.timestamp}{codec_info['ext']}")
            fourcc = cv2.VideoWriter_fourcc(*codec_info['codec'])
            writer = cv2.VideoWriter(
                path,
                fourcc,
                self.config.FPS,
                (self.config.WIDTH, self.config.HEIGHT)
            )
            if writer.isOpened():
                self.video_path = path
                return writer
        except Exception as e:
            logger.warning(f"编码器 {codec_info['codec']} 初始化失败: {str(e)}")
        return None
        
    def start(self):
        """开始录制"""
        if self.recording:
            return
            
        # 尝试不同的编码器
        for codec_info in self.config.CODEC_PRIORITY:
            self.writer = self._try_create_writer(codec_info)
            if self.writer is not None:
                logger.info(f"使用编码器 {codec_info['codec']} 录制到: {self.video_path}")
                break
                
        if self.writer is None:
            raise RuntimeError("无法初始化任何视频编码器")
            
        self.recording = True
        
    def stop(self):
        """停止录制"""
        self.recording = False
        if self.writer:
            self.writer.release()
            
    def add_frame(self, frame: np.ndarray):
        """添加帧到队列"""
        try:
            if self.recording and not self.frame_queue.full():
                # 确保帧是连续的内存
                self.frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass
            
    def process_frames(self):
        """处理帧队列"""
        frame_count = 0
        while self.recording:
            try:
                frame = self.frame_queue.get(timeout=0.5)
                if frame is not None:
                    self.writer.write(frame)
                    frame_count += 1
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"帧处理错误: {str(e)}")
                break
        logger.info(f"录制完成，共处理 {frame_count} 帧")

class CarlaInterface:
    """CARLA接口类"""
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        
    def connect(self):
        """连接到CARLA服务器"""
        try:
            self.client = carla.Client(self.config.CARLA_HOST, self.config.CARLA_PORT)
            self.client.set_timeout(self.config.CARLA_TIMEOUT)
            self.world = self.client.get_world()
            
            # 设置同步模式
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / self.config.FPS
            self.world.apply_settings(settings)
            
            return True
        except Exception as e:
            logger.error(f"CARLA连接失败: {str(e)}")
            return False
            
    def setup_vehicle(self):
        """设置车辆"""
        try:
            # 生成特斯拉
            blueprint = self.world.get_blueprint_library().find('vehicle.tesla.model3')
            spawn_points = self.world.get_map().get_spawn_points()
            self.vehicle = self.world.spawn_actor(blueprint, spawn_points[0])
            self.vehicle.set_autopilot(True)
            return True
        except Exception as e:
            logger.error(f"车辆设置失败: {str(e)}")
            return False
            
    def setup_camera(self):
        """设置相机"""
        try:
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.config.WIDTH))
            camera_bp.set_attribute('image_size_y', str(self.config.HEIGHT))
            camera_bp.set_attribute('fov', '110')
            
            camera_transform = carla.Transform(
                carla.Location(x=1.5, z=2.4),
                carla.Rotation(pitch=-5)
            )
            
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            return True
        except Exception as e:
            logger.error(f"相机设置失败: {str(e)}")
            return False
            
    def cleanup(self):
        """清理资源"""
        try:
            if self.camera:
                self.camera.stop()
                self.camera.destroy()
            if self.vehicle:
                self.vehicle.destroy()
            if self.world:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
        except Exception as e:
            logger.error(f"清理资源失败: {str(e)}")

class LaneDetectionSystem:
    """车道线检测系统主类"""
    def __init__(self):
        self.config = Config()
        self.frame_processor = None
        self.video_recorder = None
        self.carla_interface = None
        
        self.stop_flag = threading.Event()
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.start_time = None
        
        # 创建显示窗口
        cv2.namedWindow('Raw', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Lane Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Raw', self.config.WIDTH//2, self.config.HEIGHT//2)
        cv2.resizeWindow('Lane Detection', self.config.WIDTH//2, self.config.HEIGHT//2)
        
    def _create_info_overlay(self, fps: float, elapsed_time: float) -> np.ndarray:
        """创建信息显示层"""
        info_img = np.zeros((90, 250, 3), dtype=np.uint8)
        cv2.putText(info_img, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_img, f"Time: {int(elapsed_time)}s",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return info_img
        
    def process_image(self, image):
        """处理CARLA相机图像"""
        try:
            # 转换图像格式
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((self.config.HEIGHT, self.config.WIDTH, 4))
            frame = array[:, :, :3].copy()
            
            # 处理帧
            result_frame, lanes = self.frame_processor.process_frame(frame)
            
            # 添加FPS和时间信息
            current_time = time.time()
            self.frame_count += 1
            elapsed_time = current_time - self.start_time
            
            if current_time - self.last_fps_time >= 1.0:
                fps = self.frame_count / (current_time - self.last_fps_time)
                info_img = self._create_info_overlay(fps, elapsed_time)
                
                # 叠加信息层
                frame[5:95, 5:255] = cv2.addWeighted(frame[5:95, 5:255], 0.5, info_img, 0.5, 0)
                result_frame[5:95, 5:255] = cv2.addWeighted(result_frame[5:95, 5:255], 0.5, info_img, 0.5, 0)
                
                self.frame_count = 0
                self.last_fps_time = current_time
            
            # 显示画面
            cv2.imshow('Raw', frame)
            cv2.imshow('Lane Detection', result_frame)
            
            # 录制处理后的帧
            self.video_recorder.add_frame(result_frame)
            
            # 控制帧率
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_flag.set()
            
        except Exception as e:
            logger.error(f"图像处理错误: {str(e)}")
        
    def run(self):
        """运行系统"""
        try:
            logger.info("启动车道线检测系统...")
            
            # 设置OpenH264
            setup_openh264()
            
            # 初始化组件
            self.frame_processor = FrameProcessor(self.config)
            self.video_recorder = VideoRecorder(self.config)
            self.carla_interface = CarlaInterface(self.config)
            
            # 初始化CARLA
            if not self.carla_interface.connect():
                return
            if not self.carla_interface.setup_vehicle():
                return
            if not self.carla_interface.setup_camera():
                return
                
            # 启动录制
            self.video_recorder.start()
            recorder_thread = threading.Thread(target=self.video_recorder.process_frames)
            recorder_thread.start()
            
            # 启动相机
            self.carla_interface.camera.listen(self.process_image)
            
            # 记录开始时间
            self.start_time = time.time()
            logger.info(f"开始录制，按'Q'键退出...")
            
            # 主循环
            while not self.stop_flag.is_set():
                if time.time() - self.start_time >= self.config.RECORD_SECONDS:
                    break
                self.carla_interface.world.tick()
                time.sleep(0.001)
                    
        except Exception as e:
            logger.error(f"系统运行错误: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            # 清理资源
            self.stop_flag.set()
            self.video_recorder.stop()
            self.carla_interface.cleanup()
            cv2.destroyAllWindows()
            
            if 'recorder_thread' in locals():
                recorder_thread.join()
                
            if self.video_recorder.video_path:
                logger.info(f"录制完成！视频已保存到: {self.video_recorder.video_path}")

def main():
    system = LaneDetectionSystem()
    system.run()

if __name__ == '__main__':
    main()