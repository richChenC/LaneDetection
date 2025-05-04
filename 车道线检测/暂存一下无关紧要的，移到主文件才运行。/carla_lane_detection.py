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
import subprocess

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
    
    # 输出路径和文件名
    OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "my-video", "carla-vedio", "output")
    VIDEO_NAME_PREFIX: str = 'carla_record_'
    
    # 模型参数
    MODEL_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'cursor-lane-and-carla', 'model', 'lane-model', 'culane_18.pth')
    
    def __post_init__(self):
        """初始化可变默认值"""
        # 行锚点
        self.ROW_ANCHOR: np.ndarray = np.array([121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287])
        
        # 设置输出文件路径
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        existings = [f for f in os.listdir(self.OUTPUT_DIR) 
                    if f.startswith(self.VIDEO_NAME_PREFIX) and 
                    (f.endswith('.avi') or f.endswith('.mp4'))]
        nums = [int(f[len(self.VIDEO_NAME_PREFIX):-4]) 
               for f in existings if f[len(self.VIDEO_NAME_PREFIX):-4].isdigit()]
        next_num = max(nums) + 1 if nums else 1
        
        self.AVI_FILENAME = f"{self.VIDEO_NAME_PREFIX}{next_num:02d}.avi"
        self.MP4_FILENAME = f"{self.VIDEO_NAME_PREFIX}{next_num:02d}.mp4"
        self.AVI_PATH = os.path.join(self.OUTPUT_DIR, self.AVI_FILENAME)
        self.MP4_PATH = os.path.join(self.OUTPUT_DIR, self.MP4_FILENAME)
        
        # 编码器设置
        self.CODEC_PRIORITY: List[Dict] = [
            {'codec': 'mp4v', 'ext': '.mp4'},
            {'codec': 'avc1', 'ext': '.mp4'},
            {'codec': 'h264', 'ext': '.mp4'},
            {'codec': 'XVID', 'ext': '.avi'},
            {'codec': 'MJPG', 'ext': '.avi'},
            {'codec': 'DIV3', 'ext': '.avi'},
            {'codec': 'MP42', 'ext': '.avi'}
        ]
        
        # 验证模型文件是否存在
        if not os.path.exists(self.MODEL_PATH):
            # 尝试在当前目录查找
            local_model_path = os.path.join(os.path.dirname(__file__), "model", "culane_18.pth")
            if os.path.exists(local_model_path):
                self.MODEL_PATH = local_model_path
            else:
                raise FileNotFoundError(f"找不到模型文件。已尝试路径:\n1. {self.MODEL_PATH}\n2. {local_model_path}")

def setup_openh264():
    """设置OpenH264库"""
    try:
        openh264_dir = os.path.join(os.path.expanduser("~"), ".openh264")
        os.makedirs(openh264_dir, exist_ok=True)
        
        dll_path = os.path.join(openh264_dir, "openh264-1.8.0-win64.dll")
        if not os.path.exists(dll_path):
            logger.info("下载OpenH264库...")
            # 使用备用下载链接
            urls = [
                "https://github.com/cisco/openh264/releases/download/v1.8.0/openh264-1.8.0-win64.dll.bz2",
                "https://mirrors.tuna.tsinghua.edu.cn/github-release/cisco/openh264/v1.8.0/openh264-1.8.0-win64.dll.bz2"
            ]
            
            for url in urls:
                try:
                    urllib.request.urlretrieve(url, dll_path + ".bz2")
                    break
                except Exception as e:
                    logger.warning(f"从 {url} 下载失败: {str(e)}")
                    continue
            else:
                logger.warning("所有下载源均失败，跳过OpenH264设置")
                return False
            
            try:
                import bz2
                with bz2.open(dll_path + ".bz2", "rb") as f_in:
                    with open(dll_path, "wb") as f_out:
                        f_out.write(f_in.read())
                os.remove(dll_path + ".bz2")
            except Exception as e:
                logger.warning(f"解压OpenH264失败: {str(e)}")
                return False
            
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
            
            if not os.path.exists(self.config.MODEL_PATH):
                raise FileNotFoundError(f"模型文件不存在: {self.config.MODEL_PATH}")
            
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
            
            logger.info(f"使用cls_dim: {cls_dim}")
            
            from model.model import parsingNet
            net = parsingNet(pretrained=False, backbone='18', cls_dim=cls_dim, use_aux=False).to(self.device)
            net.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
            net.eval()
            
            logger.info("模型加载完成")
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
            
            # 使用softmax处理概率
            prob = torch.nn.functional.softmax(torch.tensor(out_j[:-1, :, :]), dim=0).numpy()
            
            # 计算位置
            idx = np.arange(out_j.shape[0]) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            
            # 获取最大值索引
            max_indices = np.argmax(out_j, axis=0)
            loc[max_indices == out_j.shape[0]] = 0
            
            # 可视化结果
            result_img = frame.copy()
            lanes = []
            lane_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
            
            # 处理每条车道线
            for i in range(loc.shape[1]):
                points = []
                if np.sum(loc[:, i] != 0) > 2:
                    for k in range(loc.shape[0]):
                        if loc[k, i] > 0:
                            x = int(loc[k, i] * frame.shape[1] / 800)
                            y = int(frame.shape[0] * (self.config.ROW_ANCHOR[k] / 288))
                            points.append((x, y))
                            cv2.circle(result_img, (x, y), 4, lane_colors[i % len(lane_colors)], -1)
                
                if len(points) > 1:
                    points = np.array(points)
                    cv2.polylines(result_img, [points], False, lane_colors[i % len(lane_colors)], 2)
                    lanes.append(points)
            
            return result_img, lanes
            
        except Exception as e:
            logger.error(f"帧处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame.copy(), []

class VideoRecorder:
    """视频录制类"""
    def __init__(self, config: Config):
        self.config = config
        self.frame_queue = queue.Queue(maxsize=60)  # 减小队列大小，防止内存占用过大
        self.writer_avi = None
        self.writer_mp4 = None
        self.recording = False
        self.frame_count = 0
        
    def start(self):
        """开始录制"""
        if self.recording:
            return
            
        try:
            # 创建AVI写入器
            fourcc_avi = cv2.VideoWriter_fourcc(*'XVID')
            self.writer_avi = cv2.VideoWriter(
                self.config.AVI_PATH,
                fourcc_avi,
                self.config.FPS,
                (self.config.WIDTH, self.config.HEIGHT),
                True
            )
            
            # 创建MP4写入器
            fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer_mp4 = cv2.VideoWriter(
                self.config.MP4_PATH,
                fourcc_mp4,
                self.config.FPS,
                (self.config.WIDTH, self.config.HEIGHT),
                True
            )
            
            if not self.writer_avi.isOpened() or not self.writer_mp4.isOpened():
                raise RuntimeError("无法初始化视频写入器")
                
            logger.info(f"开始录制视频:\nAVI: {self.config.AVI_PATH}\nMP4: {self.config.MP4_PATH}")
            self.recording = True
            
        except Exception as e:
            logger.error(f"视频录制初始化失败: {str(e)}")
            self.cleanup()
            raise
            
    def stop(self):
        """停止录制"""
        self.recording = False
        self.cleanup()
        
    def cleanup(self):
        """清理资源"""
        if self.writer_avi:
            self.writer_avi.release()
        if self.writer_mp4:
            self.writer_mp4.release()
            
        if self.frame_count > 0:
            logger.info(f"录制完成，共处理 {self.frame_count} 帧")
            self._convert_to_mp4()
            
    def _convert_to_mp4(self):
        """使用FFmpeg转换为高兼容性MP4"""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', self.config.AVI_PATH,
                '-vcodec', 'libx264',
                '-acodec', 'aac',
                self.config.MP4_PATH
            ]
            subprocess.run(cmd, check=True)
            logger.info(f"已转换为高兼容性MP4: {self.config.MP4_PATH}")
        except Exception as e:
            logger.error(f"MP4转换失败: {str(e)}")
            
    def add_frame(self, frame: np.ndarray):
        """添加帧到队列"""
        try:
            if self.recording and frame is not None:
                if self.writer_avi and self.writer_mp4:
                    self.writer_avi.write(frame)
                    self.writer_mp4.write(frame)
                    self.frame_count += 1
        except Exception as e:
            logger.error(f"帧写入失败: {str(e)}")
            
    def process_frames(self):
        """处理帧队列（此方法保留但不再使用）"""
        pass

class CarlaInterface:
    """CARLA接口类"""
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        
    def connect(self, max_retries=3):
        """连接到CARLA服务器，支持重试"""
        for attempt in range(max_retries):
            try:
                logger.info(f"尝试连接CARLA服务器 (第{attempt + 1}次尝试)...")
                logger.info(f"连接地址: {self.config.CARLA_HOST}:{self.config.CARLA_PORT}")
                
                # 检查CARLA进程
                self._check_carla_process()
                
                # 尝试连接
                self.client = carla.Client(self.config.CARLA_HOST, self.config.CARLA_PORT)
                self.client.set_timeout(self.config.CARLA_TIMEOUT)
                
                # 验证连接
                version = self.client.get_server_version()
                logger.info(f"已连接到CARLA服务器 (版本 {version})")
                
                self.world = self.client.get_world()
                
                # 设置同步模式
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 1.0 / self.config.FPS
                self.world.apply_settings(settings)
                
                return True
                
            except Exception as e:
                if "time-out" in str(e):
                    logger.error(f"连接超时 (第{attempt + 1}次尝试)")
                    if attempt < max_retries - 1:
                        logger.info("等待5秒后重试...")
                        time.sleep(5)
                    else:
                        logger.error(f"CARLA连接失败: {str(e)}")
                        logger.error("请检查:")
                        logger.error("1. CARLA服务器是否已启动")
                        logger.error("2. 端口号是否正确 (默认2000)")
                        logger.error("3. 是否有其他程序占用了端口")
                        logger.error("4. 防火墙设置是否允许连接")
                else:
                    logger.error(f"CARLA连接错误: {str(e)}")
                    break
        return False
        
    def _check_carla_process(self):
        """检查CARLA进程是否运行"""
        try:
            import psutil
            carla_running = False
            for proc in psutil.process_iter(['name']):
                if 'carla' in proc.info['name'].lower():
                    carla_running = True
                    logger.info(f"检测到CARLA进程: {proc.info['name']}")
                    break
            if not carla_running:
                logger.warning("未检测到运行中的CARLA进程！")
                logger.info("请先启动CARLA服务器，通常位于：")
                logger.info("Windows: <CARLA_ROOT>/CarlaUE4.exe")
                logger.info("Linux: <CARLA_ROOT>/CarlaUE4.sh")
        except Exception as e:
            logger.warning(f"进程检查失败: {str(e)}")
            
    def setup_vehicle(self):
        """设置车辆"""
        try:
            # 清理现有车辆
            for actor in self.world.get_actors().filter('vehicle.*'):
                actor.destroy()
                
            # 生成特斯拉
            blueprint = self.world.get_blueprint_library().find('vehicle.tesla.model3')
            spawn_points = self.world.get_map().get_spawn_points()
            
            if not spawn_points:
                raise RuntimeError("地图中没有可用的生成点")
                
            # 尝试不同的生成点
            for spawn_point in spawn_points:
                try:
                    self.vehicle = self.world.spawn_actor(blueprint, spawn_point)
                    self.vehicle.set_autopilot(True)
                    logger.info(f"成功生成车辆在位置: {spawn_point.location}")
                    return True
                except Exception as e:
                    continue
                    
            raise RuntimeError("所有生成点都被占用")
            
        except Exception as e:
            logger.error(f"车辆设置失败: {str(e)}")
            return False
            
    def setup_camera(self):
        """设置相机"""
        try:
            if not self.vehicle:
                raise RuntimeError("未找到车辆，无法安装相机")
                
            # 创建相机蓝图
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.config.WIDTH))
            camera_bp.set_attribute('image_size_y', str(self.config.HEIGHT))
            camera_bp.set_attribute('fov', '110')
            
            # 设置相机位置
            camera_transform = carla.Transform(
                carla.Location(x=1.5, z=2.4),
                carla.Rotation(pitch=-5)
            )
            
            # 生成相机
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            logger.info("相机设置成功")
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
                logger.info("相机已清理")
                
            if self.vehicle:
                self.vehicle.destroy()
                logger.info("车辆已清理")
                
            if self.world:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
                logger.info("世界设置已重置")
                
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
        
    def record_video(self):
        """第一步：录制原始视频"""
        try:
            logger.info("开始录制原始视频...")
            
            # 初始化录制组件
            self.video_recorder = VideoRecorder(self.config)
            self.carla_interface = CarlaInterface(self.config)
            
            # 初始化CARLA
            if not self.carla_interface.connect():
                return False
            if not self.carla_interface.setup_vehicle():
                return False
            if not self.carla_interface.setup_camera():
                return False
            
            # 创建显示窗口
            cv2.namedWindow('Raw Video', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Raw Video', self.config.WIDTH//2, self.config.HEIGHT//2)
            
            # 启动录制
            self.video_recorder.start()
            self.start_time = time.time()
            
            def on_image(image):
                if self.stop_flag.is_set():
                    return
                    
                # 转换图像格式
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((self.config.HEIGHT, self.config.WIDTH, 4))
                frame = array[:, :, :3].copy()
                
                # 显示原始画面
                cv2.imshow('Raw Video', frame)
                
                # 录制帧
                self.video_recorder.add_frame(frame)
                self.frame_count += 1
                
                # 显示进度
                elapsed_time = time.time() - self.start_time
                if self.frame_count % 30 == 0:  # 每秒更新一次
                    logger.info(f"已录制 {self.frame_count} 帧, 用时 {elapsed_time:.1f}s")
                
                # 检查退出
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_flag.set()
            
            # 启动相机
            self.carla_interface.camera.listen(on_image)
            
            logger.info("正在录制，按'Q'键退出...")
            
            # 主循环
            while not self.stop_flag.is_set():
                if time.time() - self.start_time >= self.config.RECORD_SECONDS:
                    logger.info("达到预设录制时间，准备结束...")
                    break
                self.carla_interface.world.tick()
                time.sleep(0.001)
            
            return True
            
        except Exception as e:
            logger.error(f"视频录制错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            cv2.destroyAllWindows()
            if self.video_recorder:
                self.video_recorder.stop()
            if self.carla_interface:
                self.carla_interface.cleanup()
    
    def process_video(self):
        """第二步：处理视频并检测车道线"""
        try:
            logger.info("开始处理视频...")
            
            # 加载模型
            self.frame_processor = FrameProcessor(self.config)
            
            # 创建输出目录
            output_dir = os.path.join(os.path.dirname(self.config.AVI_PATH), "processed")
            os.makedirs(output_dir, exist_ok=True)
            
            # 设置输出视频路径
            output_path = os.path.join(output_dir, f"processed_{os.path.basename(self.config.AVI_PATH)}")
            
            # 打开原始视频
            cap = cv2.VideoCapture(self.config.AVI_PATH)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频文件: {self.config.AVI_PATH}")
            
            # 创建输出视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                self.config.FPS,
                (self.config.WIDTH, self.config.HEIGHT),
                True
            )
            
            # 创建显示窗口
            cv2.namedWindow('Processing', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Processing', self.config.WIDTH//2, self.config.HEIGHT//2)
            
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"开始处理，总帧数: {total_frames}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理帧
                result_frame, lanes = self.frame_processor.process_frame(frame)
                
                # 写入处理后的帧
                out.write(result_frame)
                
                # 显示处理进度
                frame_count += 1
                if frame_count % 30 == 0:  # 每30帧更新一次进度
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"处理进度: {progress:.1f}% ({frame_count}/{total_frames})")
                
                # 显示处理结果
                cv2.imshow('Processing', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            logger.info(f"视频处理完成！输出文件: {output_path}")
            
            # 转换为MP4
            try:
                output_mp4 = output_path.replace('.avi', '.mp4')
                cmd = [
                    'ffmpeg', '-y',
                    '-i', output_path,
                    '-vcodec', 'libx264',
                    '-acodec', 'aac',
                    output_mp4
                ]
                subprocess.run(cmd, check=True)
                logger.info(f"已转换为MP4: {output_mp4}")
            except Exception as e:
                logger.error(f"MP4转换失败: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"视频处理错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            cv2.destroyAllWindows()
    
    def run(self):
        """运行系统"""
        try:
            # 第一步：录制视频
            if not self.record_video():
                logger.error("视频录制失败")
                return
            
            logger.info("视频录制完成，准备开始处理...")
            time.sleep(1)  # 稍作暂停
            
            # 第二步：处理视频
            if not self.process_video():
                logger.error("视频处理失败")
                return
            
            logger.info("所有处理完成！")
            
        except Exception as e:
            logger.error(f"系统运行错误: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_flag.set()

def main():
    system = LaneDetectionSystem()
    system.run()

if __name__ == '__main__':
    main() 