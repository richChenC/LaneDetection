import sys
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import scipy.special
import time
import glob
import re
import threading
from queue import Queue
import concurrent.futures

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, 
    QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QComboBox, QGroupBox, QRadioButton, QStyle, QStyleFactory,
    QMessageBox, QProgressBar, QTextEdit, QFrame, QCheckBox, QSlider
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor

class VideoLoadThread(QThread):
    """异步加载视频的线程类"""
    finished = pyqtSignal(bool)
    progress = pyqtSignal(int)
    
    def __init__(self, video_path):
        """初始化视频加载线程"""
        super().__init__()
        self.video_path = video_path
        self.success = False
        
    def run(self):
        """运行线程"""
        try:
            # 尝试打开视频文件
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                # 获取视频信息
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                self.success = True
                self.progress.emit(100)
            else:
                self.success = False
                
        except Exception as e:
            print(f"加载视频时出错: {str(e)}")
            self.success = False
            
        finally:
            self.finished.emit(self.success)

class LaneDetectionUI(QMainWindow):
    def __init__(self):
        """初始化车道线检测UI"""
        super().__init__()
        
        # 基础变量初始化
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_mode = "realtime"
        
        # 设置基础路径
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.video_dir = os.path.join(self.base_dir, "my-video", "test-video")
        self.output_dir = os.path.join(self.base_dir, "my-video", "output")
        self.model_path = os.path.join(self.base_dir, "my-model", "culane_18.pth")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载视频列表
        self.video_list = self._get_video_list()
        self.current_video_index = 0
        
        # 图像和帧率相关参数
        self.img_w, self.img_h = 1640, 590
        self.frame_count = 0
        self.frame_counter = 0
        self.start_time = time.time()
        self.fps = 0
        
        # 模型相关参数
        self.default_model = "culane_18"
        self.available_models = {}
        self.net = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化图像预处理
        self.img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # 初始化模型相关参数
        self.cls_num_per_lane = 18
        self.row_anchor = np.array([
            121, 131, 141, 150, 160, 170, 180, 189,
            199, 209, 219, 228, 238, 248, 258, 267,
            277, 287
        ])
        
        # 初始化UI
        self.init_ui()
        
        # 直接加载默认模型
        self.load_default_model()
        
        # 直接加载默认视频
        if self.video_list:
            self.default_video = self.video_list[0]
            if os.path.exists(self.default_video):
                self.process_video(self.default_video)
                self.path_label.setText(f"已选择视频: {os.path.basename(self.default_video)}")
                
        # 设置默认状态
        self._set_default_states()

    def _set_default_states(self):
        """设置默认状态"""
        try:
            # 设置默认选项
            self.video_btn.setChecked(True)
            self.realtime_btn.setChecked(True)
            self.show_fps_btn.setChecked(True)
            self.green_mask_cb.setChecked(True)
            self.lane_hline_cb.setChecked(True)
            self.red_lane_cb.setChecked(True)
            self.car_center_cb.setChecked(True)
            
        except Exception as e:
            self.set_op_feedback(f"设置默认状态时出错: {str(e)}", "#e74c3c")
            print(f"设置默认状态时出错: {str(e)}")

    def init_model(self):
        """初始化模型"""
        try:
            # 设置CUDA优化
            torch.backends.cudnn.benchmark = True
            
            # 设置模型参数
            self.cls_num_per_lane = 18
            self.row_anchor = np.array([
                121, 131, 141, 150, 160, 170, 180, 189,
                199, 209, 219, 228, 238, 248, 258, 267,
                277, 287
            ])
            
            # 重置模型相关变量
            self.net = None
            self.current_model = None
            self.current_model_type = None
            self.available_models = {}
            
            # 扫描模型文件
            self.scan_model_files()
            
            # 初始化图像预处理
            self.img_transforms = transforms.Compose([
                transforms.Resize((288, 800)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            
        except Exception as e:
            self.set_op_feedback(f"初始化模型时出错: {str(e)}", "#e74c3c")
            print(f"初始化模型时出错: {str(e)}")

    def get_model_type_from_name(self, model_name):
        """从模型名称获取模型类型"""
        try:
            name = model_name.lower()
            if "tusimple" in name:
                return "TuSimple"
            elif "curvelanes" in name:
                return "CurveLanes"
            else:
                return "CULane"  # 默认
                
        except Exception as e:
            print(f"获取模型类型时出错: {str(e)}")
            return "CULane"  # 出错时返回默认值

    def scan_model_files(self):
        """扫描模型文件"""
        try:
            # 获取模型目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(script_dir, "my-model")
            
            # 检查目录是否存在
            if not os.path.exists(model_dir):
                print(f"警告：模型目录 {model_dir} 不存在！")
                return
                
            # 获取模型文件列表
            model_files = glob.glob(os.path.join(model_dir, "*.pth"))
            if not model_files:
                print(f"警告：在 {model_dir} 中未找到模型文件！")
                return
                
            # 打印找到的模型文件
            print("找到以下模型文件：")
            for model in model_files:
                print(f"  - {os.path.basename(model)}")
                
        except Exception as e:
            self.set_op_feedback(f"扫描模型文件时出错: {str(e)}", "#e74c3c")
            print(f"扫描模型文件时出错: {str(e)}")

    def _get_video_list(self):
        """获取视频列表"""
        try:
            # 获取视频文件列表
            video_list = sorted(
                glob.glob(os.path.join('my-video', 'test-video', '*.mp4')) +
                glob.glob(os.path.join('my-video', 'test-video', '*.avi')) +
                glob.glob(os.path.join('my-video', 'test-video', '*.mkv'))
            )
            
            # 打印找到的视频文件
            print("找到以下视频文件：")
            for video in video_list:
                print(f"  - {video}")
                
            return video_list
            
        except Exception as e:
            self.set_op_feedback(f"获取视频列表时出错: {str(e)}", "#e74c3c")
            print(f"获取视频列表时出错: {str(e)}")
            return []

    def _get_image_list(self):
        """获取图片列表"""
        try:
            # 获取图片文件列表
            img_list = sorted(
                glob.glob(os.path.join('my-video', 'test-video', '*.jpg')) +
                glob.glob(os.path.join('my-video', 'test-video', '*.png')) +
                glob.glob(os.path.join('my-video', 'test-video', '*.jpeg')) +
                glob.glob(os.path.join('my-video', 'test-video', '*.bmp'))
            )
            
            # 打印找到的图片文件
            print("找到以下图片文件：")
            for img in img_list:
                print(f"  - {img}")
                
            return img_list
            
        except Exception as e:
            self.set_op_feedback(f"获取图片列表时出错: {str(e)}", "#e74c3c")
            print(f"获取图片列表时出错: {str(e)}")
            return []

    def _create_output_dir(self, base_name):
        """创建输出目录"""
        try:
            # 创建输出目录
            output_dir = os.path.join("my-video", "output", f"output-{base_name}")
            os.makedirs(output_dir, exist_ok=True)
            return output_dir
            
        except Exception as e:
            self.set_op_feedback(f"创建输出目录时出错: {str(e)}", "#e74c3c")
            print(f"创建输出目录时出错: {str(e)}")
            return None

    def init_ui(self):
        """初始化UI"""
        # 设置窗口基本属性
        self.setWindowTitle('车道线检测系统 Lane Detection')
        self.setGeometry(100, 100, 1500, 950)
        self.setFixedSize(1500, 950)
        self.setWindowIcon(QIcon('configs/icon.png'))
        self.setStyle(QStyleFactory.create('Fusion'))
        
        # 设置全局调色板
        self._setup_palette()
        
        # 创建主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_layout.setSpacing(24)
        main_layout.setContentsMargins(24, 16, 24, 16)
        
        # 创建左侧操作面板
        left_panel = self._create_left_panel()
        main_layout.addLayout(left_panel)
        
        # 创建右侧显示面板
        right_panel = self._create_right_panel()
        main_layout.addLayout(right_panel)
        
        # 设置主布局
        main_widget.setLayout(main_layout)
        
        # 设置默认状态
        self.video_btn.setChecked(True)
        self.realtime_btn.setChecked(True)
        self.show_fps_btn.setChecked(True)
        self.green_mask_cb.setChecked(True)
        self.lane_hline_cb.setChecked(True)
        self.red_lane_cb.setChecked(True)
        self.car_center_cb.setChecked(True)
        
        # 绑定信号
        self._connect_signals()

    def _setup_palette(self):
        """设置全局调色板"""
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.white)
        palette.setColor(QPalette.WindowText, Qt.black)
        palette.setColor(QPalette.Base, Qt.white)
        palette.setColor(QPalette.AlternateBase, Qt.white)
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.black)
        palette.setColor(QPalette.Text, Qt.black)
        palette.setColor(QPalette.Button, Qt.white)
        palette.setColor(QPalette.ButtonText, Qt.black)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        self.setPalette(palette)

    def _create_left_panel(self):
        """创建左侧操作面板"""
        left_layout = QVBoxLayout()
        left_layout.setSpacing(16)
        
        # 创建操作区
        op_panel = self._create_operation_panel()
        left_layout.addWidget(op_panel)
        
        # 创建状态显示区
        status_panel = self._create_status_panel()
        left_layout.addWidget(status_panel)
        
        # 创建终端输出区
        terminal_panel = self._create_terminal_panel()
        left_layout.addWidget(terminal_panel)
        
        return left_layout

    def _create_operation_panel(self):
        """创建操作区面板"""
        op_panel = QGroupBox("操作区")
        op_panel.setFont(QFont("微软雅黑", 13, QFont.Bold))
        op_panel.setStyleSheet(
            "QGroupBox { border: 2px solid #27ae60; border-radius: 12px; margin-top: 8px; background: #f7fcf8; }"
            "QGroupBox::title { left: 10px; padding: 0 3px 0 3px; font-weight:bold; color:#229954; }"
        )
        
        op_layout = QVBoxLayout()
        op_layout.setSpacing(16)
        op_layout.setContentsMargins(14, 44, 14, 14)
        
        # 添加模型选择区
        op_layout.addLayout(self._create_model_selection())
        
        # 添加输入源选择区
        op_layout.addWidget(self._create_input_source_group())
        
        # 添加文件操作区
        op_layout.addLayout(self._create_file_operations())
        
        # 添加模式选择区
        op_layout.addWidget(self._create_mode_group())
        
        # 添加检测按钮
        op_layout.addWidget(self._create_detection_button())
        
        # 添加可视化开关区
        op_layout.addWidget(self._create_visualization_group())
        
        op_panel.setLayout(op_layout)
        return op_panel

    def _create_model_selection(self):
        """创建模型选择区"""
        model_box = QHBoxLayout()
        
        # 模型标签
        model_label = QLabel("模型:")
        model_label.setFont(QFont("微软雅黑", 11))
        
        # 模型下拉框
        self.model_combo = QComboBox()
        self.model_combo.setFont(QFont("微软雅黑", 11))
        for model_name in self.available_models.keys():
            self.model_combo.addItem(model_name)
            
        # 刷新按钮
        self.refresh_btn = QPushButton()
        self.refresh_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.refresh_btn.setToolTip("刷新模型列表")
        self.refresh_btn.setFixedWidth(28)
        
        model_box.addWidget(model_label)
        model_box.addWidget(self.model_combo)
        model_box.addWidget(self.refresh_btn)
        
        return model_box

    def _create_input_source_group(self):
        """创建输入源选择组"""
        input_group = QGroupBox("输入源")
        input_group.setFont(QFont("微软雅黑", 12, QFont.Bold))
        input_group.setStyleSheet(
            "QGroupBox { margin-top: 8px; border: 2px solid #3498db; border-radius: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; top: 0px; font-weight:bold; color:#2980b9; }"
        )
        
        input_layout = QHBoxLayout()
        input_layout.setSpacing(12)
        input_layout.setContentsMargins(10, 18, 10, 18)
        
        # 创建输入源按钮
        self.camera_btn = QRadioButton("摄像头")
        self.video_btn = QRadioButton("视频")
        self.image_btn = QRadioButton("图片")
        
        for btn in [self.camera_btn, self.video_btn, self.image_btn]:
            btn.setFont(QFont("微软雅黑", 11))
            btn.setFixedHeight(32)
            btn.setMinimumWidth(80)
            btn.setMaximumWidth(100)
            btn.setStyleSheet("QRadioButton { padding: 2px 8px; }")
            input_layout.addWidget(btn)
        
        input_group.setMinimumHeight(100)
        input_group.setMaximumHeight(140)
        input_group.setMinimumWidth(260)
        input_group.setMaximumWidth(360)
        input_group.setLayout(input_layout)
        
        return input_group

    def _create_file_operations(self):
        """创建文件操作区"""
        file_box = QHBoxLayout()
        
        # 文件选择按钮
        self.file_btn = QPushButton("选择文件")
        self.file_btn.setFont(QFont("微软雅黑", 13, QFont.Bold))
        self.file_btn.setFixedHeight(36)
        self.file_btn.setFixedWidth(120)
        self.file_btn.setStyleSheet(
            "QPushButton { background-color: #2a82da; color: white; border-radius: 6px; font-size:15px; padding: 8px 18px; }"
        )
        
        # 导航按钮
        nav_btns_layout = QVBoxLayout()
        nav_btns_layout.setSpacing(12)
        
        self.prev_btn = QPushButton("上一个")
        self.next_btn = QPushButton("下一个")
        
        for btn in [self.prev_btn, self.next_btn]:
            btn.setFixedWidth(110)
            btn.setFixedHeight(36)
            btn.setStyleSheet(
                "QPushButton { background-color: #2a82da; color: white; font-size:15px; font-weight:bold; border-radius: 6px; }"
                "QPushButton:pressed { background-color: #1761a0; }"
            )
        
        nav_btns_layout.addWidget(self.prev_btn)
        nav_btns_layout.addWidget(self.next_btn)
        
        file_box.addWidget(self.file_btn)
        file_box.addLayout(nav_btns_layout)
        file_box.addStretch()
        
        return file_box

    def _create_mode_group(self):
        """创建模式选择组"""
        mode_group = QGroupBox("模式")
        mode_group.setFont(QFont("微软雅黑", 12, QFont.Bold))
        mode_group.setStyleSheet(
            "QGroupBox { margin-top: 8px; border: 2px solid #e67e22; border-radius: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; top: 0px; font-weight:bold; color:#e67e22; }"
        )
        
        mode_layout = QVBoxLayout()
        mode_layout.setSpacing(10)
        mode_layout.setContentsMargins(24, 28, 24, 28)
        
        self.realtime_btn = QRadioButton("实时检测")
        self.save_result_btn = QRadioButton("保存检测结果")
        
        for btn in [self.realtime_btn, self.save_result_btn]:
            btn.setFont(QFont("微软雅黑", 10))
            mode_layout.addWidget(btn)
        
        mode_group.setMinimumHeight(120)
        mode_group.setMaximumHeight(160)
        mode_group.setLayout(mode_layout)
        
        return mode_group

    def _create_detection_button(self):
        """创建检测按钮"""
        self.start_btn = QPushButton("开始检测")
        self.start_btn.setFont(QFont("微软雅黑", 12, QFont.Bold))
        self.start_btn.setStyleSheet(self.get_start_btn_style("start"))
        self.start_btn.setFixedHeight(32)
        return self.start_btn

    def _create_visualization_group(self):
        """创建可视化元素开关组"""
        vis_group = QGroupBox("可视化元素开关")
        vis_group.setFont(QFont("微软雅黑", 12, QFont.Bold))
        vis_group.setStyleSheet(
            "QGroupBox { margin-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; top: 0px; }"
        )
        
        vis_layout = QVBoxLayout()
        vis_layout.setSpacing(16)
        vis_layout.setContentsMargins(10, 28, 10, 28)
        
        # 创建开关
        self.show_fps_btn = QRadioButton("显示帧率")
        self.show_fps_btn.setFont(QFont("微软雅黑", 11))
        self.show_fps_btn.setStyleSheet("QRadioButton { color: #e74c3c; font-weight: bold; }")
        
        self.green_mask_cb = QCheckBox("绿色薄膜")
        self.lane_hline_cb = QCheckBox("车道横线")
        self.red_lane_cb = QCheckBox("红色行驶线")
        self.car_center_cb = QCheckBox("车头中心线")
        
        for cb in [self.green_mask_cb, self.lane_hline_cb, self.red_lane_cb, self.car_center_cb]:
            cb.setFont(QFont("微软雅黑", 11))
            vis_layout.addWidget(cb)
        
        vis_group.setLayout(vis_layout)
        return vis_group

    def _create_status_panel(self):
        """创建状态显示区"""
        status_panel = QGroupBox("状态显示")
        status_panel.setFont(QFont("微软雅黑", 12, QFont.Bold))
        status_panel.setStyleSheet(
            "QGroupBox { border: 2px solid #e67e22; border-radius: 12px; margin-top: 8px; background: #fdf6e3; }"
            "QGroupBox::title { left: 10px; padding: 0 3px 0 3px; font-weight:bold; color:#e67e22; }"
        )
        layout = QVBoxLayout()
        self.status_label = QLabel("模型种类: --")
        self.detect_state_label = QLabel("检测状态: 等待检测")
        self.carinfo_label = QLabel("")
        for label in [self.status_label, self.detect_state_label, self.carinfo_label]:
            label.setFont(QFont("微软雅黑", 11))
            layout.addWidget(label)
        status_panel.setLayout(layout)
        return status_panel

    def _create_terminal_panel(self):
        """创建终端输出区"""
        terminal_panel = QGroupBox("终端输出")
        terminal_panel.setFont(QFont("微软雅黑", 12, QFont.Bold))
        terminal_panel.setStyleSheet(
            "QGroupBox { border: 2px solid #2980b9; border-radius: 12px; margin-top: 8px; background: #f4faff; }"
            "QGroupBox::title { left: 10px; padding: 0 3px 0 3px; font-weight:bold; color:#2980b9; }"
        )
        layout = QVBoxLayout()
        self.terminal_text = QTextEdit()
        self.terminal_text.setReadOnly(True)
        self.terminal_text.setFont(QFont("Consolas", 10))
        self.terminal_text.setStyleSheet(
            "QTextEdit { background: #f4faff; color: #222; border-radius: 6px; padding: 8px; }"
        )
        layout.addWidget(self.terminal_text)
        terminal_panel.setLayout(layout)
        return terminal_panel

    def _create_right_panel(self):
        """创建右侧显示面板"""
        right_layout = QVBoxLayout()
        # 原始图像显示
        self.raw_label = QLabel("原始图像")
        self.raw_label.setAlignment(Qt.AlignCenter)
        self.raw_label.setFixedSize(640, 360)
        # 检测结果显示
        self.det_label = QLabel("检测结果")
        self.det_label.setAlignment(Qt.AlignCenter)
        self.det_label.setFixedSize(640, 360)
        # 帧率显示（可选）
        self.raw_fps = QLabel("")
        self.det_fps = QLabel("")
        self.raw_fps.setStyleSheet("color: #e74c3c; font-weight: bold; background: transparent;")
        self.det_fps.setStyleSheet("color: #e74c3c; font-weight: bold; background: transparent;")
        # 进度条（可选）
        self.video_progress = QProgressBar()
        self.video_progress.setValue(0)
        self.video_progress_label = QLabel("00:00/00:00")
        self.process_progress = QProgressBar()
        self.process_progress.setValue(0)
        # 路径显示
        self.path_label = QLabel("未选择文件")
        self.path_label.setStyleSheet("color: #2980b9; font-weight: bold;")
        # 布局
        right_layout.addWidget(self.path_label)
        right_layout.addWidget(self.raw_label)
        right_layout.addWidget(self.det_label)
        right_layout.addWidget(self.video_progress)
        right_layout.addWidget(self.video_progress_label)
        right_layout.addWidget(self.process_progress)
        return right_layout

    def _connect_signals(self):
        """连接信号槽"""
        # 模型相关
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.refresh_btn.clicked.connect(self.refresh_models)
        
        # 文件操作相关
        self.file_btn.clicked.connect(self.select_file)
        self.prev_btn.clicked.connect(self.play_prev)
        self.next_btn.clicked.connect(self.play_next)
        
        # 检测控制相关
        self.start_btn.clicked.connect(self.toggle_detection)
        self.save_result_btn.toggled.connect(self.on_save_result_mode)
        
        # 输入源切换相关
        for btn in [self.camera_btn, self.video_btn, self.image_btn]:
            btn.toggled.connect(self.on_input_source_changed)

    def set_op_feedback(self, msg, color="#2a82da"):
        self.op_feedback.setText(msg)
        self.op_feedback.setStyleSheet(f"color: {color}; background: #eaf2fb; padding: 8px; border-radius: 4px;")

    def append_terminal_log(self, msg):
        # 去除中间多余空格
        msg = re.sub(r' +', ' ', msg)
        self.terminal_text.append(msg)

    def on_model_changed(self, model_name):
        if self.timer.isActive():
            self.timer.stop()
            self.start_btn.setText("开始检测")
            self.start_btn.setStyleSheet(self.get_start_btn_style("start"))
            if self.cap:
                self.cap.release()
                self.cap = None
        if self.load_model(model_name):
            model_type = self.get_model_type_from_name(model_name)
            self.current_model_type = model_type
            if model_type == "CULane":
                self.img_w, self.img_h = 1640, 590
                self.row_anchor = np.array([121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287])
                self.set_op_feedback("当前模型：CULane，输入分辨率需为1640x590，支持视频、图片、图片txt列表。", "#229954")
            elif model_type == "TuSimple":
                self.img_w, self.img_h = 1280, 720
                self.row_anchor = np.array(tusimple_row_anchor)
                self.set_op_feedback("当前模型：TuSimple，输入分辨率需为1280x720，支持视频、图片、图片txt列表。", "#229954")
            elif model_type == "CurveLanes":
                self.img_w, self.img_h = 1600, 800
                self.row_anchor = np.linspace(0, 800, 72, dtype=int)
                self.set_op_feedback("当前模型：CurveLanes，输入分辨率需为1600x800，支持视频、图片、图片txt列表。", "#229954")
            else:
                self.set_op_feedback("当前模型输入要求未知，请参考模型文档。", "#e67e22")
            self.status_label.setText(f"模型种类: {model_type}")
            self.detect_state_label.setText("检测状态: 等待检测")
            if self.video_btn.isChecked() and self.path_label.text() != "未选择文件":
                video_path = self.path_label.text().replace("已选择视频: ", "")
                if os.path.exists(video_path):
                    self.process_video(video_path)
                    self.toggle_detection()
        else:
            self.status_label.setText("模型种类: --")
            self.detect_state_label.setText("检测状态: 加载模型失败")
            self.set_op_feedback("模型加载失败，请检查模型文件。", "#e74c3c")

    def smooth_lane(self, prev_lane, curr_lane, alpha=0.5):
        # prev_lane, curr_lane: list of (x, y)
        if prev_lane is None or curr_lane is None:
            return curr_lane
        if len(prev_lane) != len(curr_lane):
            return curr_lane
        smoothed = [(int(alpha * x1 + (1 - alpha) * x0), int(alpha * y1 + (1 - alpha) * y0))
                    for (x0, y0), (x1, y1) in zip(prev_lane, curr_lane)]
        return smoothed

    def detect_and_draw_lanes(self, img):
        """检测并绘制车道线"""
        try:
            if self.net is None:
                self.status_label.setText("模型种类: --")
                self.detect_state_label.setText("检测状态: 未加载模型")
                self.set_op_feedback("未加载模型，请先选择模型。", "#e74c3c")
                return img
                
            # 图像预处理
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            x = self.img_transforms(img_pil)
            x = x.unsqueeze(0).cuda(non_blocking=True)
            
            # 模型推理
            with torch.no_grad():
                out = self.net.half()(x.half()) if x.dtype == torch.float16 else self.net(x)
                
            # 后处理
            col_sample = np.linspace(0, 800 - 1, 200)
            col_sample_w = col_sample[1] - col_sample[0]
            
            out_j = out[0].data.cpu().numpy()
            out_j = out_j[:, ::-1, :]
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(200) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == 200] = 0
            out_j = loc
            
            # 创建结果图像
            result_img = img.copy()
            lanes = []
            lane_xs = []
            
            # 提取车道线点
            for i in range(out_j.shape[1]):
                lane_points = []
                xs = []
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            x = int(out_j[k, i] * col_sample_w * img.shape[1] / 800) - 1
                            y = int(img.shape[0] * (self.row_anchor[self.cls_num_per_lane - 1 - k] / 288)) - 1
                            lane_points.append((x, y))
                            xs.append(x)
                            cv2.circle(result_img, (x, y), 5, (0, 0, 255), -1)
                if lane_points:
                    lanes.append(lane_points)
                    lane_xs.append(xs[-1] if xs else 0)
                    
            # 处理有效车道线
            valid_xs = [x for x in lane_xs if x > 0]
            detected_count = len(valid_xs)
            xs_display = valid_xs + [0] * (4 - len(valid_xs))
            html_log = ' | '.join([str(v) for v in xs_display])
            self.terminal_text.append(html_log)
            
            # 确定当前车道
            center_x = img.shape[1] // 2
            lane_index = self._determine_current_lane(lanes, center_x)
            
            # 绘制可视化效果
            result_img = self._draw_visualization(result_img, lanes, lane_index, valid_xs, center_x)
            
            # 更新UI信息
            self._update_lane_info(detected_count, lane_index)
            
            return result_img
            
        except Exception as e:
            self.terminal_text.append(f"检测过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return img

    def _determine_current_lane(self, lanes, center_x):
        """确定当前车道"""
        lane_index = 1
        if len(lanes) >= 2:
            # 收集所有y坐标
            all_ys = set()
            for lane in lanes:
                for pt in lane:
                    all_ys.add(pt[1])
            all_ys = sorted(list(all_ys))
            
            # 统计车辆位于哪两条车道线之间
            between_counts = [0] * (len(lanes) - 1)
            for y in all_ys:
                xs_at_y = []
                for lane in lanes:
                    if lane:
                        pt = min(lane, key=lambda p: abs(p[1] - y))
                        xs_at_y.append(pt[0])
                xs_at_y = sorted(xs_at_y)
                for i in range(len(xs_at_y) - 1):
                    if xs_at_y[i] < center_x <= xs_at_y[i + 1]:
                        between_counts[i] += 1
                        
            # 选择最频繁的区间作为当前车道
            if between_counts:
                lane_index = between_counts.index(max(between_counts)) + 1
                
        return lane_index

    def _draw_visualization(self, img, lanes, lane_index, valid_xs, center_x):
        """绘制可视化效果"""
        # 创建叠加层
        overlay_hline = img.copy()  # 用于车道横线
        overlay_mask = img.copy()   # 用于绿色薄膜
        lane_hlines_pts = []
        
        # 如果有足够的车道线，绘制可视化效果
        if len(valid_xs) >= 2 and lane_index <= len(lanes) - 1:
            left_lane = lanes[lane_index - 1]
            right_lane = lanes[lane_index]
            
            if left_lane and right_lane:
                # 处理左右车道线
                left_pts = left_lane[::-1]
                right_pts = right_lane
                
                # 绘制横线和收集点
                for lp in left_pts:
                    rp = min(right_pts, key=lambda p: abs(p[1] - lp[1]))
                    if self.lane_hline_cb.isChecked():
                        cv2.line(overlay_hline, (lp[0], lp[1]), (rp[0], rp[1]), (60, 30, 10), 5)
                    lane_hlines_pts.append((lp, rp))
                
                # 绘制绿色薄膜
                if self.green_mask_cb.isChecked() and len(lane_hlines_pts) > 1:
                    poly_pts = [pt[0] for pt in lane_hlines_pts] + [pt[1] for pt in lane_hlines_pts[::-1]]
                    pts = np.array(poly_pts, np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay_mask, [pts], (0, 255, 0))
                
                # 绘制红色行驶线
                if self.red_lane_cb.isChecked():
                    mid_points = []
                    for lp in left_pts:
                        rp = min(right_pts, key=lambda p: abs(p[1] - lp[1]))
                        mid_x = (lp[0] + rp[0]) // 2
                        mid_y = (lp[1] + rp[1]) // 2
                        mid_points.append((mid_x, mid_y))
                    if len(mid_points) > 1:
                        cv2.polylines(img, [np.array(mid_points, np.int32)], False, (0, 0, 255), 4)
        
        # 绘制车头中心线
        if self.car_center_cb.isChecked():
            head_center_y = int(img.shape[0] * 0.8)
            cv2.line(img, (center_x, head_center_y - 20), (center_x, head_center_y + 20), (0, 255, 255), 4)
        
        # 叠加绿色薄膜和车道横线
        if self.green_mask_cb.isChecked() and len(lane_hlines_pts) > 1:
            img = cv2.addWeighted(img, 0.7, overlay_mask, 0.3, 0)
        if self.lane_hline_cb.isChecked() and len(lane_hlines_pts) > 1:
            img = cv2.addWeighted(img, 0.7, overlay_hline, 0.5, 0)
            
        return img

    def _update_lane_info(self, detected_count, lane_index):
        """更新车道信息显示"""
        # 确定车道类型
        if detected_count == 4:
            lane_type = ["左车道", "中车道", "右车道"]
            lane_name = lane_type[lane_index - 1] if 1 <= lane_index <= 3 else "未知"
        elif detected_count == 3:
            lane_type = ["左车道", "右车道"]
            lane_name = lane_type[lane_index - 1] if 1 <= lane_index <= 2 else "未知"
        elif detected_count == 2:
            lane_name = "单行道"
        else:
            lane_name = "未知"
            
        # 更新状态标签
        self.status_label.setText(f"模型种类: {self.current_model_type}")
        self.detect_state_label.setText("检测状态: 已检测")
        
        # 更新车道信息显示
        lane_count_html = (
            f"<span style='color:#e74c3c;font-size:18px;font-weight:bold;'>检测到"
            f"<span style='color:#2980b9;font-size:18px;font-weight:bold;'>{detected_count}</span>"
            f"条车道线</span>"
        )
        
        self.carinfo_label.setText(
            f"{lane_count_html}<br>"
            f"<span style='color:#e74c3c;font-weight:bold;'>当前车道: {lane_name}</span><br>"
            f"{self._get_steering_recommendation(lane_index)}"
        )

    def _get_steering_recommendation(self, lane_index):
        """获取转向建议"""
        recommend_text = ""
        if len(self.lanes) >= 2 and lane_index <= len(self.lanes) - 1:
            left_lane = self.lanes[lane_index - 1]
            right_lane = self.lanes[lane_index]
            
            if left_lane and right_lane:
                # 计算偏移
                left_x = left_lane[-1][0]
                right_x = right_lane[-1][0]
                lane_center = (left_x + right_x) // 2
                offset = self.center_x - lane_center
                lane_width = abs(right_x - left_x)
                
                # 计算转向建议
                threshold = max(30, lane_width // 10)  # 动态弹性空间
                max_angle = 30  # 最大推荐角度
                
                if abs(offset) <= threshold:
                    recommend_text = "<span style='color:#229954;font-weight:bold;'>推荐：直行</span>"
                else:
                    # 偏移与角度成比例，最大±30度
                    angle = int(max(-max_angle, min(max_angle, offset / lane_width * max_angle * 2)))
                    if angle > 0:
                        recommend_text = f"<span style='color:#e67e22;font-weight:bold;'>推荐：左转 {angle}°</span>"
                    else:
                        recommend_text = f"<span style='color:#e67e22;font-weight:bold;'>推荐：右转 {abs(angle)}°</span>"
                        
        return recommend_text

    def update_frame(self):
        """更新视频帧"""
        try:
            if self.cap is None or not self.cap.isOpened():
                return
                
            ret, frame = self.cap.read()
            if ret:
                # 保存原始帧
                self.last_raw_frame = frame.copy()
                
                # 更新帧率计算
                self._update_fps()
                
                # 处理帧
                processed_frame = self.detect_and_draw_lanes(frame)
                self.display_results(processed_frame)
                
                # 更新进度
                self._update_progress()
                
            else:
                # 视频结束，自动切换到下一个视频
                self.play_next_video()
                
        except Exception as e:
            self.set_op_feedback(f"更新帧时出错: {str(e)}", "#e74c3c")
            print(f"更新帧时出错: {str(e)}")
            self.stop_detection()

    def _update_fps(self):
        """更新帧率计算"""
        self.frame_counter += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0.5:  # 每0.5秒更新一次FPS
            self.fps = self.frame_counter / elapsed_time
            self.frame_counter = 0
            self.start_time = time.time()

    def _update_progress(self):
        """更新进度显示"""
        if self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.video_progress.setValue(current_frame)
            
            # 更新时间标签
            current_time = int(self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            total_time = int(self.total_frames / max(1, self.fps))
            self.video_progress_label.setText(
                f"{current_time//60:02d}:{current_time%60:02d}/{total_time//60:02d}:{total_time%60:02d}"
            )

    def stop_detection(self):
        """停止检测"""
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.setText("开始检测")
        self.start_btn.setStyleSheet(self.get_start_btn_style("start"))
        self.detect_state_label.setText("检测状态: 已停止")

    def seek_video(self, position):
        """跳转到视频指定位置"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            
    def get_start_btn_style(self, state):
        """获取开始按钮样式"""
        if state == "start":
            return """
                QPushButton {
                    background-color: #27ae60;
                    color: white;
                    border: none;
                    padding: 5px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #2ecc71;
                }
                QPushButton:pressed {
                    background-color: #229954;
                }
            """
        else:
            return """
                QPushButton {
                    background-color: #c0392b;
                    color: white;
                    border: none;
                    padding: 5px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #e74c3c;
                }
                QPushButton:pressed {
                    background-color: #a93226;
                }
            """

    def display_results(self, processed, original=None):
        """显示处理结果"""
        def set_label_pixmap(label, img):
            """设置标签图像"""
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qt_img)
            label.setPixmap(pix.scaled(
                label.width(), 
                label.height(), 
                Qt.KeepAspectRatioByExpanding, 
                Qt.SmoothTransformation
            ))
            label.setMinimumSize(label.width(), label.height())
            label.setMaximumSize(label.width(), label.height())
            
        # 显示原始图像
        if self.cap is not None and self.cap.isOpened():
            if self.last_raw_frame is not None:
                set_label_pixmap(self.raw_label, self.last_raw_frame)
            else:
                set_label_pixmap(self.raw_label, processed)
        elif self.image_btn.isChecked() and original is not None:
            set_label_pixmap(self.raw_label, original)
        else:
            set_label_pixmap(self.raw_label, processed)
            
        # 显示处理后的图像
        set_label_pixmap(self.det_label, processed)
        
        # 显示帧率
        self._display_fps()

    def _display_fps(self):
        """显示帧率"""
        if self.show_fps_btn.isChecked():
            fps_text = f"FPS: {self.fps:.1f}"
            for fps_label in [self.raw_fps, self.det_fps]:
                fps_label.setParent(self.raw_label if fps_label == self.raw_fps else self.det_label)
                fps_label.move(10, 10)
                fps_label.setText(fps_text)
                fps_label.show()
        else:
            self.raw_fps.hide()
            self.det_fps.hide()

    def process_video_frames(self, video_path, output_dir):
        """处理视频帧并保存结果"""
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                QMessageBox.critical(self, "错误", f"无法打开视频文件: {video_path}")
                self.set_op_feedback("无法打开视频文件。", "#e74c3c")
                return
                
            # 获取视频信息
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 准备输出文件
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video_path = os.path.join(output_dir, f"output-{base_name}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # 处理每一帧
            frame_idx = 0
            self.process_progress.setValue(0)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 处理帧
                processed = self.detect_and_draw_lanes(frame)
                writer.write(processed)
                
                # 保存帧图像
                frame_img_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
                cv2.imwrite(frame_img_path, processed)
                
                # 更新进度
                frame_idx += 1
                if total > 0:
                    progress = int(frame_idx / total * 100)
                    self.process_progress.setValue(progress)
                    
            # 清理资源
            cap.release()
            writer.release()
            
            # 更新UI
            self.process_progress.setValue(100)
            self.set_op_feedback(f"视频检测结果已保存到: {output_video_path}", "#229954")
            self.append_terminal_log(f"视频检测结果已保存到: {output_video_path}")
            
            # 显示完成消息
            QMessageBox.information(
                self, 
                "处理完成", 
                f"检测结果已保存，输出视频在: {output_video_path}\n每帧图片在: {output_dir}"
            )
            
        except Exception as e:
            self.set_op_feedback(f"处理视频帧时出错: {str(e)}", "#e74c3c")
            print(f"处理视频帧时出错: {str(e)}")
            
        finally:
            if 'cap' in locals():
                cap.release()
            if 'writer' in locals():
                writer.release()

    def play_prev_video(self):
        """播放上一个视频"""
        try:
            # 获取当前视频路径
            current_path = self.path_label.text().replace("已选择视频: ", "")
            if not current_path:
                return
                
            # 获取视频列表
            video_dir = os.path.dirname(current_path)
            video_files = sorted([
                f for f in os.listdir(video_dir) 
                if f.lower().endswith(('.mp4', '.avi', '.mkv'))
            ])
            
            if not video_files:
                self.set_op_feedback("未找到视频文件。", "#e74c3c")
                return
                
            # 找到当前视频的索引
            current_name = os.path.basename(current_path)
            if current_name in video_files:
                idx = video_files.index(current_name)
                prev_idx = (idx - 1) % len(video_files)
                prev_video = os.path.join(video_dir, video_files[prev_idx])
                
                # 加载上一个视频
                self.process_video(prev_video)
                
        except Exception as e:
            self.set_op_feedback(f"切换视频时出错: {str(e)}", "#e74c3c")
            print(f"切换视频时出错: {str(e)}")

    def play_next_video(self):
        """播放下一个视频"""
        try:
            # 获取当前视频路径
            current_path = self.path_label.text().replace("已选择视频: ", "")
            if not current_path:
                return
                
            # 获取视频列表
            video_dir = os.path.dirname(current_path)
            video_files = sorted([
                f for f in os.listdir(video_dir) 
                if f.lower().endswith(('.mp4', '.avi', '.mkv'))
            ])
            
            if not video_files:
                self.set_op_feedback("未找到视频文件。", "#e74c3c")
                return
                
            # 找到当前视频的索引
            current_name = os.path.basename(current_path)
            if current_name in video_files:
                idx = video_files.index(current_name)
                next_idx = (idx + 1) % len(video_files)
                next_video = os.path.join(video_dir, video_files[next_idx])
                
                # 加载下一个视频
                self.process_video(next_video)
                
        except Exception as e:
            self.set_op_feedback(f"切换视频时出错: {str(e)}", "#e74c3c")
            print(f"切换视频时出错: {str(e)}")

    def seek_video(self, position):
        """跳转到视频指定位置"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)

    def process_video(self, video_path):
        """处理视频文件"""
        try:
            # 停止当前视频播放
            if self.timer.isActive():
                self.timer.stop()
            
            # 释放当前视频资源
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            # 创建新的视频捕获对象
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                self.set_op_feedback("打开视频失败", "#e74c3c")
                return False
                
            # 获取视频信息
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 设置进度条
            self.video_progress.setMaximum(self.total_frames)
            self.video_progress.setValue(0)
            
            # 更新UI状态
            self.path_label.setText(f"已选择视频: {video_path}")
            self.detect_state_label.setText("检测状态: 就绪")
            self.set_op_feedback("视频加载成功，可以开始检测。", "#229954")
            
            # 如果是实时检测模式，自动开始检测
            if self.realtime_btn.isChecked():
                self.toggle_detection()
                
            return True
            
        except Exception as e:
            self.set_op_feedback(f"处理视频时出错: {str(e)}", "#e74c3c")
            print(f"处理视频时出错: {str(e)}")
            return False

    def process_image(self, image_path):
        """处理图片文件"""
        try:
            # 加载图片
            img = cv2.imread(image_path)
            if img is None:
                QMessageBox.critical(self, "错误", f"无法打开图片文件: {image_path}")
                self.set_op_feedback("无法打开图片文件。", "#e74c3c")
                return
                
            # 处理图片
            processed = self.detect_and_draw_lanes(img)
            
            # 显示结果
            self.display_results(processed, original=img)
            
            # 更新UI状态
            self.status_label.setText(f"模型种类: {self.current_model_type}")
            self.detect_state_label.setText("检测状态: 图片检测完成")
            self.set_op_feedback("图片检测完成。", "#229954")
            
        except Exception as e:
            self.set_op_feedback(f"处理图片时出错: {str(e)}", "#e74c3c")
            print(f"处理图片时出错: {str(e)}")

    def start_default_detection(self):
        """开始默认检测"""
        try:
            if os.path.exists(self.default_video):
                # 加载默认视频
                self.process_video(self.default_video)
                
                # 开始检测
                self.timer.start(25)  # 约40FPS
                self.start_btn.setText("停止检测")
                self.start_btn.setStyleSheet(self.get_start_btn_style("stop"))
                
                # 更新UI状态
                self.status_label.setText(f"模型种类: {self.current_model_type}")
                self.detect_state_label.setText("检测状态: 正在处理...")
                self.set_op_feedback("正在处理默认视频...", "#2980b9")
            else:
                self.set_op_feedback("默认视频文件不存在。", "#e74c3c")
                
        except Exception as e:
            self.set_op_feedback(f"启动默认检测时出错: {str(e)}", "#e74c3c")
            print(f"启动默认检测时出错: {str(e)}")

    def toggle_detection(self):
        """切换检测状态"""
        if self.timer.isActive():
            self.timer.stop()
            self.start_btn.setText("开始检测")
            self.start_btn.setStyleSheet(self.get_start_btn_style("start"))
            self.detect_state_label.setText("检测状态: 已暂停")
        else:
            if self.cap is None or not self.cap.isOpened():
                if self.video_list:
                    self.process_video(self.video_list[0])
                else:
                    self.set_op_feedback("请先选择视频文件", "#e74c3c")
                    return
            self.timer.start(30)  # 约30FPS
            self.start_btn.setText("停止检测")
            self.start_btn.setStyleSheet(self.get_start_btn_style("stop"))
            self.detect_state_label.setText("检测状态: 检测中")

    def on_input_source_changed(self):
        """处理输入源变化"""
        try:
            # 停止当前检测
            if self.timer.isActive():
                self.timer.stop()
                
            # 释放资源
            if self.cap:
                self.cap.release()
                self.cap = None
                
            # 重置UI状态
            self.start_btn.setText("开始检测")
            self.start_btn.setStyleSheet(self.get_start_btn_style("start"))
            self.detect_state_label.setText("检测状态: 等待检测")
            self.status_label.setText(f"模型种类: {self.current_model_type}")
            self.last_raw_frame = None
            
            # 处理不同输入源
            if self.camera_btn.isChecked():
                # 摄像头模式
                self.open_camera()
            elif self.video_btn.isChecked():
                # 视频模式
                self.process_video(self.path_label.text().replace("已选择视频: ", ""))
            elif self.image_btn.isChecked():
                # 图片模式
                self.process_image(self.path_label.text().replace("已选择图片: ", ""))
                
        except Exception as e:
            self.set_op_feedback(f"切换输入源时出错: {str(e)}", "#e74c3c")
            print(f"切换输入源时出错: {str(e)}")

    def open_camera(self):
        """打开摄像头"""
        try:
            # 释放现有资源
            if self.cap:
                self.cap.release()
                
            # 打开摄像头
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "错误", "无法打开摄像头")
                self.set_op_feedback("无法打开摄像头。", "#e74c3c")
                return
                
            # 开始检测
            self.timer.start(30)  # 约30FPS
            self.start_btn.setText("停止检测")
            self.start_btn.setStyleSheet(self.get_start_btn_style("stop"))
            
            # 更新UI状态
            self.status_label.setText(f"模型种类: {self.current_model_type}")
            self.detect_state_label.setText("检测状态: 正在处理...")
            self.set_op_feedback("正在使用摄像头实时检测...", "#2980b9")
            
        except Exception as e:
            self.set_op_feedback(f"打开摄像头时出错: {str(e)}", "#e74c3c")
            print(f"打开摄像头时出错: {str(e)}")

    def play_prev(self):
        """播放上一个文件"""
        try:
            if self.video_btn.isChecked():
                self.play_prev_video()
            elif self.image_btn.isChecked():
                self.play_prev_image()
                
        except Exception as e:
            self.set_op_feedback(f"切换文件时出错: {str(e)}", "#e74c3c")
            print(f"切换文件时出错: {str(e)}")

    def play_next(self):
        """播放下一个文件"""
        try:
            if self.video_btn.isChecked():
                self.play_next_video()
            elif self.image_btn.isChecked():
                self.play_next_image()
                
        except Exception as e:
            self.set_op_feedback(f"切换文件时出错: {str(e)}", "#e74c3c")
            print(f"切换文件时出错: {str(e)}")

    def play_prev_image(self):
        """切换到上一个图片"""
        try:
            # 获取图片列表
            img_list = sorted(
                glob.glob(os.path.join('my-vedio', 'test-vedio', '*.jpg')) +
                glob.glob(os.path.join('my-vedio', 'test-vedio', '*.png')) +
                glob.glob(os.path.join('my-vedio', 'test-vedio', '*.jpeg')) +
                glob.glob(os.path.join('my-vedio', 'test-vedio', '*.bmp'))
            )
            
            # 获取当前图片路径
            now_path = self.path_label.text().replace("已选择图片: ", "") if self.path_label.text().startswith("已选择图片: ") else None
            
            # 找到上一个图片
            if now_path in img_list:
                idx = img_list.index(now_path)
                prev_idx = (idx - 1) % len(img_list)
                prev_img = img_list[prev_idx]
            else:
                prev_img = img_list[0] if img_list else None
                
            # 加载上一个图片
            if prev_img and os.path.exists(prev_img):
                self.path_label.setText(f"已选择图片: {os.path.relpath(prev_img)}")
                self.process_image(prev_img)
                
        except Exception as e:
            self.set_op_feedback(f"切换图片时出错: {str(e)}", "#e74c3c")
            print(f"切换图片时出错: {str(e)}")

    def play_next_image(self):
        """切换到下一个图片"""
        try:
            # 获取图片列表
            img_list = sorted(
                glob.glob(os.path.join('my-vedio', 'test-vedio', '*.jpg')) +
                glob.glob(os.path.join('my-vedio', 'test-vedio', '*.png')) +
                glob.glob(os.path.join('my-vedio', 'test-vedio', '*.jpeg')) +
                glob.glob(os.path.join('my-vedio', 'test-vedio', '*.bmp'))
            )
            
            # 获取当前图片路径
            now_path = self.path_label.text().replace("已选择图片: ", "") if self.path_label.text().startswith("已选择图片: ") else None
            
            # 找到下一个图片
            if now_path in img_list:
                idx = img_list.index(now_path)
                next_idx = (idx + 1) % len(img_list)
                next_img = img_list[next_idx]
            else:
                next_img = img_list[0] if img_list else None
                
            # 加载下一个图片
            if next_img and os.path.exists(next_img):
                self.path_label.setText(f"已选择图片: {os.path.relpath(next_img)}")
                self.process_image(next_img)
                
        except Exception as e:
            self.set_op_feedback(f"切换图片时出错: {str(e)}", "#e74c3c")
            print(f"切换图片时出错: {str(e)}")

    def select_file(self):
        """选择文件"""
        try:
            if self.video_btn.isChecked():
                # 选择视频文件
                file_name, _ = QFileDialog.getOpenFileName(
                    self,
                    "选择视频文件",
                    "",
                    "Video Files (*.mp4 *.avi *.mkv)"
                )
                if file_name:
                    self.path_label.setText(f"已选择视频: {file_name}")
                    self.set_op_feedback("已选择视频文件。", "#229954")
                    self.process_video(file_name)
                    
            elif self.image_btn.isChecked():
                # 选择图片文件
                file_name, _ = QFileDialog.getOpenFileName(
                    self,
                    "选择图片文件",
                    "",
                    "Image Files (*.jpg *.jpeg *.png *.bmp)"
                )
                if file_name:
                    self.path_label.setText(f"已选择图片: {file_name}")
                    self.set_op_feedback("已选择图片文件。", "#229954")
                    self.process_image(file_name)
                    
        except Exception as e:
            self.set_op_feedback(f"选择文件时出错: {str(e)}", "#e74c3c")
            print(f"选择文件时出错: {str(e)}")

    def closeEvent(self, event):
        """关闭窗口事件"""
        try:
            # 释放资源
            if self.cap:
                self.cap.release()
            event.accept()
            
        except Exception as e:
            print(f"关闭窗口时出错: {str(e)}")
            event.accept()

    def save_detection_result(self):
        """保存检测结果"""
        try:
            # 摄像头模式不支持保存
            if self.camera_btn.isChecked():
                QMessageBox.information(self, "提示", "摄像头输入不支持保存检测结果！")
                return
                
            # 视频模式
            if self.video_btn.isChecked() and self.path_label.text().startswith("已选择视频: "):
                video_path = self.path_label.text().replace("已选择视频: ", "")
                if os.path.exists(video_path):
                    base_name = os.path.splitext(os.path.basename(video_path))[0]
                    output_dir = os.path.join("my-vedio", "output", f"output-{base_name}")
                    os.makedirs(output_dir, exist_ok=True)
                    self.process_video_frames(video_path, output_dir)
                    
            # 图片模式
            elif self.image_btn.isChecked() and self.path_label.text().startswith("已选择图片: "):
                image_path = self.path_label.text().replace("已选择图片: ", "")
                if os.path.exists(image_path):
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_dir = os.path.join("my-vedio", "output", f"output-{base_name}")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 处理图片
                    img = cv2.imread(image_path)
                    processed = self.detect_and_draw_lanes(img)
                    
                    # 保存结果
                    out_path = os.path.join(output_dir, os.path.basename(image_path))
                    cv2.imwrite(out_path, processed)
                    
                    # 更新UI
                    self.process_progress.setValue(100)
                    self.set_op_feedback(f"图片检测结果已保存到: {out_path}", "#229954")
                    self.append_terminal_log(f"图片检测结果已保存到: {out_path}")
                    
                    # 显示完成消息
                    QMessageBox.information(self, "处理完成", f"检测结果已保存，输出图片在: {out_path}")
                    
        except Exception as e:
            self.set_op_feedback(f"保存检测结果时出错: {str(e)}", "#e74c3c")
            print(f"保存检测结果时出错: {str(e)}")

    def on_save_result_mode(self):
        """处理保存结果模式切换"""
        try:
            # 只对视频和图片可用，摄像头禁用
            if self.save_result_btn.isChecked():
                if self.camera_btn.isChecked():
                    self.set_op_feedback("摄像头输入不支持保存检测结果。", "#e67e22")
                    self.save_result_btn.setChecked(False)
                    self.realtime_btn.setChecked(True)
                    return
                    
                # 自动执行保存逻辑
                self.save_detection_result()
                
                # 切回实时检测模式
                self.realtime_btn.setChecked(True)
                
        except Exception as e:
            self.set_op_feedback(f"切换保存模式时出错: {str(e)}", "#e74c3c")
            print(f"切换保存模式时出错: {str(e)}")

    def refresh_models(self):
        """刷新模型列表"""
        self.model_combo.clear()
        self.scan_model_files()  # 重新扫描模型文件
        for model_name in self.available_models.keys():
            self.model_combo.addItem(model_name)
        self.set_op_feedback("模型列表已刷新", "#229954")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LaneDetectionUI()
    window.show()
    sys.exit(app.exec_())