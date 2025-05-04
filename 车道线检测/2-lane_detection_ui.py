import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QComboBox, QGroupBox, QRadioButton, QStyle, QStyleFactory, QMessageBox, QProgressBar, QTextEdit, QFrame, QCheckBox, QSlider)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon
from PyQt5.QtCore import Qt, QTimer
import torch
from model.model import parsingNet
import torchvision.transforms as transforms
from PIL import Image
import scipy.special
import time
import glob
from utils.common import merge_config
from utils.dist_utils import dist_print
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
import concurrent.futures
import re
from drllane_carla_rl.utils.visualize_rich import detect_and_draw_lanes

class LaneDetectionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_mode = "realtime"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_path = os.path.join(script_dir, "my-video", "output")
        self.img_w, self.img_h = 1640, 590
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.default_model = "culane_18"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_video = os.path.join(script_dir, "my-video", "test-video", "train1-1.mp4")
        self.last_raw_frame = None
        self.last_main_left = None
        self.last_main_right = None
        self.last_main_left_smoothed = None
        self.last_main_right_smoothed = None
        self.init_model()
        self.init_ui()
        if self.default_model in self.available_models:
            idx = list(self.available_models.keys()).index(self.default_model)
            self.model_combo.setCurrentIndex(idx)
            self.load_model(self.default_model)
        self.video_btn.setChecked(True)
        self.realtime_btn.setChecked(True)
        self.show_fps_btn.setChecked(True)
        self.lane_hline_cb.setChecked(True)
        self.green_mask_cb.setChecked(True)
        self.red_lane_cb.setChecked(True)
        self.car_center_cb.setChecked(True)
        if os.path.exists(self.default_video):
            self.path_label.setText(f"已选择视频: {self.default_video}")
            QTimer.singleShot(1000, self.start_default_detection)

    def init_model(self):
        torch.backends.cudnn.benchmark = True
        self.cls_num_per_lane = 18
        self.row_anchor = np.array([121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287])
        self.net = None
        self.current_model = None
        self.current_model_type = None
        self.available_models = {}
        self.scan_model_files()
        self.img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def get_model_type_from_name(self, model_name):
        name = model_name.lower()
        if "tusimple" in name:
            return "TuSimple"
        elif "curvelanes" in name:
            return "CurveLanes"
        else:
            return "CULane"  # 默认

    def scan_model_files(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, "my-model")
        if not os.path.exists(model_dir):
            print(f"警告：模型目录 {model_dir} 不存在！")
            return
        model_files = glob.glob(os.path.join(model_dir, "*.pth"))
        if not model_files:
            print(f"警告：在 {model_dir} 目录中没有找到模型文件！")
            return
        self.available_models = {}
        for model_path in model_files:
            model_name = os.path.basename(model_path)
            model_name_without_ext = os.path.splitext(model_name)[0]
            model_type = self.get_model_type_from_name(model_name_without_ext)
            backbone = "18"
            if "res34" in model_name_without_ext.lower():
                backbone = "34"
            elif "res50" in model_name_without_ext.lower():
                backbone = "50"
            elif "res101" in model_name_without_ext.lower():
                backbone = "101"
            display_name = model_name_without_ext
            self.available_models[display_name] = {
                "path": model_path,
                "backbone": backbone,
                "cls_dim": (200 + 1, 18, 4),
                "dataset": model_type
            }
        if hasattr(self, 'model_combo'):
            current_model = self.model_combo.currentText()
            self.model_combo.clear()
            for model_name in self.available_models.keys():
                self.model_combo.addItem(model_name)
            if current_model in self.available_models:
                index = self.model_combo.findText(current_model)
                if index >= 0:
                    self.model_combo.setCurrentIndex(index)
            elif self.model_combo.count() > 0:
                self.model_combo.setCurrentIndex(0)
                self.load_model(self.model_combo.currentText())

    def refresh_models(self):
        self.scan_model_files()
        self.status_label.setText("模型列表已刷新")
        if self.net is None and self.model_combo.count() > 0:
            self.load_model(self.model_combo.currentText())

    def load_model(self, model_name):
        if model_name not in self.available_models:
            print(f"错误：模型 {model_name} 不存在！")
            return False
        model_info = self.available_models[model_name]
        model_path = model_info["path"]
        if not os.path.exists(model_path):
            print(f"错误：模型文件 {model_path} 不存在！")
            return False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"正在使用设备: {device}")
        self.net = parsingNet(pretrained=False, backbone=model_info["backbone"], 
                            cls_dim=model_info["cls_dim"],
                            use_aux=False).to(device)
        try:
            state_dict = torch.load(model_path, map_location=device)
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            compatible_state_dict = {}
            for k, v in state_dict.items():
                if 'module.' in k:
                    compatible_state_dict[k[7:]] = v
                else:
                    compatible_state_dict[k] = v
            self.net.load_state_dict(compatible_state_dict, strict=False)
            self.net.eval()
            self.current_model = model_name
            self.current_model_type = self.get_model_type_from_name(model_name)
            return True
        except Exception as e:
            print(f"加载模型 {model_name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def init_ui(self):
        self.setWindowTitle('车道线检测系统 Lane Detection')
        self.setGeometry(100, 100, 1500, 950)
        self.setFixedSize(1500, 950)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "configs", "icon.png")
        self.setWindowIcon(QIcon(icon_path))
        self.setStyle(QStyleFactory.create('Fusion'))
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
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_layout.setSpacing(24)
        main_layout.setContentsMargins(24, 16, 24, 16)

        # 操作区
        op_panel = QGroupBox("操作区")
        op_panel.setFont(QFont("微软雅黑", 13, QFont.Bold))
        op_panel.setStyleSheet("QGroupBox { border: 2px solid #27ae60; border-radius: 12px; margin-top: 8px; background: #f7fcf8; } QGroupBox::title { left: 10px; padding: 0 3px 0 3px; font-weight:bold; color:#229954; }")
        op_layout = QVBoxLayout()
        op_layout.setSpacing(16)
        op_layout.setContentsMargins(14, 44, 14, 14)
        # 模型选择
        model_box = QHBoxLayout()
        model_label = QLabel("模型:")
        model_label.setFont(QFont("微软雅黑", 11))
        self.model_combo = QComboBox()
        self.model_combo.setFont(QFont("微软雅黑", 11))
        for model_name in self.available_models.keys():
            self.model_combo.addItem(model_name)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.refresh_btn = QPushButton()
        self.refresh_btn.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.refresh_btn.setToolTip("刷新模型列表")
        self.refresh_btn.clicked.connect(self.refresh_models)
        self.refresh_btn.setFixedWidth(28)
        model_box.addWidget(model_label)
        model_box.addWidget(self.model_combo)
        model_box.addWidget(self.refresh_btn)
        # 输入源竖向
        input_group = QGroupBox("输入源")
        input_group.setFont(QFont("微软雅黑", 12, QFont.Bold))
        input_group.setStyleSheet(
            "QGroupBox { margin-top: 8px; border: 2px solid #3498db; border-radius: 12px; } QGroupBox::title { subcontrol-origin: margin; left: 12px; top: 0px; font-weight:bold; color:#2980b9; }"
        )
        input_layout = QHBoxLayout()
        input_layout.setSpacing(12)
        input_layout.setContentsMargins(10, 18, 10, 18)
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
        self.camera_btn.setChecked(True)
        input_group.setMinimumHeight(100)
        input_group.setMaximumHeight(140)
        input_group.setMinimumWidth(260)
        input_group.setMaximumWidth(360)
        input_group.setLayout(input_layout)
        # 文件选择
        file_box = QHBoxLayout()
        self.file_btn = QPushButton("选择文件")
        self.file_btn.setFont(QFont("微软雅黑", 13, QFont.Bold))
        self.file_btn.setFixedHeight(36)
        self.file_btn.setFixedWidth(120)
        self.file_btn.setStyleSheet("QPushButton { background-color: #2a82da; color: white; border-radius: 6px; font-size:15px; padding: 8px 18px; }")
        self.file_btn.clicked.connect(self.select_file)
        file_box.addWidget(self.file_btn)
        # 上一个/下一个按钮竖向排布，按钮适中，蓝色风格
        nav_btns_layout = QVBoxLayout()
        nav_btns_layout.setSpacing(12)
        self.prev_btn = QPushButton("上一个")
        self.next_btn = QPushButton("下一个")
        self.prev_btn.setFixedWidth(110)
        self.next_btn.setFixedWidth(110)
        self.prev_btn.setFixedHeight(36)
        self.next_btn.setFixedHeight(36)
        btn_style = "QPushButton { background-color: #2a82da; color: white; font-size:15px; font-weight:bold; border-radius: 6px; } QPushButton:pressed { background-color: #1761a0; }"
        self.prev_btn.setStyleSheet(btn_style)
        self.next_btn.setStyleSheet(btn_style)
        self.prev_btn.clicked.connect(self.play_prev)
        self.next_btn.clicked.connect(self.play_next)
        nav_btns_layout.addWidget(self.prev_btn)
        nav_btns_layout.addWidget(self.next_btn)
        file_box.addLayout(nav_btns_layout)
        file_box.addStretch()
        self.path_label = QLabel("未选择文件")
        self.path_label.setFont(QFont("微软雅黑", 13))
        self.path_label.setWordWrap(True)
        self.path_label.setFixedHeight(48)
        self.path_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 8px; border-radius: 5px; font-size:15px;}")
        # 模式竖向
        mode_group = QGroupBox("模式")
        mode_group.setFont(QFont("微软雅黑", 12, QFont.Bold))
        mode_group.setStyleSheet(
            "QGroupBox { margin-top: 8px; border: 2px solid #e67e22; border-radius: 12px; } QGroupBox::title { subcontrol-origin: margin; left: 12px; top: 0px; font-weight:bold; color:#e67e22; }"
        )
        mode_layout = QVBoxLayout()
        mode_layout.setSpacing(10)
        mode_layout.setContentsMargins(24, 28, 24, 28)
        self.realtime_btn = QRadioButton("实时检测")
        self.save_result_btn = QRadioButton("保存检测结果")
        for btn in [self.realtime_btn, self.save_result_btn]:
            btn.setFont(QFont("微软雅黑", 10))
            mode_layout.addWidget(btn)
        self.realtime_btn.setChecked(True)
        mode_group.setMinimumHeight(120)
        mode_group.setMaximumHeight(160)
        mode_group.setLayout(mode_layout)
        # 检测按钮
        self.start_btn = QPushButton("开始检测")
        self.start_btn.setFont(QFont("微软雅黑", 12, QFont.Bold))
        self.start_btn.setStyleSheet(self.get_start_btn_style("start"))
        self.start_btn.setFixedHeight(32)
        self.start_btn.clicked.connect(self.toggle_detection)
        # 保存检测结果按钮事件绑定
        self.save_result_btn.toggled.connect(self.on_save_result_mode)
        # 选项
        self.show_fps_btn = QRadioButton("显示帧率")
        self.show_fps_btn.setFont(QFont("微软雅黑", 11))
        self.show_fps_btn.setChecked(True)
        self.show_fps_btn.setStyleSheet("QRadioButton { color: #e74c3c; font-weight: bold; }")
        # 分割线
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        # 操作提示区
        self.op_feedback = QLabel("欢迎使用车道线检测系统！")
        self.op_feedback.setFont(QFont("微软雅黑", 11, QFont.Bold))
        self.op_feedback.setStyleSheet("color: #2a82da; background: #eaf2fb; padding: 8px; border-radius: 4px;")
        self.op_feedback.setWordWrap(True)

        # ========== 可视化元素开关区 ========== 
        vis_switch_group = QGroupBox("可视化元素开关")
        vis_switch_group.setFont(QFont("微软雅黑", 12, QFont.Bold))
        vis_switch_group.setStyleSheet(
            "QGroupBox { margin-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; top: 0px; }"
        )
        vis_switch_layout = QVBoxLayout()
        vis_switch_layout.setSpacing(16)
        vis_switch_layout.setContentsMargins(10, 28, 10, 28)
        self.show_fps_btn = QRadioButton("显示帧率")
        self.show_fps_btn.setFont(QFont("微软雅黑", 11))
        self.show_fps_btn.setChecked(True)
        self.show_fps_btn.setStyleSheet("QRadioButton { color: #e74c3c; font-weight: bold; }")
        self.green_mask_cb = QCheckBox("绿色薄膜")
        self.lane_hline_cb = QCheckBox("车道横线")
        self.red_lane_cb = QCheckBox("红色行驶线")
        self.car_center_cb = QCheckBox("车头中心线")
        self.green_mask_cb.setFont(QFont("微软雅黑", 11))
        self.green_mask_cb.setChecked(True)
        self.lane_hline_cb.setFont(QFont("微软雅黑", 11))
        self.lane_hline_cb.setChecked(True)
        self.red_lane_cb.setFont(QFont("微软雅黑", 11))
        self.red_lane_cb.setChecked(True)
        self.car_center_cb.setFont(QFont("微软雅黑", 11))
        self.car_center_cb.setChecked(True)
        vis_switch_layout.addWidget(self.show_fps_btn)
        vis_switch_layout.addWidget(self.green_mask_cb)
        vis_switch_layout.addWidget(self.lane_hline_cb)
        vis_switch_layout.addWidget(self.red_lane_cb)
        vis_switch_layout.addWidget(self.car_center_cb)
        vis_switch_group.setLayout(vis_switch_layout)
        vis_switch_group.setMinimumHeight(220)
        vis_switch_group.setMaximumHeight(300)
        self.green_mask_cb.toggled.connect(self.update_frame)
        self.lane_hline_cb.toggled.connect(self.update_frame)
        self.red_lane_cb.toggled.connect(self.update_frame)
        self.car_center_cb.toggled.connect(self.update_frame)
        # ========== 可视化元素开关区 END ==========

        # 依次添加控件
        op_layout.addLayout(model_box)
        op_layout.addSpacing(12)
        op_layout.addWidget(input_group)
        op_layout.addSpacing(12)
        op_layout.addLayout(file_box)
        op_layout.addWidget(self.path_label)
        op_layout.addSpacing(12)
        op_layout.addWidget(mode_group)
        op_layout.addSpacing(12)
        op_layout.addWidget(self.start_btn)
        op_layout.addSpacing(12)
        op_layout.addWidget(vis_switch_group)
        op_layout.addStretch()
        op_layout.addWidget(line1)
        op_layout.addWidget(self.op_feedback)
        op_panel.setLayout(op_layout)
        op_panel.setMaximumWidth(340)

        # 显示区
        display_panel = QGroupBox("显示区")
        display_panel.setFont(QFont("微软雅黑", 13, QFont.Bold))
        display_panel.setStyleSheet("QGroupBox { border: 2px solid #2a82da; border-radius: 12px; margin-top: 8px; background: #f8fbff; } QGroupBox::title { left: 10px; padding: 0 3px 0 3px; font-weight:bold; color:#2a82da; }")
        display_layout = QVBoxLayout()
        display_layout.setSpacing(18)
        display_layout.setContentsMargins(18, 38, 18, 18)
        raw_title = QLabel("原始画面")
        raw_title.setFont(QFont("微软雅黑", 12, QFont.Bold))
        raw_title.setAlignment(Qt.AlignCenter)
        self.raw_label = QLabel()
        self.raw_label.setMinimumSize(760, 360)
        self.raw_label.setMaximumSize(760, 360)
        self.raw_label.setScaledContents(True)
        self.raw_label.setStyleSheet("QLabel { background-color: #f0f0f0; border-radius: 5px; border: 2px solid #2a82da; }")
        self.raw_fps = QLabel("FPS: 0.0")
        self.raw_fps.setFont(QFont("Consolas", 13, QFont.Bold))
        self.raw_fps.setStyleSheet("color: red; background: transparent; border: none;")
        self.raw_fps.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        det_title = QLabel("检测结果")
        det_title.setFont(QFont("微软雅黑", 12, QFont.Bold))
        det_title.setAlignment(Qt.AlignCenter)
        self.det_label = QLabel()
        self.det_label.setMinimumSize(760, 360)
        self.det_label.setMaximumSize(760, 360)
        self.det_label.setScaledContents(True)
        self.det_label.setStyleSheet("QLabel { background-color: #f0f0f0; border-radius: 5px; border: 2px solid #e67e22; }")
        self.det_fps = QLabel("FPS: 0.0")
        self.det_fps.setFont(QFont("Consolas", 13, QFont.Bold))
        self.det_fps.setStyleSheet("color: red; background: transparent; border: none;")
        self.det_fps.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        display_layout.addWidget(raw_title)
        display_layout.addWidget(self.raw_label)
        display_layout.addWidget(line2)
        display_layout.addWidget(det_title)
        display_layout.addWidget(self.det_label)
        display_panel.setLayout(display_layout)
        display_panel.setMinimumWidth(800)
        display_panel.setMaximumWidth(800)

        # 信息区
        info_panel = QGroupBox("信息区")
        info_panel.setFont(QFont("微软雅黑", 14, QFont.Bold))
        info_panel.setStyleSheet("QGroupBox { border: 2px solid #f0ad4e; border-radius: 12px; margin-top: 8px; background: #fffbe6; } QGroupBox::title { left: 10px; padding: 0 3px 0 3px; font-weight:bold; color:#f0ad4e; }")
        info_layout = QVBoxLayout()
        info_layout.setSpacing(18)
        info_layout.setContentsMargins(18, 38, 18, 18)
        self.status_label = QLabel("模型种类: --")
        self.status_label.setFont(QFont("微软雅黑", 16, QFont.Bold))
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("QLabel { color: #d35400; font-weight: bold; font-size:18px; background: #fffbe6;}")
        self.status_label.setMinimumHeight(38)
        self.detect_state_label = QLabel("检测状态: --")
        self.detect_state_label.setFont(QFont("微软雅黑", 16, QFont.Bold))
        self.detect_state_label.setWordWrap(True)
        self.detect_state_label.setStyleSheet("QLabel { color: #2a82da; font-weight: bold; font-size:18px; background: #fffbe6;}")
        self.detect_state_label.setMinimumHeight(38)
        self.carinfo_label = QLabel("车辆信息：--")
        self.carinfo_label.setFont(QFont("微软雅黑", 16, QFont.Bold))
        self.carinfo_label.setWordWrap(True)
        self.carinfo_label.setStyleSheet("QLabel { color: #229954; font-size:18px; background: #eafaf1; border-radius: 4px; padding: 10px;}")
        self.carinfo_label.setMinimumHeight(48)
        self.video_progress = QSlider(Qt.Horizontal)
        self.video_progress.setMinimum(0)
        self.video_progress.setMaximum(100)
        self.video_progress.setSingleStep(1)
        self.video_progress.sliderMoved.connect(self.seek_video)
        self.video_progress.setMinimumWidth(220)
        self.video_progress.setMaximumWidth(340)
        self.video_progress_label = QLabel("00:00/00:00")
        self.video_progress_label.setFont(QFont("Consolas", 11))
        self.process_progress = QProgressBar()
        self.process_progress.setFormat("处理进度: %p%")
        self.process_progress.setValue(0)
        self.process_progress.setMinimumWidth(220)
        self.process_progress.setMaximumWidth(340)
        terminal_title = QLabel("终端输出")
        terminal_title.setFont(QFont("微软雅黑", 15, QFont.Bold))
        self.terminal_text = QTextEdit()
        self.terminal_text.setReadOnly(True)
        self.terminal_text.setFont(QFont("Consolas", 12))  # 字体更小
        self.terminal_text.setMinimumHeight(420)
        self.terminal_text.setMaximumHeight(16777215)
        self.terminal_text.setMinimumWidth(260)
        self.terminal_text.setMaximumWidth(340)
        self.terminal_text.setStyleSheet("background: #f3f3f3; border-radius: 4px; color: #333; line-height: 1.1em;")
        info_layout.addWidget(self.status_label)
        info_layout.addWidget(self.detect_state_label)
        info_layout.addWidget(self.carinfo_label)
        info_layout.addWidget(self.video_progress)
        info_layout.addWidget(self.video_progress_label)
        info_layout.addWidget(self.process_progress)
        info_layout.addWidget(terminal_title)
        info_layout.addWidget(self.terminal_text)
        info_layout.addStretch()
        info_panel.setLayout(info_layout)
        info_panel.setMaximumWidth(480)

        main_layout.addWidget(op_panel)
        main_layout.addWidget(display_panel)
        main_layout.addWidget(info_panel)
        main_widget.setLayout(main_layout)
        if self.model_combo.count() > 0:
            self.load_model(self.model_combo.currentText())

        self.camera_btn.toggled.connect(self.on_input_source_changed)
        self.video_btn.toggled.connect(self.on_input_source_changed)
        self.image_btn.toggled.connect(self.on_input_source_changed)

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
        try:
            if self.net is None:
                self.status_label.setText("模型种类: --")
                self.detect_state_label.setText("检测状态: 未加载模型")
                self.set_op_feedback("未加载模型，请先选择模型。", "#e74c3c")
                return img
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            x = self.img_transforms(img_pil)
            x = x.unsqueeze(0).cuda(non_blocking=True)
            with torch.no_grad():
                out = self.net.half()(x.half()) if x.dtype == torch.float16 else self.net(x)
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
            result_img = img.copy()
            lanes = []
            lane_xs = []
            for i in range(out_j.shape[1]):
                lane_points = []
                xs = []
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            ppp = (int(out_j[k, i] * col_sample_w * img.shape[1] / 800) - 1,
                                   int(img.shape[0] * (self.row_anchor[self.cls_num_per_lane - 1 - k] / 288)) - 1)
                            lane_points.append(ppp)
                            xs.append(ppp[0])
                            cv2.circle(result_img, ppp, 5, (0, 0, 255), -1)
                if lane_points:
                    lanes.append(lane_points)
                    lane_xs.append(xs[-1] if xs else 0)
            valid_xs = [x for x in lane_xs if x > 0]
            detected_count = len(valid_xs)
            xs_display = valid_xs + [0] * (4 - len(valid_xs))
            html_log = ' | '.join([str(v) for v in xs_display])
            self.terminal_text.append(html_log)
            center_x = img.shape[1] // 2
            lane_index = 1
            if len(lanes) >= 2:
                all_ys = set()
                for lane in lanes:
                    for pt in lane:
                        all_ys.add(pt[1])
                all_ys = sorted(list(all_ys))
                between_counts = [0]*(len(lanes)-1)
                for y in all_ys:
                    xs_at_y = []
                    for lane in lanes:
                        if lane:
                            pt = min(lane, key=lambda p: abs(p[1]-y))
                            xs_at_y.append(pt[0])
                    xs_at_y = sorted(xs_at_y)
                    for i in range(len(xs_at_y)-1):
                        if xs_at_y[i] < center_x <= xs_at_y[i+1]:
                            between_counts[i] += 1
                if between_counts:
                    lane_index = between_counts.index(max(between_counts)) + 1
            # 新增：单独overlay用于横线，overlay_mask用于薄膜
            overlay = result_img.copy()
            lane_hlines_pts = []
            overlay_hline = result_img.copy()
            overlay_mask = result_img.copy()
            if len(valid_xs) >= 2 and lane_index <= len(lanes)-1:
                left_lane = lanes[lane_index-1]
                right_lane = lanes[lane_index]
                if left_lane and right_lane:
                    def interp_x_at_y(lane, target_y):
                        for i in range(len(lane)-1):
                            y1, y2 = lane[i][1], lane[i+1][1]
                            x1, x2 = lane[i][0], lane[i+1][0]
                            if (y1 - target_y) * (y2 - target_y) <= 0 and y1 != y2:
                                ratio = (target_y - y1) / (y2 - y1)
                                return int(x1 + ratio * (x2 - x1))
                        idx_closest = np.argmin([abs(p[1] - target_y) for p in lane])
                        return lane[idx_closest][0]
                    left_pts = left_lane[::-1]
                    right_pts = right_lane
                    for lp in left_pts:
                        rp = min(right_pts, key=lambda p: abs(p[1] - lp[1]))
                        if self.lane_hline_cb.isChecked():
                            cv2.line(overlay_hline, (lp[0], lp[1]), (rp[0], rp[1]), (60, 30, 10), 5)  # 深棕色加粗
                        lane_hlines_pts.append((lp, rp))
                    # 绿色薄膜为所有车道横线端点围成的区域
                    if self.green_mask_cb.isChecked() and len(lane_hlines_pts) > 1:
                        poly_pts = [pt[0] for pt in lane_hlines_pts] + [pt[1] for pt in lane_hlines_pts[::-1]]
                        pts = np.array(poly_pts, np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(overlay_mask, [pts], (0, 255, 0))
                    # 红色行驶线
                    if self.red_lane_cb.isChecked():
                        mid_points = []
                        for lp in left_pts:
                            rp = min(right_pts, key=lambda p: abs(p[1] - lp[1]))
                            mid_x = (lp[0] + rp[0]) // 2
                            mid_y = (lp[1] + rp[1]) // 2
                            mid_points.append((mid_x, mid_y))
                        if len(mid_points) > 1:
                            cv2.polylines(result_img, [np.array(mid_points, np.int32)], False, (0, 0, 255), 4)
            # 车头中心线
            if self.car_center_cb.isChecked():
                head_center_y = int(img.shape[0] * 0.8)
                cv2.line(result_img, (center_x, head_center_y - 20), (center_x, head_center_y + 20), (0, 255, 255), 4)
            # 绿色薄膜和车道横线可同时叠加
            if self.green_mask_cb.isChecked() and len(lane_hlines_pts) > 1:
                result_img = cv2.addWeighted(result_img, 0.7, overlay_mask, 0.3, 0)
            if self.lane_hline_cb.isChecked() and len(lane_hlines_pts) > 1:
                result_img = cv2.addWeighted(result_img, 0.7, overlay_hline, 0.5, 0)
            # 车辆信息区域显示当前车道类型和动态推荐
            # 计算推荐转向角度
            recommend_text = ""
            if len(valid_xs) >= 2 and lane_index <= len(lanes)-1:
                left_lane = lanes[lane_index-1]
                right_lane = lanes[lane_index]
                if left_lane and right_lane:
                    # 取底部点
                    left_x = left_lane[-1][0]
                    right_x = right_lane[-1][0]
                    lane_center = (left_x + right_x) // 2
                    offset = center_x - lane_center
                    lane_width = abs(right_x - left_x)
                    # 动态弹性空间
                    threshold = max(30, lane_width // 10)
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
            if detected_count == 4:
                lane_type = ["左车道", "中车道", "右车道"]
                lane_name = lane_type[lane_index-1] if 1 <= lane_index <= 3 else "未知"
            elif detected_count == 3:
                lane_type = ["左车道", "右车道"]
                lane_name = lane_type[lane_index-1] if 1 <= lane_index <= 2 else "未知"
            elif detected_count == 2:
                lane_name = "单行道"
            else:
                lane_name = "未知"
            self.status_label.setText(f"模型种类: {self.current_model_type}")
            self.detect_state_label.setText("检测状态: 已检测")
            # 车辆信息区最上方只显示检测到几条车道线
            lane_count_html = (
                f"<span style='color:#e74c3c;font-size:18px;font-weight:bold;'>检测到"
                f"<span style='color:#2980b9;font-size:18px;font-weight:bold;'>{detected_count}</span>"
                f"条车道线</span>"
            )
            self.carinfo_label.setText(
                f"{lane_count_html}<br>"
                f"<span style='color:#e74c3c;font-weight:bold;'>当前车道: {lane_name}</span><br>"
                f"{recommend_text}"
            )
            return result_img
        except Exception as e:
            self.terminal_text.append(f"检测过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return img

    def update_frame(self):
        try:
            if self.cap is None:
                return
                
            # 尝试读取帧
            try:
                ret, frame = self.cap.read()
            except Exception as e:
                self.append_terminal_log(f"读取视频帧时出错: {str(e)}")
                # 视频读取出错，释放资源并重置状态
                self.timer.stop()
                if self.cap:
                    self.cap.release()
                    self.cap = None
                self.start_btn.setText("开始检测")
                self.start_btn.setStyleSheet(self.get_start_btn_style("start"))
                self.detect_state_label.setText("检测状态: 视频读取错误")
                self.set_op_feedback("视频读取错误，请重新选择视频或检查文件", "#e74c3c")
                return
                
            if ret:
                # 成功读取帧，处理并显示
                self.last_raw_frame = frame.copy()
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 0.15:
                    self.fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.start_time = time.time()
                processed_frame = self.detect_and_draw_lanes(frame)
                self.display_results(processed_frame)
                
                # 更新进度条和时间信息
                if self.cap and self.cap.isOpened() and self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                    try:
                        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                        self.video_progress.setMaximum(total)
                        self.video_progress.setValue(pos)
                        # 更新时间标签
                        cur_sec = int(self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
                        total_sec = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1, self.cap.get(cv2.CAP_PROP_FPS)))
                        self.video_progress_label.setText(f"{cur_sec//60:02d}:{cur_sec%60:02d}/{total_sec//60:02d}:{total_sec%60:02d}")
                    except Exception as e:
                        # 忽略进度更新错误，不影响主要功能
                        pass
            else:
                # 视频播放结束，先停止计时器和释放资源
                self.timer.stop()
                if self.cap:
                    self.cap.release()
                    self.cap = None
                
                # 重置状态变量
                self.frame_count = 0
                self.fps = 0
                self.last_raw_frame = None
                
                # 自动播放下一个视频
                video_list = sorted(glob.glob(os.path.join('my-video', 'test-vedio', '*.mp4')))
                if not video_list:
                    self.set_op_feedback("未找到视频文件", "#e74c3c")
                    self.start_btn.setText("开始检测")
                    self.start_btn.setStyleSheet(self.get_start_btn_style("start"))
                    return
                    
                now_path = getattr(self, 'current_video_path', self.default_video)
                if now_path in video_list:
                    idx = video_list.index(now_path)
                    next_idx = (idx + 1) % len(video_list)
                    next_video = video_list[next_idx]
                else:
                    next_video = video_list[0] if video_list else None
                    
                if next_video and os.path.exists(next_video):
                    # 设置新视频路径
                    self.current_video_path = next_video
                    self.path_label.setText(f"已选择视频: {os.path.relpath(next_video)}")
                    self.set_op_feedback(f"正在切换到下一个视频: {os.path.basename(next_video)}", "#2980b9")
                    
                    # 延迟一小段时间再启动新视频，确保资源完全释放
                    QTimer.singleShot(100, lambda: self.process_video(next_video))
                else:
                    self.start_btn.setText("开始检测")
                    self.start_btn.setStyleSheet(self.get_start_btn_style("start"))
                    self.set_op_feedback("视频播放完毕", "#229954")
                return
        except Exception as e:
            print(f"更新帧时出错: {str(e)}")
            self.append_terminal_log(f"更新帧时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 确保停止计时器和释放资源
            self.timer.stop()
            if self.cap:
                self.cap.release()
                self.cap = None
                
            # 重置状态变量
            self.frame_count = 0
            self.fps = 0
            self.last_raw_frame = None
            self.last_main_left = None
            self.last_main_right = None
            self.last_main_left_smoothed = None
            self.last_main_right_smoothed = None
            
            # 更新UI状态
            self.start_btn.setText("开始检测")
            self.start_btn.setStyleSheet(self.get_start_btn_style("start"))
            self.status_label.setText(f"模型种类: {self.current_model_type if self.current_model_type else '--'}")
            self.detect_state_label.setText("检测状态: 处理出错")
            self.set_op_feedback(f"视频处理出错: {str(e)}", "#e74c3c")
            self.process_progress.setValue(0)

    def display_results(self, processed, original=None):
        def set_label_pixmap(label, img):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qt_img)
            label.setPixmap(pix.scaled(label.width(), label.height(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
            label.setMinimumSize(label.width(), label.height())
            label.setMaximumSize(label.width(), label.height())
        if self.cap is not None and self.cap.isOpened():
            if self.last_raw_frame is not None:
                set_label_pixmap(self.raw_label, self.last_raw_frame)
            else:
                set_label_pixmap(self.raw_label, processed)
        elif self.image_btn.isChecked() and original is not None:
            set_label_pixmap(self.raw_label, original)
        else:
            set_label_pixmap(self.raw_label, processed)
        set_label_pixmap(self.det_label, processed)
        # 帧率显示到视频左上角
        if self.show_fps_btn.isChecked():
            self.raw_fps.setParent(self.raw_label)
            self.raw_fps.move(10, 10)
            self.raw_fps.setText(f"FPS: {self.fps:.1f}")
            self.raw_fps.show()
            self.det_fps.setParent(self.det_label)
            self.det_fps.move(10, 10)
            self.det_fps.setText(f"FPS: {self.fps:.1f}")
            self.det_fps.show()
        else:
            self.raw_fps.hide()
            self.det_fps.hide()

    def process_video_frames(self, video_path, output_dir):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            QMessageBox.critical(self, "错误", f"无法打开视频文件: {video_path}")
            self.set_op_feedback("无法打开视频文件。", "#e74c3c")
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(output_dir, f"output-{base_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        frame_idx = 0
        self.process_progress.setValue(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed = self.detect_and_draw_lanes(frame)
            writer.write(processed)
            frame_img_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(frame_img_path, processed)
            frame_idx += 1
            if total > 0:
                progress = int(frame_idx / total * 100)
                self.process_progress.setValue(progress)
        cap.release()
        writer.release()
        self.process_progress.setValue(100)
        self.set_op_feedback(f"视频检测结果已保存到: {output_video_path}", "#229954")
        self.append_terminal_log(f"视频检测结果已保存到: {output_video_path}")
        QMessageBox.information(self, "处理完成", f"检测结果已保存，输出视频在: {output_video_path}\n每帧图片在: {output_dir}")

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

    def stop_current_detection(self):
        """停止当前的检测（视频或摄像头）并释放资源。"""
        was_active = self.timer.isActive()
        if was_active:
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        # 重置UI状态
        self.start_btn.setText("开始检测")
        self.start_btn.setStyleSheet(self.get_start_btn_style("start"))
        # 仅在确实停止了活动检测时更新状态标签和日志
        if was_active:
            self.detect_state_label.setText("检测状态: 已停止")
            self.append_terminal_log("检测已停止，资源已释放。")
        # 重置进度条总是安全的
        self.video_progress.setValue(0)
        self.video_progress_label.setText("00:00/00:00")
        self.process_progress.setValue(0)
        # 可以选择性地清除图像显示区域
        # self.raw_label.clear()
        # self.det_label.clear()
        return was_active # 返回之前是否在活动状态

    def select_file(self):
        self.stop_current_detection() # <--- 添加停止逻辑
        if self.video_btn.isChecked():
            file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mkv)")
            if file_name:
                rel_path = os.path.relpath(file_name)
                self.path_label.setText(f"已选择视频: {rel_path}")
                self.set_op_feedback("已选择视频文件。", "#229954")
                self.process_video(file_name)
        elif self.image_btn.isChecked():
            file_name, _ = QFileDialog.getOpenFileName(self, "选择图片文件", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
            if file_name:
                rel_path = os.path.relpath(file_name)
                self.path_label.setText(f"已选择图片: {rel_path}")
                self.set_op_feedback("已选择图片文件。", "#229954")
                self.process_image(file_name)

    def process_video(self, video_path):
        # 停止逻辑已移至调用处 (select_file, play_prev/next_video, toggle_detection)
        # 重置状态变量
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.last_raw_frame = None
        self.last_main_left = None
        self.last_main_right = None
        self.last_main_left_smoothed = None
        self.last_main_right_smoothed = None
        
        # 打开新视频
        try:
            self.cap = cv2.VideoCapture(video_path)
            self.current_video_path = video_path
            rel_path = os.path.relpath(video_path)
            self.path_label.setText(f"已选择视频: {rel_path}")
            if not self.cap.isOpened():
                QMessageBox.critical(self, "错误", f"无法打开视频文件: {video_path}")
                self.set_op_feedback("无法打开视频文件。", "#e74c3c")
                return False
                
            # 启动计时器，开始处理
            self.timer.start(30)
            self.start_btn.setText("停止检测")
            self.start_btn.setStyleSheet(self.get_start_btn_style("stop"))
            self.status_label.setText(f"模型种类: {self.current_model_type}")
            self.detect_state_label.setText("检测状态: 正在处理...")
            self.set_op_feedback(f"正在处理视频: {os.path.basename(video_path)}", "#2980b9")
            return True
        except Exception as e:
            self.set_op_feedback(f"打开视频时出错: {str(e)}", "#e74c3c")
            return False

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            QMessageBox.critical(self, "错误", f"无法打开图片文件: {image_path}")
            self.set_op_feedback("无法打开图片文件。", "#e74c3c")
            return
        processed = self.detect_and_draw_lanes(img)
        self.display_results(processed, original=img)
        self.status_label.setText(f"模型种类: {self.current_model_type}")
        self.detect_state_label.setText("检测状态: 图片检测完成")
        self.set_op_feedback("图片检测完成。", "#229954")

    def start_default_detection(self):
        if os.path.exists(self.default_video):
            self.process_video(self.default_video)
            self.timer.start(25)
            self.start_btn.setText("停止检测")
            self.start_btn.setStyleSheet(self.get_start_btn_style("stop"))
            self.status_label.setText(f"模型种类: {self.current_model_type}")
            self.detect_state_label.setText("检测状态: 正在处理...")
            self.set_op_feedback("正在处理默认视频...", "#2980b9")

    def get_start_btn_style(self, state):
        if state == "start":
            return "background-color: #2a82da; color: white; padding: 8px; border-radius: 4px; font-weight: bold; font-size:13px;"
        elif state == "stop":
            return "background-color: #e74c3c; color: white; padding: 8px; border-radius: 4px; font-weight: bold; font-size:13px;"
        elif state == "resume":
            return "background-color: #f0ad4e; color: white; padding: 8px; border-radius: 4px; font-weight: bold; font-size:13px;"
        return ""

    def toggle_detection(self):
        if self.timer.isActive(): # 检查计时器是否活动
            self.stop_current_detection() # <--- 使用统一的停止函数
            # self.set_op_feedback("检测已停止。", "#e67e22") # stop_current_detection 会记录日志
        else:
            # 处理后检测模式
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
            # 实时检测模式
            else:
                if self.camera_btn.isChecked():
                    self.open_camera()
                    return
                if self.video_btn.isChecked() and self.path_label.text().startswith("已选择视频: "):
                    video_path = self.path_label.text().replace("已选择视频: ", "")
                    if os.path.exists(video_path):
                        self.process_video(video_path)
                        return
                elif self.image_btn.isChecked() and self.path_label.text().startswith("已选择图片: "):
                    image_path = self.path_label.text().replace("已选择图片: ", "")
                    if os.path.exists(image_path):
                        self.process_image(image_path)
                        return
                # 否则才用默认视频
                self.start_default_detection()

    def on_input_source_changed(self):
        # 切换输入源时，释放上一个输入源，重置状态
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.setText("开始检测")
        self.start_btn.setStyleSheet(self.get_start_btn_style("start"))
        self.detect_state_label.setText("检测状态: 等待检测")
        self.status_label.setText(f"模型种类: {self.current_model_type}")
        self.last_raw_frame = None
        # 摄像头
        if self.camera_btn.isChecked():
            self.path_label.setText("未选择文件")
            self.set_op_feedback("已切换到摄像头输入。", "#229954")
            self.save_result_btn.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            if self.save_result_btn.isChecked():
                self.set_op_feedback("保存检测结果暂不支持摄像头。", "#e67e22")
            # 不自动打开摄像头，等用户点击开始检测
        # 视频
        elif self.video_btn.isChecked():
            self.set_op_feedback("请选择视频文件。", "#229954")
            self.save_result_btn.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
        # 图片
        elif self.image_btn.isChecked():
            self.set_op_feedback("请选择图片文件。", "#229954")
            self.save_result_btn.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
        else:
            self.save_result_btn.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)

    def open_camera(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "错误", "无法打开摄像头")
            self.set_op_feedback("无法打开摄像头。", "#e74c3c")
            return
        self.timer.start(30)
        self.start_btn.setText("停止检测")
        self.start_btn.setStyleSheet(self.get_start_btn_style("stop"))
        self.status_label.setText(f"模型种类: {self.current_model_type}")
        self.detect_state_label.setText("检测状态: 正在处理...")
        self.set_op_feedback("正在使用摄像头实时检测...", "#2980b9")

    def seek_video(self, pos):
        if self.cap and self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = self.cap.read()
            if ret:
                processed_frame = self.detect_and_draw_lanes(frame)
                self.display_results(processed_frame)

    def play_prev(self):
        if self.video_btn.isChecked():
            self.play_prev_video()
        elif self.image_btn.isChecked():
            self.play_prev_image()

    def play_next(self):
        if self.video_btn.isChecked():
            self.play_next_video()
        elif self.image_btn.isChecked():
            self.play_next_image()

    def play_prev_image(self):
        # 切换到上一个图片（在当前图片所在目录内遍历）
        now_path = self.path_label.text().replace("已选择图片: ", "") if self.path_label.text().startswith("已选择图片: ") else None
        if now_path and os.path.exists(now_path):
            current_dir = os.path.dirname(now_path)
            img_list = sorted(
                glob.glob(os.path.join(current_dir, '*.jpg')) +
                glob.glob(os.path.join(current_dir, '*.png')) +
                glob.glob(os.path.join(current_dir, '*.jpeg')) +
                glob.glob(os.path.join(current_dir, '*.bmp'))
            )
            if now_path in img_list:
                idx = img_list.index(now_path)
                prev_idx = (idx - 1) % len(img_list)
                prev_img = img_list[prev_idx]
            else:
                prev_img = img_list[0] if img_list else None
            if prev_img and os.path.exists(prev_img):
                rel_path = os.path.relpath(prev_img)
                self.path_label.setText(f"已选择图片: {rel_path}")
                self.append_terminal_log(f"切换到上一个图片: {rel_path}")
                self.process_image(prev_img)

    def play_next_image(self):
        # 切换到下一个图片（在当前图片所在目录内遍历）
        now_path = self.path_label.text().replace("已选择图片: ", "") if self.path_label.text().startswith("已选择图片: ") else None
        if now_path and os.path.exists(now_path):
            current_dir = os.path.dirname(now_path)
            img_list = sorted(
                glob.glob(os.path.join(current_dir, '*.jpg')) +
                glob.glob(os.path.join(current_dir, '*.png')) +
                glob.glob(os.path.join(current_dir, '*.jpeg')) +
                glob.glob(os.path.join(current_dir, '*.bmp'))
            )
            if now_path in img_list:
                idx = img_list.index(now_path)
                next_idx = (idx + 1) % len(img_list)
                next_img = img_list[next_idx]
            else:
                next_img = img_list[0] if img_list else None
            if next_img and os.path.exists(next_img):
                rel_path = os.path.relpath(next_img)
                self.path_label.setText(f"已选择图片: {rel_path}")
                self.append_terminal_log(f"切换到下一个图片: {rel_path}")
                self.process_image(next_img)

    def save_detection_result(self):
        # 保存检测结果，视频和图片可用，摄像头禁用
        if self.camera_btn.isChecked():
            QMessageBox.information(self, "提示", "摄像头输入不支持保存检测结果！")
            return
        if self.video_btn.isChecked() and self.path_label.text().startswith("已选择视频: "):
            video_path = self.path_label.text().replace("已选择视频: ", "")
            if os.path.exists(video_path):
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_dir = os.path.join("my-video", "output", f"output-{base_name}")
                os.makedirs(output_dir, exist_ok=True)
                self.process_video_frames(video_path, output_dir)
        elif self.image_btn.isChecked() and self.path_label.text().startswith("已选择图片: "):
            image_path = self.path_label.text().replace("已选择图片: ", "")
            if os.path.exists(image_path):
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_dir = os.path.join("my-video", "output", f"output-{base_name}")
                os.makedirs(output_dir, exist_ok=True)
                img = cv2.imread(image_path)
                processed = self.detect_and_draw_lanes(img)
                out_path = os.path.join(output_dir, os.path.basename(image_path))
                cv2.imwrite(out_path, processed)
                self.process_progress.setValue(100)
                self.set_op_feedback(f"图片检测结果已保存到: {out_path}", "#229954")
                self.append_terminal_log(f"图片检测结果已保存到: {out_path}")
                QMessageBox.information(self, "处理完成", f"检测结果已保存，输出图片在: {out_path}")

    def on_save_result_mode(self):
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

    def play_prev_video(self):
        self.stop_current_detection()
        self.process_progress.setValue(0)
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.last_raw_frame = None

        now_path = getattr(self, 'current_video_path', self.default_video)
        # 获取当前视频所在目录
        current_dir = os.path.dirname(now_path)
        # 支持多种视频格式
        video_list = sorted(
            glob.glob(os.path.join(current_dir, '*.mp4')) +
            glob.glob(os.path.join(current_dir, '*.avi')) +
            glob.glob(os.path.join(current_dir, '*.mkv')) +
            glob.glob(os.path.join(current_dir, '*.mov'))
        )
        if not video_list:
            self.set_op_feedback(f"在目录 {os.path.basename(current_dir)} 中未找到视频文件", "#e74c3c")
            return
        if now_path in video_list:
            idx = video_list.index(now_path)
            prev_idx = (idx - 1) % len(video_list)
            prev_video = video_list[prev_idx]
        else:
            prev_video = video_list[0]
        if prev_video and os.path.exists(prev_video):
            rel_path = os.path.relpath(prev_video)
            self.current_video_path = prev_video
            self.path_label.setText(f"已选择视频: {rel_path}")
            self.append_terminal_log(f"切换到上一个视频: {rel_path}")
            self.process_video(prev_video)

    def play_next_video(self):
        self.stop_current_detection()
        self.process_progress.setValue(0)
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        self.last_raw_frame = None

        now_path = getattr(self, 'current_video_path', self.default_video)
        current_dir = os.path.dirname(now_path)
        video_list = sorted(
            glob.glob(os.path.join(current_dir, '*.mp4')) +
            glob.glob(os.path.join(current_dir, '*.avi')) +
            glob.glob(os.path.join(current_dir, '*.mkv')) +
            glob.glob(os.path.join(current_dir, '*.mov'))
        )
        if not video_list:
            self.set_op_feedback("未找到视频文件", "#e74c3c")
            return
        if now_path in video_list:
            idx = video_list.index(now_path)
            next_idx = (idx + 1) % len(video_list)
            next_video = video_list[next_idx]
        else:
            next_video = video_list[0]
        if next_video and os.path.exists(next_video):
            rel_path = os.path.relpath(next_video)
            self.current_video_path = next_video
            self.path_label.setText(f"已选择视频: {rel_path}")
            self.append_terminal_log(f"切换到下一个视频: {rel_path}")
            self.process_video(next_video)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LaneDetectionUI()
    window.show()
    sys.exit(app.exec_())