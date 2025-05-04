import torch
import cv2
import numpy as np
from drllane_carla_rl.lane_det.ultrafastlane.model import parsingNet
import torchvision.transforms as transforms
from PIL import Image
from drllane_carla_rl.utils.visualize_rich import detect_and_draw_lanes

# 统一参数定义，和UI/录制完全一致
CLS_NUM_PER_LANE = 18
ROW_ANCHOR = np.array([121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287])
IMG_TRANSFORMS = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

class LaneDetector:
    """
    车道线检测主类，支持加载parsingNet模型，对输入图片进行车道线检测，并输出可视化结果。
    """
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        初始化，加载parsingNet模型
        :param model_path: 预训练模型路径
        :param device: 运行设备（cuda或cpu）
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = parsingNet(backbone='18', cls_dim=(201, 18, 4), use_aux=False)
        state_dict = torch.load(model_path, map_location=self.device)
        # 兼容module.前缀
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.cls_num_per_lane = CLS_NUM_PER_LANE
        self.row_anchor = ROW_ANCHOR
        self.img_transforms = IMG_TRANSFORMS
        self.prev_lanes = None  # 用于平滑

    def detect_and_draw(self, img):
        # 支持BGR/RGB输入
        if img.shape[2] == 3 and np.mean(img[..., 0]) > np.mean(img[..., 2]):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img.copy()
        pil_img = Image.fromarray(img_rgb)
        input_tensor = self.img_transforms(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(input_tensor)
        vis_img, lanes = detect_and_draw_lanes(
            img, self.model, self.img_transforms, self.row_anchor, self.cls_num_per_lane,
            show_green_mask=True, show_hline=False, show_red_lane=True, show_car_center=True
        )
        self.prev_lanes = lanes
        return vis_img, lanes

    def get_lanes(self, img):
        # 只返回点坐标
        if img.shape[2] == 3 and np.mean(img[..., 0]) > np.mean(img[..., 2]):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img.copy()
        pil_img = Image.fromarray(img_rgb)
        input_tensor = self.img_transforms(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(input_tensor)
        vis_img, lanes = detect_and_draw_lanes(
            img, self.model, self.img_transforms, self.row_anchor, self.cls_num_per_lane,
            show_green_mask=True, show_hline=False, show_red_lane=True, show_car_center=True
        )
        self.prev_lanes = lanes
        return vis_img, lanes

    def visualize(self, img: np.ndarray, lanes, lane_mask=None):
        # 可视化（示例，需根据后处理结果绘制）
        vis = img.copy()
        # 这里只做占位，实际应根据lanes内容绘制点/线
        if lane_mask is not None:
            vis[lane_mask > 0] = [0, 255, 0]
        return vis 