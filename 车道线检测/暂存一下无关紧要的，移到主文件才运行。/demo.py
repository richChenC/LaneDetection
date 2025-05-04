import torch, os, cv2
# 导入PyTorch、os、OpenCV等常用库
from model.model import parsingNet
# 导入自定义的车道线检测模型parsingNet
from utils.common import merge_config
# 导入配置合并工具
from utils.dist_utils import dist_print
# 导入分布式打印工具
import torch
import scipy.special, tqdm
# 导入scipy的特殊函数和tqdm进度条
import numpy as np
import torchvision.transforms as transforms
# 导入numpy和torchvision的图像变换工具
from data.dataset import LaneTestDataset
# 导入自定义的数据集类
from data.constant import culane_row_anchor, tusimple_row_anchor
# 导入CULane和Tusimple数据集的锚点常量

# =================== 宏定义区 ===================
VIDEO_FILE = 'ygc_test3.avi'  # 只需填写视频文件名，默认在my-vedio\目录下
GRAY_INPUT = False  # 是否对视频帧增强对比度后再识别
# 可视化元素开关
SWITCHES = {
    'show_green_mask': True,                # 显示绿色薄膜
    'show_red_lane': True,                  # 显示红色行驶线
    'show_lane_center_short_line': True,    # 显示车道中间短竖线
    'show_car_center_short_line': True      # 显示车头屏幕中间竖短线
}
# ==============================================

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args, cfg = merge_config()
    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    # =================== 视频帧提取与预处理区 ===================
    # 1. 根据所选模型确定目标分辨率
    if cfg.dataset == 'CULane':
        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError
    # 2. 构造输入输出路径
    video_path = os.path.join('my-video', VIDEO_FILE)
    video_name, video_ext = os.path.splitext(VIDEO_FILE)
    frame_dir = os.path.join('my-video', 'temp_frames_' + video_name)
    os.makedirs(frame_dir, exist_ok=True)
    txt_path = os.path.join('my-video', f'{video_name}_input.txt')
    output_dir = os.path.join('my-video', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f'output{VIDEO_FILE}')
    # 3. 提取视频帧并resize到目标分辨率，生成txt路径文件
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    idx = 0
    with open(txt_path, 'w') as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (img_w, img_h))
            # 如果宏开关为True，对视频帧进行对比度增强（如CLAHE），再转回3通道
            if GRAY_INPUT:
                # 转为YUV，增强Y通道对比度
                yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                yuv[:,:,0] = clahe.apply(yuv[:,:,0])
                frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            frame_file = os.path.join(frame_dir, f'{idx:05d}.jpg')
            cv2.imwrite(frame_file, frame)
            f.write(f'{os.path.abspath(frame_file)}\n')
            frame_list.append(frame_file)
            idx += 1
    cap.release()
    # =================== 视频帧提取与预处理区 END ===================
    # 1. 自动将输入视频帧提取为图片，resize到模型要求分辨率，保存到临时目录。
    # 2. 可选：转为灰度图再转回3通道，便于模型输入。
    # 3. 生成txt路径文件，供后续推理使用。
    # =============================================================

    # =================== 模型加载区 ===================
    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError
    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,4),
                    use_aux=False).cuda()
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    # =================== 模型加载区 END ===================

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # =================== 数据集加载区 ===================
    splits = [txt_path]
    datasets = [LaneTestDataset(cfg.data_root, txt_path, img_transform=img_transforms)]
    # =================== 数据集加载区 END ===================

    # =================== 推理与可视化输出区 ===================
    for split, dataset in zip(splits, datasets):
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
        # 根据视频扩展名选择编码器
        if video_ext.lower() == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        vout = cv2.VideoWriter(output_video_path, fourcc , 30.0, (img_w, img_h))
        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            imgs = imgs.cuda()
            with torch.no_grad():
                out = net(imgs)
            col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
            col_sample_w = col_sample[1] - col_sample[0]
            out_j = out[0].data.cpu().numpy()
            out_j = out_j[:, ::-1, :]
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(cfg.griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == cfg.griding_num] = 0
            out_j = loc
            vis = cv2.imread(names[0])
            # ========== 可视化增强区 ========== 
            # 1. 绿色薄膜
            if SWITCHES['show_green_mask']:
                green_mask = np.zeros_like(vis, dtype=np.uint8)
                green_mask[:] = (0, 255, 0)
                vis = cv2.addWeighted(vis, 0.7, green_mask, 0.3, 0)
            # 2. 红色行驶线（示例：画在画面正中）
            if SWITCHES['show_red_lane']:
                cv2.line(vis, (img_w//2, 0), (img_w//2, img_h), (0,0,255), 2)
            # 3. 车道中间短竖线（示例：画在画面1/4和3/4处）
            if SWITCHES['show_lane_center_short_line']:
                for x in [img_w//4, 3*img_w//4]:
                    cv2.line(vis, (x, img_h//2-20), (x, img_h//2+20), (255,255,0), 3)
            # 4. 车头屏幕中间竖短线（示例：画在画面底部中央）
            if SWITCHES['show_car_center_short_line']:
                cv2.line(vis, (img_w//2, img_h-40), (img_w//2, img_h), (0,255,255), 4)
            # ========== 可视化增强区 END ==========
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                            cv2.circle(vis,ppp,5,(0,255,0),-1)
            vout.write(vis)
        vout.release()
    # =================== 推理与可视化输出区 END ===================
    print(f'输出视频已保存到: {os.path.abspath(output_video_path)}')