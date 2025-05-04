import cv2
import numpy as np
import torch
from PIL import Image
import scipy.special

def draw_lanes_on_image(img, net, img_transforms, row_anchor, cls_num_per_lane):
    """
    复用2-lane_detection_ui.py的可视化逻辑，输入原始BGR图像、lane net、transforms、row_anchor、cls_num_per_lane，返回可视化后图片。
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    x = img_transforms(img_pil)
    x = x.unsqueeze(0).cuda(non_blocking=True)
    with torch.no_grad():
        out = net.half()(x.half()) if x.dtype == torch.float16 else net(x)
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
                           int(img.shape[0] * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                    lane_points.append(ppp)
                    xs.append(ppp[0])
                    cv2.circle(result_img, ppp, 5, (0, 0, 255), -1)
        if lane_points:
            lanes.append(lane_points)
            lane_xs.append(xs[-1] if xs else 0)
    # 可选：绿色mask、红色中线等可扩展
    return result_img, lanes 