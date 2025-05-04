import cv2
import numpy as np
import torch
from PIL import Image
import scipy.special

def detect_and_draw_lanes(img, net, img_transforms, row_anchor, cls_num_per_lane,
                         show_green_mask=True, show_hline=False, show_red_lane=True, show_car_center=True):
    """
    车道线检测可视化，完全复用UI/录制的后处理和可视化逻辑。
    """
    input_w, input_h = 800, 288
    h, w = img.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    x = img_transforms(img_pil)
    x = x.unsqueeze(0)
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
        net = net.cuda()
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
    out_j_idx = np.argmax(out_j, axis=0)
    loc[out_j_idx == 200] = 0
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
                    # y坐标严格与UI/录制一致
                    y_pt = int(img.shape[0] * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1
                    x_pt = int(out_j[k, i] * col_sample_w * img.shape[1] / 800) - 1
                    lane_points.append((x_pt, y_pt))
                    xs.append(x_pt)
                    cv2.circle(result_img, (x_pt, y_pt), 5, (0, 0, 255), -1)
        if lane_points:
            lanes.append(lane_points)
            lane_xs.append(xs[-1] if xs else 0)
    # 绿色薄膜
    if show_green_mask and len(lanes) >= 2:
        left_pts = np.array(lanes[0], dtype=np.int32)
        right_pts = np.array(lanes[-1][::-1], dtype=np.int32)
        poly_pts = np.vstack([left_pts, right_pts])
        cv2.fillPoly(result_img, [poly_pts], (0, 255, 0))
        result_img = cv2.addWeighted(result_img, 0.4, img, 0.6, 0)
    # 红色中心线
    if show_red_lane and len(lanes) >= 2:
        for k in range(min(len(lanes[0]), len(lanes[-1]))):
            x_c = int((lanes[0][k][0] + lanes[-1][k][0]) / 2)
            y_c = int((lanes[0][k][1] + lanes[-1][k][1]) / 2)
            cv2.circle(result_img, (x_c, y_c), 3, (0, 0, 255), -1)
        for k in range(1, min(len(lanes[0]), len(lanes[-1]))):
            x_c1 = int((lanes[0][k-1][0] + lanes[-1][k-1][0]) / 2)
            y_c1 = int((lanes[0][k-1][1] + lanes[-1][k-1][1]) / 2)
            x_c2 = int((lanes[0][k][0] + lanes[-1][k][0]) / 2)
            y_c2 = int((lanes[0][k][1] + lanes[-1][k][1]) / 2)
            cv2.line(result_img, (x_c1, y_c1), (x_c2, y_c2), (0, 0, 255), 2)
    # 车头中心线
    if show_car_center:
        cv2.line(result_img, (w//2, h-1), (w//2, h//2), (0, 255, 255), 2)
    # 横线（默认不画）
    if show_hline:
        for k in range(len(row_anchor)):
            y = int(img.shape[0] * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1
            cv2.line(result_img, (0, y), (w, y), (255, 255, 0), 1)
    return result_img, lanes 