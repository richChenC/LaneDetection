import cv2
import os
from drllane_carla_rl.lane_det.detector import LaneDetector

# 单张图片可视化
def visualize_image(img_path, model_path='../../车道线检测/my-model/culane_18.pth'):
    img = cv2.imread(img_path)
    detector = LaneDetector(model_path)
    lanes, lane_mask = detector.detect(img)
    vis = detector.visualize(img, lanes, lane_mask)
    cv2.imshow('Lane Detection Visualization', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 批量图片可视化与结果保存
def batch_visualize_images(img_dir, save_dir, model_path='../../车道线检测/my-model/culane_18.pth'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    detector = LaneDetector(model_path)
    for fname in os.listdir(img_dir):
        if fname.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(img_dir, fname)
            img = cv2.imread(img_path)
            lanes, lane_mask = detector.detect(img)
            vis = detector.visualize(img, lanes, lane_mask)
            save_path = os.path.join(save_dir, fname)
            cv2.imwrite(save_path, vis)

# 单个视频可视化
def visualize_video(video_path, model_path='../../车道线检测/my-model/culane_18.pth'):
    cap = cv2.VideoCapture(video_path)
    detector = LaneDetector(model_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        lanes, lane_mask = detector.detect(frame)
        vis = detector.visualize(frame, lanes, lane_mask)
        cv2.imshow('Lane Detection Video', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# 批量视频推理与结果保存
def batch_infer_videos(video_dir, save_dir, model_path='../../车道线检测/my-model/culane_18.pth'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    detector = LaneDetector(model_path)
    for fname in os.listdir(video_dir):
        if fname.lower().endswith(('.mp4', '.avi')):
            video_path = os.path.join(video_dir, fname)
            cap = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_path = os.path.join(save_dir, fname)
            out = None
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                lanes, lane_mask = detector.detect(frame)
                vis = detector.visualize(frame, lanes, lane_mask)
                if out is None:
                    h, w = vis.shape[:2]
                    out = cv2.VideoWriter(out_path, fourcc, 20, (w, h))
                out.write(vis)
            cap.release()
            if out:
                out.release() 