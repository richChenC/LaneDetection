import sys
import os
print("sys.path:")
for p in sys.path:
    print(p)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import carla
import numpy as np
import cv2
import time
import queue
import threading
import traceback
import logging
import glob
import random

# ========== 参数设置 ==========
MP4_SAVE_DIR = r'my-video/carla-video-mp4'
AVI_SAVE_DIR = r'my-video/carla-video-avi'
VIDEO_NAME_PREFIX = 'carla'
MP4_EXT = '.mp4'
AVI_EXT = '.avi'
FPS = 30
WIDTH, HEIGHT = 1280, 640
RECORD_SECONDS = 20

# 固定地图为 Town02
MAP_NAME = 'Town02'

for d in [MP4_SAVE_DIR, AVI_SAVE_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# 屏蔽carla的INFO和WARNING日志
logging.getLogger('carla').setLevel(logging.ERROR)

def get_carla_client(host='localhost', port=2000):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    return client

def cleanup_actors(world):
    actors = world.get_actors()
    for actor in actors:
        if actor.type_id.startswith('vehicle.') or actor.type_id.startswith('sensor.camera.'):
            try:
                actor.destroy()
            except Exception:
                pass

def switch_to_map(client, map_name):
    current_map = client.get_world().get_map().name
    if map_name.lower() not in current_map.lower():
        print(f'正在切换地图到 {map_name} ...')
        try:
            client.load_world(map_name)
        except RuntimeError as e:
            print(f'切换地图失败: {e}')
            sys.exit(2)
        time.sleep(2)

def spawn_vehicle(world):
    blueprint_library = world.get_blueprint_library()
    bp = np.random.choice(blueprint_library.filter('vehicle.*'))
    spawn_points = world.get_map().get_spawn_points()
    spawn_idx = 11  # 固定为第12号点
    spawn_point = spawn_points[spawn_idx]
    vehicle = world.spawn_actor(bp, spawn_point)
    return vehicle, spawn_point

def spawn_camera(world, vehicle):
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')  # RGB彩色摄像头
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '640')
    camera_bp.set_attribute('fov', '90')
    # 驾驶员视角，能看到部分车头
    camera_transform = carla.Transform(
        carla.Location(x=0.5, y=0.0, z=1.4),  # x=0.5 更靠近驾驶员
        carla.Rotation(pitch=-10, yaw=0, roll=0)  # pitch=-10 俯视
    )
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    return camera

def preprocess_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]  # BGR for OpenCV
    return rgb

def record_one_video():
    client = get_carla_client()
    switch_to_map(client, MAP_NAME)
    world = client.get_world()
    cleanup_actors(world)
    spawn_points = world.get_map().get_spawn_points()
    spawn_idx = 11  # 固定为第12号点
    pattern = os.path.join(MP4_SAVE_DIR, f"{VIDEO_NAME_PREFIX}-{MAP_NAME}-spawn{spawn_idx+1}-*.mp4")
    exist_files = glob.glob(pattern)
    next_idx = len(exist_files) + 1
    video_name = f"{VIDEO_NAME_PREFIX}-{MAP_NAME}-spawn{spawn_idx+1}-{next_idx:04d}"
    # 固定为特斯拉车型
    blueprint_library = world.get_blueprint_library()
    tesla_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(tesla_bp, spawn_points[spawn_idx])
    camera = spawn_camera(world, vehicle)
    vehicle.set_autopilot(True)
    mp4_path = os.path.join(MP4_SAVE_DIR, video_name + MP4_EXT)
    avi_path = os.path.join(AVI_SAVE_DIR, video_name + AVI_EXT)
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc_avi = cv2.VideoWriter_fourcc(*'XVID')
    out_mp4 = cv2.VideoWriter(mp4_path, fourcc_mp4, FPS, (WIDTH, HEIGHT))
    out_avi = cv2.VideoWriter(avi_path, fourcc_avi, FPS, (WIDTH, HEIGHT))
    frame_queue = queue.Queue(maxsize=60)
    stop_flag = threading.Event()
    frame_count = [0]
    start_time = time.time()

    def _on_image(image):
        frame = preprocess_image(image)
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def writer_thread_func():
        while not stop_flag.is_set():
            try:
                frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            out_mp4.write(frame)
            out_avi.write(frame)
            frame_count[0] += 1

    camera.listen(_on_image)
    writer_thread = threading.Thread(target=writer_thread_func)
    writer_thread.start()

    print(f'正在录制 {MAP_NAME}，spawn点: {spawn_idx+1}，保存为: {mp4_path} 和 {avi_path}，按q可提前退出...')
    window_name = 'Camera View'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WIDTH, HEIGHT)
    while True:
        if not frame_queue.empty():
            frame = frame_queue.queue[-1]
            cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('手动退出')
            break
        time.sleep(0.001)

    # 录制主循环结束后
    try:
        camera.stop()
    except Exception:
        pass
    stop_flag.set()
    try:
        writer_thread.join(timeout=2)
    except Exception:
        pass
    try:
        out_mp4.release()
        out_avi.release()
    except Exception:
        pass
    try:
        vehicle.destroy()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    actual_duration = time.time() - start_time
    print(f'共录制帧数: {frame_count[0]}，地图: {MAP_NAME}，spawn点编号: {spawn_idx+1}')
    print(f'视频已保存: {mp4_path}')
    print(f'视频已保存: {avi_path}')
    print(f'实际录制时长: {actual_duration:.2f} 秒')
    if frame_count[0] == 0:
        print("警告：本次录制未采集到任何帧，请检查Carla仿真环境和摄像头是否正常！")

if __name__ == '__main__':
    record_one_video()