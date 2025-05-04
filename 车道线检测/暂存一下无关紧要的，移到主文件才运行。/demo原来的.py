import torch, os, cv2  # 导入 PyTorch、操作系统和 OpenCV 库
from model.model import parsingNet  # 从 model 模块中导入 parsingNet 类
from utils.common import merge_config  # 从 utils.common 模块中导入 merge_config 函数
from utils.dist_utils import dist_print  # 从 utils.dist_utils 模块中导入 dist_print 函数
import torch
import scipy.special, tqdm  # 导入 scipy.special 用于计算 softmax，tqdm 用于显示进度条
import numpy as np  # 导入 NumPy 库用于数值计算
import torchvision.transforms as transforms  # 导入 torchvision 的 transforms 模块用于图像预处理
from data.dataset import LaneTestDataset  # 从 data.dataset 模块中导入 LaneTestDataset 类
from data.constant import culane_row_anchor, tusimple_row_anchor  # 从 data.constant 模块中导入车道线锚点信息

if __name__ == "__main__":
    # 设置 PyTorch 的 cuDNN 基准模式，以提高计算效率
    torch.backends.cudnn.benchmark = True

    # 合并配置信息
    args, cfg = merge_config()

    # 打印开始测试的提示信息
    dist_print('start testing...')

    # 确保配置的骨干网络在支持的列表中
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    # 根据数据集类型设置每条车道的分类数量
    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        # 如果数据集类型不支持，抛出 NotImplementedError 异常
        raise NotImplementedError

    # 初始化解析网络，不使用预训练模型，设置骨干网络和分类维度，不使用辅助分割
    net = parsingNet(pretrained=False, backbone=cfg.backbone, cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4),
                     use_aux=False).cuda()  # we dont need auxiliary segmentation in testing

    # 加载测试模型的状态字典
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    # 处理模型状态字典中的键名，去除 'module.' 前缀
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    # 加载处理后的状态字典到网络中
    net.load_state_dict(compatible_state_dict, strict=False)
    # 将网络设置为评估模式
    net.eval()

    # 定义图像预处理的转换操作
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),  # 调整图像大小为 288x800
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 归一化图像
    ])

    # 根据数据集类型设置测试文件列表和相关参数
    if cfg.dataset == 'CULane':
        # 修改 splits 列表，移除多余引号
        splits = ['G:\\Graduation_Design\\ReferenceCodes\\Lane_Detection-study-main\\my-video\\pic3\\ygc_test2.txt']
        # 创建数据集对象
        datasets = [LaneTestDataset(cfg.data_root, os.path.join(cfg.data_root, split),
                                    img_transform=img_transforms) for split in splits]
    
        img_w, img_h = 1640, 590  # 设置输出视频的宽度和高度
        row_anchor = culane_row_anchor  # 设置车道线锚点信息
    elif cfg.dataset == 'Tusimple':
        splits = ['test.txt']
        datasets = [LaneTestDataset(cfg.data_root, os.path.join(cfg.data_root, split), img_transform=img_transforms) for
                    split in splits]
        img_w, img_h = 1280, 720  # 设置输出视频的宽度和高度
        row_anchor = tusimple_row_anchor  # 设置车道线锚点信息
    else:
        # 如果数据集类型不支持，抛出 NotImplementedError 异常
        raise NotImplementedError

    # 遍历测试文件列表和对应的数据集
    for split, dataset in zip(splits, datasets):
        # 创建数据加载器
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        # 设置视频编码器的四字符代码
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # 打印输出视频的文件名
        print(split[:-3] + 'avi')
        # 创建视频写入对象
        vout = cv2.VideoWriter(split[:-3] + 'avi', fourcc, 30.0, (img_w, img_h))

        # 遍历数据加载器中的数据
        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data  # 解包数据
            imgs = imgs.cuda()  # 将图像数据移动到 GPU 上

            # 禁用梯度计算，以提高推理速度
            with torch.no_grad():
                out = net(imgs)  # 前向传播，得到网络输出

            # 生成网格采样点
            col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
            col_sample_w = col_sample[1] - col_sample[0]  # 计算采样点的宽度

            out_j = out[0].data.cpu().numpy()  # 将网络输出转换为 NumPy 数组并移动到 CPU 上
            out_j = out_j[:, ::-1, :]  # 反转数组的第二维
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)  # 计算 softmax 概率
            idx = np.arange(cfg.griding_num) + 1  # 生成索引数组
            idx = idx.reshape(-1, 1, 1)  # 调整索引数组的形状
            loc = np.sum(prob * idx, axis=0)  # 计算位置信息
            out_j = np.argmax(out_j, axis=0)  # 获取最大值的索引
            loc[out_j == cfg.griding_num] = 0  # 将无效位置的索引置为 0
            out_j = loc  # 更新输出结果

            # 读取图像文件
            vis = cv2.imread(os.path.join(cfg.data_root, names[0]))

            # 遍历输出结果的列
            for i in range(out_j.shape[1]):
                # 检查该列的有效位置数量是否大于 2
                if np.sum(out_j[:, i] != 0) > 2:
                    # 遍历输出结果的行
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            # 计算绘制圆圈的坐标
                            ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                                   int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                            # 在图像上绘制绿色圆圈
                            cv2.circle(vis, ppp, 5, (0, 255, 0), -1)

            # 将处理后的图像写入视频文件
            vout.write(vis)

        # 释放视频写入对象
        vout.release()
