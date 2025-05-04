# ==================== 数据集配置 ====================
dataset = 'Tusimple'  # 定义数据集名称，使用 Tusimple 数据集
data_root = r'G:/Graduation_Design/DateSet/archive/TUSimple' # 数据根目录，运行前需修改为实际路径

# ==================== 训练配置 ====================
epoch = 100  # 训练的总轮数
batch_size = 32  # 每个训练批次的样本数量

# ==================== 优化器配置 ====================
optimizer = 'SGD'  # 优化器类型，采用随机梯度下降（SGD）
learning_rate = 0.05  # 学习率，控制模型参数更新的步长
weight_decay = 0.0001  # 权重衰减系数，用于防止过拟合
momentum = 0.9  # 动量系数，加速 SGD 在相关方向上的收敛并抑制振荡

# ==================== 学习率调度配置 ====================
scheduler = 'multi'  # 学习率调度器类型，使用多步衰减调度器
steps = [50, 75]  # 学习率衰减的步数，在这些步数时学习率会发生衰减
gamma = 0.1  # 学习率衰减因子，每次衰减时学习率乘以该因子
warmup = 'linear'  # 热身策略类型，采用线性热身
warmup_iters = 100  # 热身迭代次数

# ==================== 模型配置 ====================
backbone = '18'  # 骨干网络的编号，使用 ResNet-18
griding_num = 100  # 网格数量，用于车道线检测
use_aux = False  # 是否使用辅助损失

# ==================== 损失函数配置 ====================
sim_loss_w = 0.0  # 相似性损失的权重
shp_loss_w = 0.0  # 形状损失的权重
var_loss_power = 2.0  # 方差损失的幂次
mean_loss_w = 0.05  # 均值损失的权重
cls_loss_col_w = 1.0  # 列方向分类损失的权重
cls_ext_col_w = 1.0  # 列方向分类扩展损失的权重
mean_loss_col_w = 0.05  # 列方向均值损失的权重

# ==================== 其他配置 ====================
note = ''  # 备注信息
log_path = ''  # 日志保存路径
finetune = None  # 微调模型的路径，若为 None 则不进行微调
resume = None  # 恢复训练的模型路径，若为 None 则不恢复
test_model = ''  # 测试模型的路径
test_work_dir = ''  # 测试工作目录
num_lanes = 4  # 车道线的数量
auto_backup = True  # 是否自动备份模型
num_row = 56  # 行的数量
num_col = 41  # 列的数量
train_width = 800  # 训练图像的宽度
train_height = 320  # 训练图像的高度
num_cell_row = 100  # 行方向的单元格数量
num_cell_col = 100  # 列方向的单元格数量
fc_norm = False  # 全连接层是否使用归一化
soft_loss = True  # 是否使用软损失
eval_mode = 'normal'  # 评估模式，使用正常模式
crop_ratio = 0.8  # 裁剪比例