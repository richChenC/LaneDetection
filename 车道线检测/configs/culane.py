# DATA
dataset = 'CULane'
# data_root = './CULane'
data_root = r'G:/Graduation_Design/DateSet/CULane'
# TRAIN
epoch = 20
batch_size = 18
optimizer = 'SGD'  # ['SGD','Adam']
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi'  # ['multi', 'cos']
steps = [25, 38]
gamma = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = True
griding_num = 200
backbone = '18'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = './log'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = None
test_work_dir = None

num_lanes = 4
