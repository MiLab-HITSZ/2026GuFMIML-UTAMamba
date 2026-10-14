
import os
import torch

# Dataset Config
DATASET = 'voc2007'  # Options: 'voc2007', 'voc2012', 'scene', 'msrc'
DATA_ROOT = '/home/gjh/datasets'

SCENE_IMAGE_DIR = os.path.join(DATA_ROOT, 'scene', 'original')
SCENE_LABEL_PATH = os.path.join(DATA_ROOT, 'scene', 'miml data.mat')
MSRC_ROOT = os.path.join(DATA_ROOT, 'MSRCv2', 'msrc2_seg')
COCO_2014_ROOT = os.path.join(DATA_ROOT, 'coco2014')
# Model Config
MODEL_NAME = 'bimamba'
NUM_CLASSES = 20
D_MODEL = 256
GLOVE_PATH = '/home/gjh/My_code7/data/glove.6B.300d.txt'
FREEZE_BACKBONE = 'layer4'  # Options: False (train all), 'all' (freeze all), 'layer4' (train only layer4)
USE_GCN = True
USE_DECODER = True
USE_OT = True
OT_TYPE = 'soft_uot'  # 'sigmoid' (旧) | 'soft_uot' (闭式不平衡软 OT, 推荐)

# OT auxiliary loss weights (only effective when USE_OT=True and OT_TYPE='soft_uot')
LAMBDA_NEG = 0.05    # Penalises peak transport mass on absent labels.
LAMBDA_MASS = 0.05   # Direct BCE supervision on the OT mass head against ground-truth targets. Set to 0 to disable.
LAMBDA_CONSISTENCY = 1e-3  # KL between forward (T_f) and backward (T_b) plans on present labels. Small weight is enough.

# Training Config
SEED = 82
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
LR = 3e-5
WEIGHT_DECAY = 1e-5
EPOCHS = 5  # Local epochs
ROUNDS = 50
NUM_CLIENTS = 5
PARTITION_MODE = 'iid'
DIRICHLET_ALPHA = 0.5
LABELS_PER_CLIENT = 6

# Paths
SAVE_DIR = 'checkpoints'
LOG_DIR = 'logs'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
