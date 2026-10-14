
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VOCDetection
from PIL import Image
import scipy.io
import numpy as np
import json
from pycocotools.coco import COCO
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def get_transforms(size=224, split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

class VOCDataset(Dataset):
    def __init__(self, root, year='2007', image_set='train', transform=None, download=False):
        self.voc = VOCDetection(root, year=year, image_set=image_set, download=download)
        self.transform = transform
        self.class_names = VOC_CLASSES
        self.label_map = {cls: i for i, cls in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, target = self.voc[idx]
        objs = target['annotation'].get('object', [])
        if isinstance(objs, dict):
            objs = [objs]
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        for obj in objs:
            name = obj.get('name')
            if name in self.label_map:
                labels[self.label_map[name]] = 1.0
        
        if self.transform:
            img = self.transform(img)
        return img, labels

class SceneDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Check if files exist
        if not os.path.exists(label_path):
            print(f"⚠️ Label file not found: {label_path}. Using Mock Data.")
            self.labels = np.random.randint(0, 2, (100, 5)).astype(np.float32)
            self.image_names = [f"mock_{i}.jpg" for i in range(100)]
            self.num_classes = 5
            self.class_names = [f"scene_class_{i}" for i in range(self.num_classes)]
            self.is_mock = True
        else:
            mat = scipy.io.loadmat(label_path)
            self.labels = ((mat['targets'].T + 1) / 2).astype(np.float32)
            self.image_names = [f"{i + 1}.jpg" for i in range(len(self.labels))]
            self.num_classes = self.labels.shape[1]
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]
            self.is_mock = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if getattr(self, 'is_mock', False):
            image = Image.new('RGB', (224, 224), color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        else:
            img_path = os.path.join(self.image_dir, self.image_names[idx])
            image = Image.open(img_path).convert("RGB")
            
        label = torch.from_numpy(self.labels[idx]).float()
        if self.transform:
            image = self.transform(image)
        return image, label

class MSRCDataset(Dataset):
    def __init__(self, root, label_file, num_classes, transform=None):
        # 初始化MSRC数据集
        self.image_dir = os.path.join(root, "images")
        self.samples = []
        self.num_classes = num_classes
        self.transform = transform
        
        # MSRC v2 Color Map (23 classes found in GT)
        self.msrc_colors = [
            (0, 0, 128),
            (0, 64, 0),
            (0, 128, 0),
            (0, 128, 128),
            (0, 192, 0),
            (0, 192, 128),
            (64, 0, 0),
            (64, 0, 128),
            (64, 64, 0),
            (64, 128, 0),
            (64, 128, 128),
            (128, 0, 0),
            (128, 0, 128),
            (128, 64, 0),
            (128, 64, 128),
            (128, 128, 0),
            (128, 128, 128),
            (128, 192, 128),
            (192, 0, 0),
            (192, 0, 128),
            (192, 64, 0),
            (192, 128, 0),
            (192, 128, 128)
        ]
        self.color_to_idx = {c: i for i, c in enumerate(self.msrc_colors)}
        
        label_path = os.path.join(root, label_file)
        if not os.path.exists(label_path):
             print(f"⚠️ Label file not found: {label_path}")
             return

        print(f"Loading MSRC dataset from {label_path}...")
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 1: 
                    parts = line.strip().split()
                if len(parts) < 1: continue
                
                filename = parts[0]
                label_vec = torch.zeros(num_classes, dtype=torch.float32)
                
                # Try to load GT
                gt_filename = filename.replace('.bmp', '_GT.bmp')
                gt_path = os.path.join(root, 'gt', gt_filename)
                
                if os.path.exists(gt_path):
                    try:
                        gt_img = Image.open(gt_path).convert('RGB')
                        gt_arr = np.array(gt_img)
                        gt_pixels = gt_arr.reshape(-1, 3)
                        unique_colors = np.unique(gt_pixels, axis=0)
                        for color in unique_colors:
                            c = tuple(color)
                            if c in self.color_to_idx:
                                idx = self.color_to_idx[c]
                                if idx < num_classes:
                                    label_vec[idx] = 1.0
                    except Exception as e:
                        print(f"Error loading GT {gt_filename}: {e}")
                elif len(parts) >= 2:
                    try:
                        labels = list(map(int, [p for p in parts[1:] if p.strip()]))
                        for idx in labels:
                            if 0 <= idx < num_classes:
                                label_vec[idx] = 1.0
                            elif 1 <= idx <= num_classes:
                                label_vec[idx-1] = 1.0
                    except ValueError:
                        pass
                
                self.samples.append((filename, label_vec))

        self.class_names = [f"class_{i}" for i in range(self.num_classes)]
        print(f"Loaded {len(self.samples)} samples.")


    def __len__(self):
        # 获取数据集长度
        return len(self.samples)

    def __getitem__(self, idx):
        # 获取单个样本
        name, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, name)
        # If image not found, try appending .bmp or .jpg just in case
        if not os.path.exists(img_path):
             if os.path.exists(img_path + ".bmp"):
                 img_path += ".bmp"
             elif os.path.exists(img_path + ".jpg"):
                 img_path += ".jpg"
        
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

class COCOMultiLabelDataset(Dataset):
    def __init__(self, root_dir, ann_file, transform=None):
        """
        Args:
            root_dir (string): 图片所在目录 (例如 train2017)
            ann_file (string): 标注文件路径 (例如 instances_train2017.json)
            transform (callable, optional): 图像变换
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 加载 COCO 标注
        print(f"Loading COCO annotations from {ann_file}...")
        self.coco = COCO(ann_file)
        
        # 🌟 核心防坑：COCO 有 90 个 ID，但只有 80 个类别
        # 我们必须建立从 category_id 到 0~79 连续索引的映射
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat2idx = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.idx2cat = {i: cat_id for i, cat_id in enumerate(self.cat_ids)}
        self.num_classes = len(self.cat_ids) # 绝对是 80
        
        # 获取所有包含标注的图片 ID
        self.img_ids = sorted(self.coco.getImgIds())
        print(f"Loaded {len(self.img_ids)} images with {self.num_classes} classes.")

        # 预先提取所有 80 个类别的名称（供你的 load_glove_embeddings 使用）
        cats = self.coco.loadCats(self.cat_ids)
        self.class_names = [cat['name'] for cat in cats]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        # 1. 加载图片
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # 2. 生成多标签 Multi-hot Vector (长度为 80)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for ann in anns:
            # 过滤掉极其微小的无关噪点标注 (面积 < 10 的多半是标注错误)
            if ann['area'] > 10.0: 
                cat_id = ann['category_id']
                class_idx = self.cat2idx[cat_id]
                target[class_idx] = 1.0  # 将对应类别设为 1
                
        return image, target