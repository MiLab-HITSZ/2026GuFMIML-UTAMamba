import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# ================= 你的配置 =================
SOURCE_MASK_DIR = "/home/gjh/datasets/MSRCv2/msrc2_seg/gt"  # 你的原始图片路径
TARGET_MASK_DIR = "/home/gjh/datasets/MSRCv2" # 输出路径

# 将你的 map 转为标准的 tuple 字典，方便处理
COLOR_MAP = {
    (0, 0, 0): 255,
    (0, 0, 128): 0,
    (0, 64, 0): 1,
    (0, 128, 0): 2,
    (0, 128, 128): 3,
    (0, 192, 0): 4,
    (0, 192, 128): 5,
    (64, 0, 0): 6,
    (64, 0, 128): 7,
    (64, 64, 0): 8,
    (64, 128, 0): 9,
    (64, 128, 128): 10,
    (128, 0, 0): 11,
    (128, 0, 128): 12,
    (128, 64, 0): 13,
    (128, 64, 128): 14,
    (128, 128, 0): 15,
    (128, 128, 128): 16,
    (128, 192, 128): 17,
    (192, 0, 0): 18,
    (192, 0, 128): 19,
    (192, 64, 0): 20,
    (192, 128, 0): 21,
    (192, 128, 128): 22
}
# ===========================================

def convert_to_index_mask():
    if not os.path.exists(TARGET_MASK_DIR):
        os.makedirs(TARGET_MASK_DIR)

    print(f"🚀 开始转换 (共 {len(COLOR_MAP)-1} 个类别 + 背景)...")
    
    files = [f for f in os.listdir(SOURCE_MASK_DIR) if f.endswith(('.bmp', '.png', '.jpg'))]
    
    for filename in tqdm(files):
        img_path = os.path.join(SOURCE_MASK_DIR, filename)
        
        # 1. 读取并转为 RGB 数组
        rgb = np.array(Image.open(img_path).convert('RGB'))
        h, w = rgb.shape[:2]
        
        # 2. 初始化结果 Mask (默认填充 255)
        # 如果你的图片里有上面字典里没定义的杂色，它们就会变成 255
        mask = np.full((h, w), 255, dtype=np.uint8)
        
        # 3. 匹配颜色
        for color, class_id in COLOR_MAP.items():
            # color 是 (r, g, b) 元组
            # rgb 是 (H, W, 3) 数组
            # 我们利用 numpy 的广播机制比较
            matches = np.all(rgb == color, axis=2)
            mask[matches] = class_id
            
        # 4. 保存
        save_name = os.path.splitext(filename)[0] + '.png'
        Image.fromarray(mask).save(os.path.join(TARGET_MASK_DIR, save_name))

    print("✅ 转换完成！")
    print(f"有效类别 ID: 0 ~ 22")
    print(f"背景/忽略 ID: 255")

if __name__ == "__main__":
    convert_to_index_mask()