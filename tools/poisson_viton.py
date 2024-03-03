import os
import cv2
import numpy as np
from PIL import Image


# 前景——真值
path1 = "/home/sd/Harddisk/zjh/Dataset/test/image"
# 后景——生成
path2 = "/home/sd/Harddisk/zjh/CAT8/results50/P-GP2D"
# pair.txt
dataset_file = "/home/sd/Harddisk/zjh/Dataset/test_pairs.txt"
dataset_list = []
with open(dataset_file, 'r') as f:
    for line in f.readlines():
        name, _ = line.strip().split()
        dataset_list.append(name)

os.makedirs("/home/sd/Harddisk/zjh/CAT8/results50/P-GP2P")
for i in range(len(dataset_list)):
    name = dataset_list[i]
    # mask 路径
    mask_path = os.path.join("/home/sd/Harddisk/zjh/Dataset/test/mask", name[:-4]+".png")
    # 前景路径
    src_path = os.path.join(path1, name)
    # 后景路径
    dst_path = os.path.join(path2, str(i)+".png")
    # 加载前景
    src = cv2.imread(src_path)
    src = cv2.resize(src, (384, 512))
    # 加载后景
    dst = cv2.imread(dst_path)
    # 加载 mask
    mask = Image.open(mask_path).convert("L").resize((384, 512))
    mask = np.array(mask)
    mask = 255-mask
    # 融合
    output = cv2.seamlessClone(src, dst, mask, (192,256), cv2.NORMAL_CLONE)
    # 保存
    cv2.imwrite(os.path.join("/home/sd/Harddisk/zjh/CAT8/results50/P-GP2P", name.replace("jpg", "png")), output)
