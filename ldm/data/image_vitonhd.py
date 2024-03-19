import os
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms.functional as F
from PIL import Image

class OpenImageDataset(data.Dataset):
    def __init__(self, state, dataset_dir, type="paired"):
        self.state=state
        self.dataset_dir = dataset_dir
        self.dataset_list = []

        if state == "train":
            self.dataset_file = os.path.join(dataset_dir, "train_pairs.txt")
            with open(self.dataset_file, 'r') as f:
                for line in f.readlines():
                    person, garment = line.strip().split()
                    self.dataset_list.append([person, person])

        if state == "test":
            self.dataset_file = os.path.join(dataset_dir, "test_pairs.txt")
            if type == "unpaired":
                with open(self.dataset_file, 'r') as f:
                    for line in f.readlines():
                        person, garment = line.strip().split()
                        self.dataset_list.append([person, garment])

            if type == "paired":
                with open(self.dataset_file, 'r') as f:
                    for line in f.readlines():
                        person, garment = line.strip().split()
                        self.dataset_list.append([person, person])

    def __len__(self):
        return len(self.dataset_list)
    
    def __getitem__(self, index):

        person, garment = self.dataset_list[index]

        # 确定路径
        img_path = os.path.join(self.dataset_dir, self.state, "image", person)
        reference_path = os.path.join(self.dataset_dir, self.state, "cloth", garment)
        mask_path = os.path.join(self.dataset_dir, self.state, "mask", person[:-4]+".png")                              
        densepose_path = os.path.join(self.dataset_dir, self.state, "image-densepose", person)
        
        # 加载图像
        img = Image.open(img_path).convert("RGB").resize((512, 512))
        img = torchvision.transforms.ToTensor()(img)
        refernce = Image.open(reference_path).convert("RGB").resize((224, 224))
        refernce = torchvision.transforms.ToTensor()(refernce)
        mask = Image.open(mask_path).convert("L").resize((512, 512))
        mask = torchvision.transforms.ToTensor()(mask)
        mask = 1-mask
        densepose = Image.open(densepose_path).convert("RGB").resize((512, 512))
        densepose = torchvision.transforms.ToTensor()(densepose)

        # 正则化
        img = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        refernce = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                    (0.26862954, 0.26130258, 0.27577711))(refernce)
        densepose = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(densepose)

        # 生成 inpaint 和 hint
        inpaint = img * mask
        hint = torchvision.transforms.Resize((512, 512))(refernce)
        hint = torch.cat((hint,densepose),dim = 0)

        return {"GT": img,                  # [3, 512, 512]
                "inpaint_image": inpaint,   # [3, 512, 512]
                "inpaint_mask": mask,       # [1, 512, 512]
                "ref_imgs": refernce,       # [3, 224, 224]
                "hint": hint,               # [6, 512, 512]
                }

