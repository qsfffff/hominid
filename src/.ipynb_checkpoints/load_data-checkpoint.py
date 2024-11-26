import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

class DriveEyeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # 只保留有效的图像文件
        self.image_names = [
            fname for fname in os.listdir(image_dir) 
            if fname.endswith(".jpg") or fname.endswith(".png")
        ]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.image_names[idx].replace('.jpg', '_training_mask.png'))

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # 掩码为单通道灰度图
        except Exception as e:
            print(f"Error loading file: {img_path} or {mask_path}. Exception: {e}")
            raise e

        if self.transform:
            image = self.transform(image)
            # mask = self.transform(mask)
            mask = transforms.Resize((256, 256))(mask)  # 单独调整掩码大小

        # mask = (mask > 0).float()  # 将掩码值二值化（0 或 1）
        # 将掩码二值化，并转为整数类型
        #mask = torch.tensor((mask > 0).numpy(), dtype=torch.long)
        if isinstance(mask, Image.Image):  # 确保导入了 PIL.Image
            mask = np.array(mask)

        # 进行比较后转换为 PyTorch 张量
        mask = torch.tensor((mask > 0), dtype=torch.long)

        return image, mask

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整大小
    transforms.ToTensor(),         # 转为张量
transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 归一化
])

# 创建数据集和数据加载器
dataset = DriveEyeDataset("/public/home/zengqiong007/hominid/data/drive_eye/training/images", "/public/home/zengqiong007/hominid/data/drive_eye/training/mask", transform=transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
