from unet import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import train_loader  # 导入 train_loader

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, num_classes=1).to(device)  # 输入3通道，输出1通道（二分类）
criterion = nn.BCEWithLogitsLoss()  # 二分类损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs["out"], masks)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

# 开始训练
train(model, train_loader, criterion, optimizer, num_epochs=20)

# 保存模型
torch.save(model.state_dict(), "/public/home/zengqiong007/hominid/static/assets/weights/unet_best_model.pth")
print("模型已保存到 unet_drive_eye.pth")
