from src.fcn_model import FCN
import torch
import torch.nn as nn
import torch.optim as optim
from src.load_data import train_loader
from src.fcn_model import fcn_resnet50

# 定义模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = FCN(backbone="resnet50", num_classes=3).to(device)  # FCN 支持 ResNet 主干网络
model = fcn_resnet50(aux=True, num_classes=3, pretrain_backbone=True).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train_fcn(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            # 将 masks 从 [batch_size, 1, height, width] 转为 [batch_size, height, width]
            masks = masks.squeeze(1)
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs["out"], masks.long())  # 输出为 logits

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

# 开始训练
train_fcn(model, train_loader, criterion, optimizer, num_epochs=20)

# 保存模型
torch.save(model.state_dict(), "/public/home/zengqiong007/hominid/static/assets/weights/fcn_best_model.pth")
print("FCN 模型已保存到 fcn_best_model.pth")
