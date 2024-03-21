import os
import glob
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from PIL import Image

from temporal import CSGO_model  # 假设你已经在csgo_model.py文件中定义了CSGO_model类
from create_dataset import create_dataset  # 假设你已经在create_dataset.py文件中定义了create_dataset函数

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置数据目录前缀和游戏名称
csv_files = ["csv_file_1.csv", "csv_file_2.csv"]  # 替换为你的CSV文件列表
epochs = 100
batch_size = 32
learning_rate = 0.001




# 获取数据并分割为训练集和验证集
obss, actions, _, _, _, _ = create_dataset(csv_files)

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((1024, 768)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

obss = [transform(Image.fromarray(np.uint8(obs * 255))) for obs in obss]
obss = torch.stack(obss)

X_train, X_val, y_train, y_val = train_test_split(obss, actions, test_size=0.2, random_state=42)

# 创建DataLoader
train_dataset = TensorDataset(X_train, torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val, torch.tensor(y_val))
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 初始化模型并将其移动到GPU上
input_shape = (1024, 768, 3)  # 根据实际图像尺寸修改
model = CSGO_model(input_shape)
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # 验证模型
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{epochs}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

