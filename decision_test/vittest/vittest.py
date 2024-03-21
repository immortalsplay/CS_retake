import torch
import torch.nn as nn
from vit_pytorch import ViT
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import os
import glob
import torch.nn.functional as F
# 定义模型
class vision_processor_net_work(nn.Module):
    def __init__(self):
        super(vision_processor_net_work, self).__init__()
        self.vit = ViT(
            image_size=256,
            patch_size=32,
            num_classes=1000,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.model = torch.nn.Sequential(*list(self.vit.children())[:-1])  # 提取特征,移除分类层

    def forward(self, x):
        x = self.model(x)
        return x


# 加载预训练权重
# model.load_state_dict(torch.load("path/to/pretrained_weights.pth"))
def train_vit_features(model, img_list):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    for i, image in tqdm(enumerate(img_list)):
        image_tensor = transform(image).unsqueeze(0)
        features = model(image_tensor)

# 加载并处理图像
# image_path = "./r0.jpg"  # 图像路径

image_dir= "pretrain_data"
image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
image_list = []
for path in image_paths:
    img = Image.open(path)
    print(img.size)
    image_list.append(img)
model = vision_processor_net_work()
train_vit_features(model, image_list)

"""
state_encoder
由于已经使用了 ViT 进行了特征提取，所以这里不需要再使用卷积层进行特征提取了，直接使用全连接层对特征进行编码即可
"""

"""
state states变量的形状为(batch:训练批次的样本大小, block_size:连续的游戏帧数, 每一帧的特征维度)
"""
"""
dt中的naive方法可以适用于模仿学习,用于训练模型
"""
# n_embd = 1024
# state_encoder = nn.Sequential(nn.Flatten(), nn.Linear(768*1024, n_embd), nn.Tanh())
# state = torch.zeros((1,768,1024))
# encoded_state = state_encoder(state)
# print(encoded_state.shape)



# # 使用 state_encoder 对特征进行编码
# encoded_state = self.state_encoder(state)
#
# # 将编码后的特征作为输入，传递到 transformer 的 block 中进行处理
# outputs = self.blocks(encoded_state)


