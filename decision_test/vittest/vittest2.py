import torch
import torch.nn as nn
from vit_pytorch import ViT
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import os
import glob
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义模型
from PIL import Image
import os

def get_image_size(img_path):
    with Image.open(img_path) as img:
        return img.size

def get_all_image_sizes(folder_path):
    sizes = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                sizes.append(get_image_size(img_path))
    return sizes
class vision_processor_net_work(nn.Module):
    def __init__(self):
        super(vision_processor_net_work, self).__init__()
        self.vit = ViT(
            image_size=(1024,768),
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
        print(x.shape)
        return x


# 加载预训练权重
# model.load_state_dict(torch.load("path/to/pretrained_weights.pth"))
def train_vit_features(model, img_list):
    transform = transforms.Compose([
        transforms.Resize((1024, 768)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    for i, path in tqdm(enumerate(img_list)):
        with Image.open(path) as img:
            image_tensor = transform(img).unsqueeze(0).to(device)
            features = model(image_tensor)
        del img

# 加载并处理图像
# image_path = "./r0.jpg"  # 图像路径

image_dir= "pretrain_data"
image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
model = vision_processor_net_work()
model.to(device)
train_vit_features(model, image_paths)
torch.save(model.state_dict(), "vit_features.pth")
print(torch.cuda.current_device())
# tmp = get_image_size('./pretrain_data/1679671600-993255400.jpg')
# print(tmp)