import h5py
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from collections import deque
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from newdataset import CSGODataset, SmallCSGODataset
from temporal import CSGO_model
from config import Config
from torch.utils.data import Dataset

# mouse_x_possibles = [-140, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.00, 10.0, 20.0, 30.0, 60.0,
#                      100.0, 140]
# mouse_y_possibles = [-75, -60, -50.00, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 60.0, 75]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset = SmallCSGODataset(Config["file_path"], num_samples=282,frames_per_sample=Config["horizon"])
# dataset.print_sample(0)
# dataset.print_sample(1)
# dataset.print_sample(10)
# for i in range(len(dataset) - 1):
#     data1, _ = dataset[i]
#     data2, _ = dataset[i + 1]
#
#     # 检查除了最后一帧之外的其他帧是否相同
#     for j in range(data1.shape[0] - 1):
#         if not np.array_equal(data1[j].numpy(), data2[j + 1].numpy()):
#             print(f"Mismatch found at index {i}, frame {j}")
class TestDataset(Dataset):
    def __init__(self, num_samples, frames_per_sample):
        self.num_samples = num_samples
        self.frames_per_sample = frames_per_sample

        # 创建一个简单的测试数据
        self.data = np.array([i for i in range(num_samples)])

    def __len__(self):
        return self.num_samples * (1100 // self.frames_per_sample)

    def __getitem__(self, idx):
        data_idx = idx // (1100 // self.frames_per_sample)
        frame_idx = (idx % (1100 // self.frames_per_sample)) * self.frames_per_sample

        if frame_idx + self.frames_per_sample >= 1100:
            # 处理边界情况，防止索引超出数据范围
            frames = np.array([self.data[data_idx]] * (self.frames_per_sample - frame_idx))
            frames = np.concatenate((frames, np.array([self.data[data_idx+1]])))
        else:
            frames = np.array([self.data[data_idx]] * self.frames_per_sample)

        return frames

dataset = TestDataset(1000, 10)
for i in range(10):
    print(dataset[i])
