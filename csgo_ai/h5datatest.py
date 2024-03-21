import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from collections import deque
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from newdataset import CSGODataset, SmallCSGODataset
# from csgo_dataset import CSGODataset, SmallCSGODataset
from temporal import CSGO_model
from newconfig import Config

h5_file = 'data/h5_data/csgo_new_10.h5'

small_cs_go_dataset = SmallCSGODataset(Config["file_path"], num_samples=282,frames_per_sample=Config["horizon"])
# DataLoader
small_data_loader = DataLoader(small_cs_go_dataset, batch_size=Config["batch_size"], shuffle=True)
# print(small_data_loader)
data_iter = iter(small_data_loader)
data = next(data_iter)
print(len(data))

images, labels = data


print('Images:', images.shape)
print('Labels:', labels.shape)
