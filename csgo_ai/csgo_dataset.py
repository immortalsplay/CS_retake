import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

def find_interval(value, intervals):
    """Find the right interval index for the given value.

    Args:
        value (float): The value for which to find the interval.
        intervals (list of float): A sorted list of interval bounds.

    Returns:
        int: The index of the interval.
    """
    # If the value is less than the smallest bound, return 0
    if value < intervals[0]:
        return 0

    # If the value is greater than the largest bound, return the last index
    if value > intervals[-1]:
        return len(intervals) - 1

    # Otherwise, find the correct interval
    for i in range(1, len(intervals)):
        if intervals[i-1] <= value < intervals[i]:
            return i
    # In case the value is exactly the largest bound
    return len(intervals) - 1
class CSGODataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(self.h5_file, 'r') as f:
            self.length = len(f['data'])

        self.new_size = (224, 224)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            data = f['data'][idx]
            label = f['label'][idx]

        data = data.reshape(1100, 125, 200, 3)  
        data = np.array([cv2.resize(frame, self.new_size) for frame in data])
        data = np.transpose(data, (0, 3, 1, 2))  

        return torch.from_numpy(data).float(), torch.tensor(label).long()


class SmallCSGODataset(Dataset):
    def __init__(self, h5_file, num_samples, frames_per_sample):
        self.h5_file = h5_file
        self.num_samples = num_samples
        self.frames_per_sample = frames_per_sample

        with h5py.File(self.h5_file, 'r') as f:
            self.data = f['data'][0:self.num_samples]
            self.label = f['label'][0:self.num_samples]

        self.new_size = (224, 224)

    def __len__(self):
        return min(self.num_samples * (1100 // self.frames_per_sample), len(self.data))

    def __getitem__(self, idx):
        data_idx = idx // (1100 // self.frames_per_sample)
        frame_idx = (idx % (1100 // self.frames_per_sample)) * self.frames_per_sample

        data = self.data[data_idx][frame_idx:frame_idx + self.frames_per_sample]
        label = self.label[data_idx][frame_idx:frame_idx + self.frames_per_sample]

        data = data.reshape(-1, 125, 200, 3) 
        data = np.array([cv2.resize(frame, self.new_size) for frame in data])
        data = np.transpose(data, (0, 3, 1, 2)) 
        # mouse_x_intervals = [-140, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.00, 10.0, 20.0, 30.0, 60.0, 100.0, 140]
        # mouse_y_intervals = [-75, -60,  -50.00, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 60.0, 75]
        mouse_x_intervals = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8,
                             9, 10]
        mouse_y_intervals = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8,
                             9, 10]
        for i in range(label.shape[0]):
            label[i, 0] = find_interval(label[i, 0], mouse_x_intervals)
            label[i, 1] = find_interval(label[i, 1], mouse_y_intervals)
        return torch.from_numpy(data).float(), torch.tensor(label).long()


class TestCSGODataset(Dataset):
    def __init__(self, h5_file, num_episodes):
        self.h5_file = h5_file
        self.num_episodes = num_episodes


        with h5py.File(self.h5_file, 'r') as f:
            self.data = f['data'][0:self.num_episodes]
            self.label = f['label'][0:self.num_episodes]

        self.new_size = (224, 224)

    def __len__(self):
        return min(self.num_episodes * 1100, len(self.data))

    def __getitem__(self, idx):
        data_idx = idx // 1100
        frame_idx = idx % 1100

        data = self.data[data_idx][frame_idx]
        label = self.label[data_idx][frame_idx]

        data = data.reshape(1, 125, 200, 3)  
        data = np.array([cv2.resize(frame, self.new_size) for frame in data])
        data = np.transpose(data, (0, 3, 1, 2)) 
        mouse_x_intervals = [-140, -100.0, -60.0, -30.0, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.00, 10.0, 20.0, 30.0,
                             60.0, 100.0, 140]
        mouse_y_intervals = [-75, -60, -50.00, -20.0, -10.0, -4.0, -2.0, -0.0, 2.0, 4.0, 10.0, 20.0, 50.0, 60.0, 75]

        label[0] = find_interval(label[0], mouse_x_intervals)
        label[1] = find_interval(label[1], mouse_y_intervals)

        return torch.from_numpy(data).float(), torch.tensor(label).long()
