import h5py
import torch
import torch.nn as nn
import cv2
import os
import torchvision.transforms as transforms
import numpy as np
from PIL import ImageGrab
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch
import pandas as pd
import time
import pyautogui as pg
import output_keys
import ctypes
import keyboard

import os
from collections import deque
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from newdataset import CSGODataset, SmallCSGODataset
from csgo_dataset import CSGODataset, SmallCSGODataset
from temporal import CSGO_model
from newconfig import Config

from collections import deque

# que for
horizon = Config["horizon"]
screen_captures = deque(maxlen=horizon)
mouse_x_possibles = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,5,6,7,8,9,10]
mouse_y_possibles = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,5,6,7,8,9,10]

SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)

w_char = 0x11
s_char = 0x1F
a_char = 0x1E
d_char = 0x20
q_char = 0x10
n_char = 0x31  # is bound on my machine to clear decals
r_char = 0x13
one_char = 0x02
two_char = 0x03
three_char = 0x04
four_char = 0x05
five_char = 0x06
seven_char = 0x08
ctrl_char = 0x1D
shift_char = 0x2A
space_char = 0x39
b_char = 0x30
i_char = 0x17
v_char = 0x2F
h_char = 0x23
o_char = 0x18
p_char = 0x19
e_char = 0x12
c_char_ = 0x2E
t_char = 0x14
u_char = 0x16
m_char = 0x32
g_char = 0x22
k_char = 0x25
x_char = 0x2D
c_char2 = 0x2E
y_char = 0x15
under_char = 0x0C  # actually minus, use in combo w shift for underscore
cons_char = 0x29
ret_char = 0x1C
esc_char = 0x01


def contains_epoch_number(filename):
    parts = filename.split('_')
    if len(parts) > 1 and parts[-2] == 'epoch' and parts[-1].split('.')[0].isdigit():
        return True
    return False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# init
model = CSGO_model(horizon = Config["horizon"],
                   num_feature=Config["num_feature"],
                   depth = Config["depth"],
                   num_heads = Config["num_heads"],
                   head_dim = Config["head_dim"],
                   inverse_dynamic_dim = Config["inverse_dynamic_dim"],
                   layer_norm_cfg=Config["layer_norm_cfg"],
                   model_option=Config["model_option"],
                   frame_count=Config["frame_count"])

model.to(device)

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

start_epoch = 0
save_freq = 5   # 
best_loss = float('inf')


def img_preprocess(img):
    # yifeng:cv2.readBGR to model, PIL reading RGB, so need RGB2BGR
    img = img[:, :, ::-1] # RGB2BGR

    # resize to 1080p
    img = cv2.resize(img, (1920,1080))
    # center crop
    height, width, _ = img.shape
    crop_size = (500, 800) # yifeng:original crop_size = (500, 500)
    center_x = width // 2
    center_y = height // 2
    crop_x = center_x - crop_size[1] // 2
    crop_y = center_y - crop_size[0] // 2
    crop_x2 = crop_x + crop_size[1]
    crop_y2 = crop_y + crop_size[0]
    img = img[crop_y:crop_y2,crop_x:crop_x2,:]
    img = cv2.resize(img,(200,125))  
    img = cv2.resize(img,(224,224))
    return img

checkpoint_path = r'D:\python_code\csgo_behavior_cloning_ai\csgo_ai\new_checkpoints\layer_norm_cfg_1\model_option_3\frame_count_10\checkpoint_epoch_911.pth'
ckpt = torch.load(checkpoint_path)
model.load_state_dict(ckpt, False)
model.eval()

try:
    while True:
    # Get screen capture
        screen_capture = ImageGrab.grab()  

        # Convert the PIL Image object to a numpy array
        screen_capture = np.array(screen_capture)  

        # Preprocess the screenshot
        screen_capture = img_preprocess(screen_capture)

        # Add to queue
        screen_captures.append(screen_capture)

        # When we have enough frames, use them as input for the model
        if len(screen_captures) >= horizon:
            # Create a new tensor containing all the screenshots
            input_tensor = torch.stack([transforms.ToTensor()(img) for img in screen_captures], dim=1)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            input_tensor = input_tensor.to(device)  # Move tensor to GPU
            input_tensor = input_tensor.transpose(1, 2)  # Transform tensor dimensions to (B, C, H, W, T)
            # print('input_tensor', input_tensor.size(), type(input_tensor))
            # Use the model for prediction
            outputs = model(input_tensor)
            # print('outputs', outputs)

            # Get mouse X and Y axis outputs
            mouse_x_output = outputs[0].cpu().detach().numpy()
            mouse_y_output = outputs[1].cpu().detach().numpy()

            # Use argmax to get the index of the action with the highest probability
            mouse_x_index = np.argmax(mouse_x_output)
            mouse_y_index = np.argmax(mouse_y_output)
            sensitivity = 2.5
            m_pitch = 0.022
            m_yaw = 0.022
            # Get the specific mouse movement values based on the index
            mouse_x = mouse_x_possibles[mouse_x_index]
            mouse_y = mouse_y_possibles[mouse_y_index]
            pixel_y = mouse_y / (sensitivity * m_pitch)  # sensitivity = 2.5; m_pitch = 0.022
            pixel_x = mouse_x / (sensitivity * m_yaw)  # sensitivity = 2.5; m_yaw= 0.022
            # Get key output and convert to numpy array
            keys_output = outputs[2].cpu().detach().numpy()

            # Convert key output to 0-1 format
            keys_output = (keys_output > 0.5).astype(int)  # Assuming 0.5 is your threshold

            print('Mouse movement: mouse ({}, {}); pixel ({},{})'.format(mouse_x, mouse_y, pixel_x, pixel_y))
            #print('Keys output: {}'.format(keys_output))


            start_time = time.time()
            if mouse_x != 0 or mouse_y != 0:
                output_keys.set_pos(-int(pixel_x), int(pixel_y))  # based on 200 pixels
            if keys_output[0][0] != 0:
                output_keys.hold_left_click()
            if keys_output[0][0] == 0:
                output_keys.release_left_click()
            if keys_output[0][1] != 0:
                output_keys.hold_right_click()
            if keys_output[0][1] == 0:
                output_keys.release_right_click()
            if keys_output[0][3] != 0:
                output_keys.HoldKey(w_char)
            if keys_output[0][3] == 0:
                output_keys.HoldKey(w_char)
            if keys_output[0][4] != 0:
                output_keys.HoldKey(a_char)
            if keys_output[0][4] == 0:
                output_keys.HoldKey(a_char)
            if keys_output[0][5] != 0:
                output_keys.HoldKey(s_char)
            if keys_output[0][5] == 0:
                output_keys.ReleaseKey(s_char)
            if keys_output[0][6] != 0:
                output_keys.HoldKey(d_char)
            if keys_output[0][6] == 0:
                output_keys.ReleaseKey(d_char)
            if keys_output[0][7] != 0:
                output_keys.HoldKey(r_char)
            if keys_output[0][7] == 0:
                output_keys.ReleaseKey(r_char)
            if keys_output[0][8] != 0:
                output_keys.HoldKey(q_char)
            if keys_output[0][8] == 0:
                output_keys.ReleaseKey(q_char)
            if keys_output[0][21] != 0:
                output_keys.HoldKey(shift_char)
            if keys_output[0][21] == 0:
                output_keys.ReleaseKey(shift_char)
            if keys_output[0][22] != 0:
                output_keys.HoldKey(space_char)
            if keys_output[0][22] == 0:
                output_keys.ReleaseKey(space_char)
            if keys_output[0][23] != 0:
                output_keys.HoldKey(ctrl_char)
            if keys_output[0][23] == 0:
                output_keys.ReleaseKey(ctrl_char)
            # time.sleep(0.1)
            end_time = time.time()
            print(end_time - start_time)
except KeyboardInterrupt:
    print('Exit requested.')





