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

mouse_x_intervals = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8,
                     9, 10]
mouse_y_intervals = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8,
                     9, 10]


def contains_epoch_number(filename):
    parts = filename.split('_')
    if len(parts) > 1 and parts[-2] == 'epoch' and parts[-1].split('.')[0].isdigit():
        return True
    return False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# make dataset
cs_go_dataset = CSGODataset(Config["file_path"])
small_cs_go_dataset = SmallCSGODataset(Config["file_path"], num_samples=282,frames_per_sample=Config["horizon"])
# make DataLoader
small_data_loader = DataLoader(small_cs_go_dataset, batch_size=Config["batch_size"], shuffle=True)
# init model
model = CSGO_model(horizon = Config["horizon"],
                   num_feature=Config["num_feature"],
                   depth = Config["depth"],
                   num_heads = Config["num_heads"],
                   head_dim = Config["head_dim"],
                   inverse_dynamic_dim = Config["inverse_dynamic_dim"],
                   layer_norm_cfg=Config["layer_norm_cfg"],
                   model_option=Config["model_option"],
                   frame_count=Config["frame_count"])
# move to GPU
model.to(device)
# set optimizer
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
# set epoch
start_epoch = 0
save_freq = 5   # step 5
best_loss = float('inf')



checkpoint_path = os.path.join('new_checkpoints',
                           f'layer_norm_cfg_{Config["layer_norm_cfg"]}',
                           f'model_option_{Config["model_option"]}',
                           f'frame_count_{Config["frame_count"]}')
os.makedirs(checkpoint_path, exist_ok=True)
checkpoint_files = deque(maxlen=3)

latest_checkpoint = None  # Initialize latest_checkpoint to None

# Check for "best_model.pth" first

checkpoint_path = os.path.join('new_checkpoints',
                           f'layer_norm_cfg_{Config["layer_norm_cfg"]}',
                           f'model_option_{Config["model_option"]}',
                           f'frame_count_{Config["frame_count"]}')
os.makedirs(checkpoint_path, exist_ok=True)


# Check for "best_model.pth" first
best_model_path = os.path.join(checkpoint_path, 'best_model.pth')
if os.path.exists(best_model_path):
    latest_checkpoint = 'best_model.pth'
else:
    # If "best_model.pth" does not exist, find the latest checkpoint
    if os.listdir(checkpoint_path):
        filtered_files = list(filter(contains_epoch_number, os.listdir(checkpoint_path)))
        if filtered_files:
            latest_checkpoint = max(filtered_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

if latest_checkpoint is not None:  # Only load the checkpoint if it exists
    print(f"Loading checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(os.path.join(checkpoint_path, latest_checkpoint))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
else:
    print("No valid checkpoint files found, starting from scratch.")




def combined_loss(batch_size,outputs, targets):
    mouse_x_outputs ,mouse_y_outputs,other_outputs = outputs


    mouse_x_loss=0
    mouse_y_loss=0
    other_loss=0
    batch_size = outputs[0].shape[0]


    for b in range(batch_size):
        mouse_x_loss += nn.NLLLoss()(mouse_x_outputs[b], targets[b, 0])
        mouse_y_loss += nn.NLLLoss()(mouse_y_outputs[b], targets[b, 1])
        other_loss += nn.BCELoss()(other_outputs[b, :], targets[b,  2:].float().unsqueeze(1))

    return mouse_x_loss + mouse_y_loss + other_loss


# TensorBoard
writer = SummaryWriter()
# Training
model.train()
n_epochs=Config["num_epochs"]
losses = []
for epoch in range(start_epoch, n_epochs):
    epoch_loss = 0
    pbar = tqdm(enumerate(small_data_loader), total=len(small_data_loader), desc=f"Epoch {epoch + 1}/{n_epochs}")
    for batch_idx, (data, label) in pbar:
        data = data.to(device)
        # print("data.shape",data.shape)
        # print("data",data)
        label = label.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        # Loss
        loss = combined_loss(Config["batch_size"],outputs, label) # 用 combined_loss 替代之前的损失计算方式
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    avg_loss = epoch_loss / len(small_data_loader)
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
    losses.append(avg_loss)
    scheduler.step(avg_loss)
    # TensorBoard 
    writer.add_scalar('Loss/train', avg_loss, epoch)
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_epoch = epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
         }, os.path.join(checkpoint_path, 'best_model.pth'))
    # save freq
    if epoch % save_freq == 0:
        checkpoint_file = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_file)

        # add model
        checkpoint_files.append(checkpoint_file)

        # if full, delete the old
        if len(checkpoint_files) == 3:
            os.remove(checkpoint_files.popleft())

    # close TensorBoard
    writer.close()

    # save
    torch.save(model.state_dict(), "csgo_model.pth")