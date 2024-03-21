import torch
from torch import nn
from torch.utils.data import DataLoader
from csgo_dataset import CSGODataset, SmallCSGODataset
from temporal import CSGO_model
from config import Config
def combined_loss(batch_size,outputs, targets):
    mouse_x_outputs ,mouse_y_outputs,other_outputs = outputs



    mouse_x_loss=0
    mouse_y_loss=0
    other_loss=0


    for b in range(batch_size):
        mouse_x_loss += nn.NLLLoss()(mouse_x_outputs[b], targets[b,0, 0])
        mouse_y_loss += nn.NLLLoss()(mouse_y_outputs[b], targets[b,0, 1])
        other_loss += nn.BCELoss()(other_outputs[b, :], targets[b, 0, 2:].float().unsqueeze(1))


    return mouse_x_loss + mouse_y_loss + other_loss
# loading
model = CSGO_model(horizon = Config["horizon"],
                   num_feature=Config["num_feature"],
                   depth = Config["depth"],
                   num_heads = Config["num_heads"],
                   head_dim = Config["head_dim"],
                   inverse_dynamic_dim = Config["inverse_dynamic_dim"],
                   layer_norm_cfg=Config["layer_norm_cfg"],
                   model_option=Config["model_option"],
                   frame_count=Config["frame_count"])

model.load_state_dict(torch.load('csgo_model.pth'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# init
test_dataset = CSGODataset(Config["file_path"], train=False) 
test_loader = DataLoader(test_dataset, batch_size=Config["batch_size"], shuffle=True)

model.eval()  # eval mode

# test
with torch.no_grad():  
    total_loss = 0
    for data, label in test_loader:
        data = data.to(device)
        label = label.to(device)
        outputs = model(data)
        loss = combined_loss(Config["batch_size"],outputs, label)
        total_loss += loss.item()

average_test_loss = total_loss / len(test_loader)
print("Test Loss: ", average_test_loss)