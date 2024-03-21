
import os

# set hyper-parameters
layer_norm_cfg = True
model_option = 'GRU'
frame_count = 10

# set log
checkpoint_path = os.path.join('checkpoints',
                           f'layer_norm_cfg_{layer_norm_cfg}',
                           f'model_option_{model_option}',
                           f'frame_count_{frame_count}')
os.makedirs(checkpoint_path, exist_ok=True)

# print 
print(checkpoint_path)

# make dirs
with open(os.path.join(checkpoint_path, 'config.yaml'), 'w') as f:
    f.write('model: CSGO_model')
with open(os.path.join(checkpoint_path, 'best_model.pth'), 'w') as f:
    f.write('model_state_dict')
with open(os.path.join(checkpoint_path, 'checkpoint_5.pth'), 'w') as f:
    f.write('epoch: 5')