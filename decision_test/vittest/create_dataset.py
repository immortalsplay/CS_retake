import os
import numpy as np
from PIL import Image
import csv

def image_loader(image_paths):
    for image_path in image_paths:
        image = Image.open( image_path)
        image_np = np.array(image)
        yield image_np



def create_dataset(csv_files):
    actions = []
    image_paths = []
    time_steps = []
    done_idxs = []

    step_counter = 0
    game_ended = False

    # 读取CSV文件中的数据
    for csv_file in csv_files:
        with open(csv_file, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)  # 跳过标题行
            for row in csvreader:
                actions_row = [float(action) if idx in [0, 1] else int(action) for idx, action in enumerate(row[:-1])]

                # 检查游戏是否提前结束
                if all(action == 0 for action in actions_row) and row[-1] == '0':
                    game_ended = True
                    done_idxs.append(len(actions) - 1)
                else:
                    actions += actions_row
                    image_paths.append(row[-1])  # 读取图像地址并将其添加到image_paths列表中
                    step_counter += 1

                # 如果游戏结束，重置step_counter
                if game_ended:
                    time_steps += [step_counter] * (len(actions) - len(time_steps))
                    step_counter = 0
                    game_ended = False

    # 如果最后一局游戏没有提前结束，添加剩余的time_steps和done_idxs
    if step_counter > 0:
        time_steps += [step_counter] * (len(actions) - len(time_steps))
        done_idxs.append(len(actions) - 1)

    # 根据image_paths加载图像
    obss = list(image_loader(image_paths))

    # 初始化reward-to-go和returns为0
    rtg = np.zeros(len(actions))
    returns = np.zeros(len(actions) + 1)

    return obss, actions, returns, rtg, time_steps, done_idxs

