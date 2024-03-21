import time, keyboard

import numpy
import win32api
from PIL import Image
from PIL import ImageGrab
import win32gui
import winsound
from api.api import CSAPI
from api.api import normalizeAngles
import matplotlib.pyplot as plt00
import cv2
from dem2lable import find_nearest_index_from_list

# mouse_y_possibles = [-10,-9,-8,-7,-6,-5,-4, -3, -2, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,3,4,5,6,7,8,9,10]
# mouse_x_possibles = [-170,-130,-100,-80,-70,-60,-50.0,-40.0,-30.0,-20.0,-18.0,-16.0,-14.0,-12.0,-10.0,-8.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,8.0,10.0,12.0,14.0,16.0,18.0,20.0,30.0,40.0,50.0,60,70,80,100,130,170]

mouse_y_possibles = [-10,-9,-8,-7,-6,-5,-4, -3, -2, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,3,4,5,6,7,8,9,10]
mouse_x_possibles = [-170,-130,-100,-80,-70,-60,-50.0,-40.0,-30.0,-20.0,-18.0,-16.0,-14.0,-12.0,-10.0,-8.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,8.0,10.0,12.0,14.0,16.0,18.0,20.0,30.0,40.0,50.0,60,70,80,100,130,170]

#压缩图片，损失小压缩率高
def compress_image(image, quality=75):  # PIL RGB
    image = image.convert("RGB")
    image = image.resize((800, 600), Image.BILINEAR)
    image = image.convert("RGB")
    image.save("temp.jpg", quality=quality)
    image = Image.open("temp.jpg")
    return image


    

    


def convert_frames_to_video(frame_array, pathOut, fps):
    frames = []
    for frame in frame_array:
        #reading each files
        img = numpy.array(frame)[-600:, -800:, ::-1].copy()  # RGB PIL to numpy BGR
        height, width, layers = img.shape
        size = (width, height)
        #inserting the frames into an image array
        frames.append(img)
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(frames)):
        # writing to a image array
        out.write(frames[i])
    out.release()


def action_list_to_label(actions, out_path):
    labels = []
    for action in actions:
        d_pitch, d_yaw, fire = action
        mouse_y = find_nearest_index_from_list(mouse_y_possibles, d_pitch)
        mouse_x = find_nearest_index_from_list(mouse_x_possibles, d_yaw)
        fire = int(fire)
        action_str = "0,0,0,0,0,0,{},{},{},0,0,0,0,0,0,0".format(str(mouse_x),str(mouse_y),str(fire)).replace(",","\t")
        labels.append(action_str)
    with open("./labels/{}".format(out_path), "w") as f:
        f.write("clockTime,tick,w,a,s,d,mouse_x,mouse_y,is_fire,is_scope,is_jump,is_crouch,is_walking,is_reload,is_e,switch\n".replace(
                ",", "\t"))
        f.write("\n".join(labels))


handle = CSAPI(r".\api\csgo.json")
mouse_movements = []
game_frames = []
# hold "s" to start
while not keyboard.is_pressed("s"):
    pass
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
# grab pic
hwnd = win32gui.FindWindow(None, u"Counter-Strike: Global Offensive - Direct3D 9")
dimensions = win32gui.GetWindowRect(hwnd)
print("dimension", dimensions)
image = ImageGrab.grab(dimensions)  # PIL rgb
game_frames.append(image)
pitch, yaw = handle.get_current_xy()
time.sleep(1/16)
while not keyboard.is_pressed("e"):
    # hold p to pause/start
    if keyboard.is_pressed("p"):
        while not keyboard.is_pressed("s"):
            pass
    t0 = time.time()
    # compare the yaw/pitch, save angles change to history list
    fire = bool(win32api.GetKeyState(0x01) < 0)
    n_pitch, n_yaw = handle.get_current_xy()
    d_pitch, d_yaw = normalizeAngles(n_pitch - pitch, n_yaw - yaw)

    mouse_movements.append([d_pitch, d_yaw, fire])
    t1 = time.time()
    game_frames.append(ImageGrab.grab(dimensions))
    print("grab cost", (time.time()- t0))
    time.sleep(1/16-(time.time() - t0) if 1/16-(time.time() - t0) > 0 else 0)
    pitch, yaw = n_pitch, n_yaw
# preprocess the video
game_frames = game_frames[:-1]  # RGB PIL inside
double_frames = []
# the 5eplay video we recorded has 32 fps, here we simply repeat the same frame to cope with that.
# during training, they does the same effect because we skip 1 frame each time we pick a frame.
for each in game_frames:
    double_frames.append(each)
    double_frames.append(each)
fps = 32
# the 5eplay video we recorded has 12 seconds of invalid frames, we therefore keep our expert data same format here.
skip_frame = 32*12-1
game_frames = double_frames[-skip_frame:] + double_frames

print("mouse len ", len(mouse_movements))
print("frames len", len(game_frames))
time = int(time.time())
convert_frames_to_video(game_frames, r"./expert_{}.mp4".format(time), fps)
action_list_to_label(mouse_movements, r"./expert/expert_{}.csv".format(time))

# print("mouse ", mouse_movements)
# plt.subplot(1, 2, 1)
# plt.hist([each[0] for each in mouse_movements], bins=180)
# plt.xlabel("pitch")
# plt.title("竖直行为频率绘图")
# plt.subplot(1, 2, 2)
# plt.hist([each[1] for each in mouse_movements], bins=60)
# plt.xlabel("yaw")
# plt.title("水平行为频率绘图")
# plt.show()