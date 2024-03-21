import win32gui
import win32ui
import win32con
import win32api
import os,time
from PIL import ImageGrab
import numpy
import cv2
from PIL import Image

now=time.strftime('%y%m%d%H%M',time.localtime())
path='./result/'
if not os.path.isdir(path):
    os.mkdir(path)
path='./result/'+now
if not os.path.isdir(path):
    os.mkdir(path)

times = int(time.time())

# hwnd = win32gui.FindWindow("NotePad",None)
i=0

game_frames = []

def output(frame_array, pathOut, fps):
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

def compress_image(image, quality=75):  # PIL RGB
    image = image.convert("RGB")
    image = image.resize((800, 600), Image.BILINEAR)
    image = image.convert("RGB")
    image.save("temp.jpg", quality=quality)
    image = Image.open("temp.jpg")
    return image

def output_img(frame_array):
    frames = []
    j=0
    for frame in frame_array:
        #reading each files
        j+=1
        img = numpy.array(frame)[-1080:, -1920:, ::-1].copy()  # RGB PIL to numpy BGR
        # print(img)
        #inserting the frames into an image array
        compress_image(Image.fromarray(img)).save(path+'/{}.jpg'.format(j), quality=75)
        

hdesktop = win32gui.GetDesktopWindow()
dimensions = win32gui.GetWindowRect(hdesktop)

try:
    while i<100:
        i+=1
        # 获取桌面
        image = ImageGrab.grab(dimensions)
        game_frames.append(image)

        
except KeyboardInterrupt:
    print('结束')

game_frames = game_frames[:-1]
fps=32
# output(game_frames,  r"./expert_{}.mp4".format(times), fps) #输出视频
output_img(game_frames)

# # 分辨率适应
# width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
# height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
# left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
# top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
# # 创建设备描述表
# desktop_dc = win32gui.GetWindowDC(hdesktop)
# img_dc = win32ui.CreateDCFromHandle(desktop_dc)
# # 创建一个内存设备描述表
# mem_dc = img_dc.CreateCompatibleDC()
# # 创建位图对象
# screenshot = win32ui.CreateBitmap()
# screenshot.CreateCompatibleBitmap(img_dc, width, height)
# mem_dc.SelectObject(screenshot)
# # 截图至内存设备描述表
# mem_dc.BitBlt((0, 0), (width, height), img_dc, (0, 0), win32con.SRCCOPY)
# # 将截图保存到文件中
# screenshot.SaveBitmapFile(mem_dc, path+'/{}.png'.format(time))
# # 内存释放
# mem_dc.DeleteDC()
# win32gui.DeleteObject(screenshot.GetHandle())


