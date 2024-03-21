import time,os
import pyautogui as pag
import keyboard as kb
from pynput import mouse
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import *
from PIL import ImageGrab
import numpy
import win32gui
import sys

'''
缺少鼠标像素和水平视角比值
缺少gui
时间帧对齐
'''
now=time.strftime('%y%m%d%H%M',time.localtime())
path='./result/'
if not os.path.isdir(path):
    os.mkdir(path)
path='./result/'+now
if not os.path.isdir(path):
    os.mkdir(path)
f=open(path+'/ctrl_data.txt','w')
f.close()

hwnd = win32gui.FindWindow(None, 'C:\Windows\system32\cmd.exe')
app = QApplication(sys.argv)
screen = QApplication.primaryScreen()

i=0;prex=0;prec=0;pres=0

def callback(x):
    global prex
    event_type=x.event_type
    name=x.name
    if x!=prex:
        f.write('%s:%s ' % (name,event_type))
        print(x,end=' ')
        prex=x
def on_click(x, y, button, pressed):
    global prec
    if pressed:
        if prec!=button:
            f.write('{}:down '.format(button))
            print(button,pressed,end=' ')
            prec=button
    else:
        if prec!=0:
            f.write('{}:up '.format(button))
            print(button,pressed,end=' ')
        prec=0
        return False
def on_scroll(x,y,dx,dy):
    global pres
    if pres!=0:
        f.write('scr:{} '.format('up' if dy>0 else 'down'))
        pres=0

try:
    while True:
        f=open(path+'/ctrl_data.txt','a')
        i+=1
        
        #f.write(str(i))
        # 获取屏幕的尺寸
        screenWidth, screenHeight = pag.size()
        #截屏
        
        img = numpy.array(frame)[-600:, -800:, ::-1].copy()
        img.save(path+'/{}.png'.format(i))
        # pag.screenshot(path+'/{}.png'.format(i))
        # pag.screenshot(path+'/img{}.png'.format(i),region=(350, 200, 1400 ,800)) #xyxy
        x, y = pag.position()
        #记录键盘与鼠标
        kb.hook(callback)
        lis=mouse.Listener(on_click=on_click,on_scroll=on_scroll)
        lis.start()
        print('屏幕宽高: (%s %s),  鼠标坐标 : (%s, %s)' % (screenWidth, screenHeight, x, y))
        # # 每个1s中打印一次 , 并执行清屏
        # #prex=-1#这个控制按键是down/up模式还是持续模式
        time.sleep(0.0001)
        prec=-1;pres=-1
        # 执行系统清屏指令
        os.system('cls')
        f.write('%s:%s sys time is %s\n'%(x,y,i))
        f.close()
except KeyboardInterrupt:
    print('结束')

