import csv
import numpy as np
import torch
import pandas as pd 
import time
import pyautogui as pg

import ctypes
import pynput
from pynput.keyboard import Key, Listener, KeyCode
from pynput import keyboard

SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)

w_char=0x11
s_char=0x1F
a_char=0x1E
d_char=0x20
q_char=0x10
n_char=0x31 # is bound on my machine to clear decals
r_char=0x13
one_char=0x02
two_char=0x03
three_char=0x04
four_char=0x05
five_char=0x06
seven_char=0x08
ctrl_char=0x1D
shift_char=0x2A
space_char=0x39
b_char=0x30
i_char=0x17
v_char=0x2F
h_char=0x23
o_char=0x18
p_char=0x19
e_char=0x12
c_char_=0x2E
t_char=0x14
u_char=0x16
m_char=0x32
g_char=0x22
k_char=0x25
x_char=0x2D
c_char2=0x2E
y_char=0x15
under_char=0x0C # actually minus, use in combo w shift for underscore
cons_char=0x29
ret_char=0x1C
esc_char=0x01

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
            

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# def set_pos(x, y):
#     x = 1 + int(x * 65536./1920.)
#     y = 1 + int(y * 65536./1080.)
#     extra = ctypes.c_ulong(0)
#     ii_ = pynput._util.win32.INPUT_union()
#     ii_.mi = pynput._util.win32.MOUSEINPUT(x, y, 0, (0x0001 | 0x8000), 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
#     command=pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
#     SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

def set_pos(x, y):
    # raw input off, mouse sensitivity 2.50, res 800x600, box =400
    # x = 1 + int(x * 65536./1920)
    # y = 1 + int(y * 65536./1080)w

    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(x, y, 0, (0x0001 | 0x8000), 0, ctypes.pointer(extra))
    command=Input(ctypes.c_ulong(0), ii_)

    # comment this out if want to test something without movement
    SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

def HoldKey(hexKeyCode):
    # with ctypes only
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def left_click():
    # https://www.reddit.com/r/learnpython/comments/bognbs/direct_input_for_python_with_pynput/
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0002, 0, ctypes.pointer(extra))
    x= Input(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0004, 0, ctypes.pointer(extra))
    x= Input(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def hold_left_click():
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0002, 0, ctypes.pointer(extra))
    x=Input(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def release_left_click():
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0004, 0, ctypes.pointer(extra))
    x=Input(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def right_click():
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0008, 0, ctypes.pointer(extra))
    x= Input(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0010, 0, ctypes.pointer(extra))
    x= Input(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def hold_right_click():
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0008, 0, ctypes.pointer(extra))
    x=Input(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def release_right_click():
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(0, 0, 0, 0x0010, 0, ctypes.pointer(extra))
    x=Input(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))



path = 'labelled2.csv'


mouse = pd.read_csv(path, usecols=['mouse_x', 'mouse_y'])
mouse = np.array(mouse)

click_left = pd.read_csv(path, usecols=['click_left'])
click_left = np.array(click_left)

click_right = pd.read_csv(path, usecols=[ 'click_right'])
click_right = np.array(click_right)

scroll = pd.read_csv(path, usecols=['scroll'])
scroll = np.array(scroll)

w = pd.read_csv(path, usecols=['w'])
w = np.array(w)

a = pd.read_csv(path, usecols=['a'])
a = np.array(a)

s = pd.read_csv(path, usecols=['s'])
s = np.array(s)

d = pd.read_csv(path, usecols=['d'])
d = np.array(d)

r = pd.read_csv(path, usecols=['r'])
r = np.array(r)

q = pd.read_csv(path, usecols=['q'])
q = np.array(q)

shift = pd.read_csv(path, usecols=['shift'])
shift = np.array(shift)

space = pd.read_csv(path, usecols=['space'])
space = np.array(space)

ctrl = pd.read_csv(path, usecols=['ctrl'])
ctrl = np.array(ctrl)


try:               
    for i in range(len(mouse)):
        start_time=time.time()
        if mouse[i][0]!=0 or mouse[i][1]!=0:
            set_pos(int(mouse[i][0]), int(mouse[i][1]))  # 基于当前位置瞬间向下移动200像素    
        if click_left[i]!=0:
            hold_left_click()
        if click_left[i]==0:
            release_left_click()
        if click_right[i]!=0:
            right_click()        
        if w[i]!=0:
            HoldKey(w_char)
        if w[i]==0:
            ReleaseKey(w_char)
        if a[i]!=0:
            HoldKey(a_char)
        if a[i]==0:
            ReleaseKey(a_char)
        if s[i]!=0:
            HoldKey(s_char)
        if s[i]==0:
            ReleaseKey(s_char)
        if d[i]!=0:
            HoldKey(d_char)
        if d[i]==0:
            ReleaseKey(d_char)
        if r[i]!=0:
            HoldKey(r_char)
        if r[i]==0:
            ReleaseKey(r_char)
        if q[i]!=0:
            HoldKey(q_char)
        if q[i]==0:
            ReleaseKey(q_char)
        if shift[i]!=0:
            HoldKey(shift_char)
        if shift[i]==0:
            ReleaseKey(shift_char)
        if space[i]!=0:
            HoldKey(space_char)
        if space[i]==0:
            ReleaseKey(space_char)
        if ctrl[i]!=0:
            HoldKey(ctrl_char)
        if ctrl[i]==0:
            ReleaseKey(ctrl_char)
        time.sleep(0.1)
        end_time=time.time()
        print(end_time-start_time)
except KeyboardInterrupt:
    print("\n") 


