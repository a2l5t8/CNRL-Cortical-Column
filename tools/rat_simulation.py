import numpy as np
from matplotlib import pyplot as plt
import random
import math
import pandas as pd
import tqdm
import seaborn as sns


WINDOW_WIDTH = 50
WINDOW_HEIGHT = 50

window_x = [-WINDOW_WIDTH/2, +WINDOW_WIDTH/2]
window_y = [-WINDOW_HEIGHT/2, +WINDOW_HEIGHT/2]

def conv(angel) : 
    x = np.cos(np.radians(angel))
    y = np.sin(np.radians(angel))

    return x, y

def random_walk(length, R = 20,  initialize = True) : 

    if(initialize) : 
        pos_x = [0]
        pos_y = [0]

    theta = np.random.randint(0, 360)
    cnt = 0
    length_cnt = 0

    for _ in range(length) : 

        dist = np.sqrt(pos_x[-1]**2 + pos_y[-1]**2)
        # print(dist)
        if(dist > R) : 
            ang = np.angle(complex(pos_x[-1], pos_y[-1]), deg = True)
            theta = ang + np.random.randint(90, 180) % 360

        # theta = np.random.randint(0, 360)

        pos_x.append(pos_x[-1] + (conv(theta)[0] + 1/5 * np.random.uniform(-0.5,0.5)) * 1/10)
        pos_y.append(pos_y[-1] + (conv(theta)[1] + 1/5 * np.random.uniform(-0.5,0.5)) * 1/10)
            
    return pos_x, pos_y
    

def walk_initialize(length, theta = 0) : 

    pos_x = [0]
    pos_y = [0]

    cnt = 0
    length_cnt = 0

    for _ in range(length) : 
        # theta = np.random.randint(0, 360)

        pos_x.append(pos_x[-1] + (conv(theta)[0])/9)
        pos_y.append(pos_y[-1] + (conv(theta)[1])/9)
    
    pos_x = pos_x + pos_x[::-1][1:-1]
    pos_y = pos_y + pos_y[::-1][1:-1]
      
    return pos_x, pos_y

def generate_walk(length, R, theta = 0) : 
    init_x, init_y = walk_initialize(10, theta = theta)
    walk_x, walk_y = random_walk(length, R = 10)
    pos_x, pos_y = init_x + walk_x, init_y + walk_y

    return pos_x, pos_y

def generate_walk_multi(length, R, theta_lst, n = 1) :
    Px = []
    Py = []

    walk_x, walk_y = random_walk(length, R = 10)

    for i in range(n) : 
        init_x, init_y = walk_initialize(30, theta = theta_lst[i])
        pos_x, pos_y = init_x + walk_x, init_y + walk_y

        Px.append(pos_x)
        Py.append(pos_y)
    
    return Px, Py


def speed_vector_converter(Xw, Yw) : 

    X = []
    Y = []

    for i in range(1, len(Xw)) : 
        X.append(Xw[i] - Xw[i - 1])
        Y.append(Yw[i] - Yw[i - 1])
    
    return (X, Y)