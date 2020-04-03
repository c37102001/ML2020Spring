import cv2
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import json
from collections import Counter


def no_axis_show(img, title='', cmap=None):
    fig = plt.imshow(img, interpolation='nearest', cmap=cmap)
    # hide axis
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)


def show_canny():
    plt.figure(figsize=(18, 18))

    original_img = plt.imread(f'real_or_drawing/train_data/0/0.bmp')
    plt.subplot(1, 5, 1)
    no_axis_show(original_img, title='original')

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    plt.subplot(1, 5, 2)
    no_axis_show(gray_img, title='gray scale', cmap='gray')

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    plt.subplot(1, 5, 2)
    no_axis_show(gray_img, title='gray scale', cmap='gray')

    canny_50100 = cv2.Canny(gray_img, 50, 100)
    plt.subplot(1, 5, 3)
    no_axis_show(canny_50100, title='Canny(50, 100)', cmap='gray')

    canny_150200 = cv2.Canny(gray_img, 150, 200)
    plt.subplot(1, 5, 4)
    no_axis_show(canny_150200, title='Canny(150, 200)', cmap='gray')

    canny_250300 = cv2.Canny(gray_img, 250, 300)
    plt.subplot(1, 5, 5)
    no_axis_show(canny_250300, title='Canny(250, 300)', cmap='gray')


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plot_lr(arch):
    with open(f'{arch}/history.json', 'r') as f:
        history = json.loads(f.read())

    d_loss = history['d_loss']
    f_loss = history['f_loss']
    acc = history['acc']

    plt.figure(figsize=(7, 5))
    plt.title('Loss')
    plt.plot(d_loss, label='d_loss')
    plt.plot(f_loss, label='f_loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{arch}/Loss.png')

    plt.figure(figsize=(7, 5))
    plt.title('Acc')
    plt.plot(acc, label='accuracy')
    plt.grid(True)
    plt.savefig(f'{arch}/Acc.png')

def plot_distribution(arch, labels):
    c = Counter(labels)
    keys = sorted(c.keys())
    counts = [c[k] for k in keys]
    
    plt.bar(keys, counts)
    plt.xticks(keys)
    plt.savefig(f'{arch}/label_analysis.png')

