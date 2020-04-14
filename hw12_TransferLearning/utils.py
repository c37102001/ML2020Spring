import cv2
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import json
from collections import Counter
from torch.utils.data import Dataset
from ipdb import set_trace as pdb
from PIL import Image
import os
from tqdm import tqdm
import torch.nn.functional as F


def readfile(path):
    img_dir = os.listdir(path)
    x = np.zeros((len(img_dir), 28, 28, 3), dtype=np.uint8)
    
    for i, file in tqdm(enumerate(sorted(img_dir)), total=len(img_dir)):   # file = '0_0.jpg'
        img = cv2.imread(os.path.join(path, file))   # (512, 512, 3)
        x[i] = cv2.resize(img, (28, 28))
    
    return x


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


def plot_distribution(arch, labels, name='label_analysis'):
    c = Counter(labels)
    keys = sorted(c.keys())
    counts = [c[k] for k in keys]
    
    plt.bar(keys, counts)
    plt.xticks(keys)
    plt.savefig(f'{arch}/{name}.png')
    plt.clf()


# ======================= fine-tune =================================

class ImgDataset(Dataset):
    def __init__(self, x, y, transform=None):       # (9866, 128, 128, 3), (9866,)
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)

        Y = self.y[index]
        Y = torch.LongTensor([Y])
        
        return X, Y

def get_sudo_label(arch, device, test_dataloader, feature_extractor, label_predictor):
    if not os.path.exists(f'{arch}/sudo_data'):
        os.makedirs(f'{arch}/sudo_data')

    feature_extractor.to(device)
    feature_extractor.eval()

    label_predictor.to(device)
    label_predictor.eval()

    accept_indice = []
    sudo_labels = []
    total_labels = []
    desc = '[Predict sudo labels...]'
    for i, (test_data, _) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=desc):
        test_data = test_data.to(device)
        class_logits = label_predictor(feature_extractor(test_data))    # (b, 10)
        class_logits = F.softmax(class_logits, dim=1)      # (b, 10)

        prob, label = torch.max(class_logits, dim=1)
        total_labels.append(label.cpu().detach().numpy())

        accept_index = torch.where(prob > 0.9)[0].cpu().numpy()
        sudo_label = label[accept_index]                 # (b')
        accept_index += test_dataloader.batch_size * i

        sudo_label = sudo_label.cpu().detach().numpy()
        sudo_labels.append(sudo_label)
        accept_indice.append(accept_index)
    
    accept_indice = np.concatenate(accept_indice)
    sudo_labels = np.concatenate(sudo_labels)
    total_labels = np.concatenate(total_labels)
    
    return accept_indice, sudo_labels, total_labels
