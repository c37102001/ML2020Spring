import os
import json
import pickle
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from ipdb import set_trace as pdb
from model import Classifier
from trainer import Trainer
from dataset import ImgDataset
import torchvision.transforms as transforms
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('arch', type=str)
parser.add_argument('--do_preprocess', action='store_true')
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--do_predict', action='store_true')
parser.add_argument('--do_plot', action='store_true')
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--cuda', type=int, default=0)
args = parser.parse_args()

arch = f'arch/{args.arch}'

def readfile(path, label):  # path = 'food-11/training'
    img_dir = os.listdir(path)
    x = np.zeros((len(img_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros(len(img_dir), dtype=np.uint8)
    for i, file in tqdm(enumerate(sorted(img_dir)), total=len(img_dir)):   # file = '0_0.jpg'
        img = cv2.imread(os.path.join(path, file))   # (512, 512, 3)
        x[i] = cv2.resize(img, (128, 128))
        if label:
            y[i] = int(file.split('_')[0])
    if label:
        return x, y
    else:
        return x

if args.do_preprocess:
    train_x, train_y = readfile('food-11/training', True)       # (9866, 128, 128, 3), (9866,)
    valid_x, valid_y = readfile('food-11/validation', True)     # (3430, 128, 128, 3), (3430,)
    test_x = readfile('food-11/testing', False)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    train_dataset = ImgDataset(train_x, train_y, transform=train_transform)
    valid_dataset = ImgDataset(valid_x, valid_y, transform=train_transform)
    test_dataset = ImgDataset(test_x, transform=test_transform)

    print('[*] Saving training dataset...')
    with open('preprocessed/train.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)

    print('[*] Saving valid dataset...')
    with open('preprocessed/valid.pkl', 'wb') as f:
        pickle.dump(valid_dataset, f)

    print('[*] Saving test dataset...')
    with open('preprocessed/test.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)

if args.do_train:
    print('[*] Loading pickles...')
    with open('preprocessed/train.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open('preprocessed/valid.pkl', 'rb') as f:
        valid_dataset = pickle.load(f)

    model = Classifier()
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    trainer = Trainer(arch, model, args.batch_size, args.lr, train_dataset, valid_dataset, device)

    print('[*] Start training...')
    for epoch in range(args.max_epoch):
        print('Epoch: {}'.format(epoch))
        trainer.run_epoch(epoch, True)
        trainer.run_epoch(epoch, False)

if args.do_predict:
    with open('preprocessed/test.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = Classifier()
    model.load_state_dict(torch.load(f'{arch}/model.ckpt'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train(False)

    trange = tqdm(test_dataloader, total=len(test_dataloader), desc='[Predict]')
    prediction = []
    for x, _ in trange:                        # (b, 3, 128, 128)
        x = x.to(device)
        o_labels = model(x)                 # (b, 11)
        o_labels = o_labels.argmax(dim=1)   # (b,)
        o_labels = o_labels.cpu().numpy().tolist()
        prediction.extend(o_labels)

    with open(f'{arch}/prediction.csv', 'w') as f:
        f.write('Id,Category\n')
        for i, predict in enumerate(prediction):
            f.write(f'{i},{predict}\n')

if args.do_plot:
    with open(f'{arch}/history.json', 'r') as f:
        history = json.loads(f.read())

    train_loss = [l['loss'] for l in history['train']]
    valid_loss = [l['loss'] for l in history['valid']]

    plt.figure(figsize=(7, 5))
    plt.title('Loss')
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.legend()
    plt.savefig(f'{arch}/Loss.png')
    print('Lowest Loss ', min([[l['loss'], idx + 1] for idx, l in enumerate(history['valid'])]))

    train_acc = [l['acc'] for l in history['train']]
    valid_acc = [l['acc'] for l in history['valid']]
    plt.figure(figsize=(7, 5))
    plt.title('Acc')
    plt.plot(train_acc, label='train')
    plt.plot(valid_acc, label='valid')
    plt.legend()
    plt.savefig(f'{arch}/Acc.png')
    print('Best acc', max([[l['acc'], idx + 1] for idx, l in enumerate(history['valid'])]))
