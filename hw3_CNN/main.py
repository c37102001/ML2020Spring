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
from trainer import Trainer
from dataset import ImgDataset
import torchvision.transforms as transforms
from tqdm import tqdm
from models import *
import torch.backends.cudnn as cudnn

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('arch', type=str)
parser.add_argument('--do_preprocess', action='store_true')
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--do_predict', action='store_true')
parser.add_argument('--do_plot', action='store_true')
parser.add_argument('--do_all', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--cuda', type=int, default=0)
args = parser.parse_args()

arch = f'arch/{args.arch}'
img_size = 128

def readfile(path, label=False, cnt_mean_std=False):  # path = 'food-11/training'
    img_dir = os.listdir(path)
    x = np.zeros((len(img_dir), img_size, img_size, 3), dtype=np.uint8)
    y = np.zeros(len(img_dir), dtype=np.uint8)
    
    for i, file in tqdm(enumerate(sorted(img_dir)), total=len(img_dir)):   # file = '0_0.jpg'
        img = cv2.imread(os.path.join(path, file))   # (512, 512, 3)
        x[i] = cv2.resize(img, (img_size, img_size))
        if label:
            y[i] = int(file.split('_')[0])
    
    if cnt_mean_std:
        mean = x.reshape(len(img_dir), -1, 3).mean(1).mean(0) / 255
        std = x.reshape(len(img_dir), -1, 3).std(1).mean(0) / 255
        print('Mean: ', mean)
        print('Std: ', std)

    if label:
        return x, y
    else:
        return x

def build_model():
    # Model
    print('[*] Building model...')
    # model = VanillaCNN()
    # model = VanillaFCN()
    # model = VGG('VGG16')
    # model = ResNet18()
    model = ResNet101()
    # model = PreActResNet18()
    # model = GoogLeNet()
    # model = DenseNet121()
    # model = ResNeXt29_2x64d()
    # model = MobileNet()
    # model = MobileNetV2()
    # model = DPN92()
    # model = ShuffleNetG2()
    # model = SENet18()
    # model = ShuffleNetV2(1)
    # model = EfficientNetB0()
    
    return model

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomApply([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomAffine(45),
        transforms.ColorJitter(brightness=0.5),
        transforms.ColorJitter(contrast=0.5),
        transforms.ColorJitter(saturation=0.5),
        transforms.ColorJitter(hue=0.3),
        transforms.RandomGrayscale(p=0.1),
    ]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.343, 0.451, 0.555], std=[0.239, 0.240, 0.230])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.343, 0.451, 0.555], std=[0.239, 0.240, 0.230])
])

if args.do_preprocess:
    train_x, train_y = readfile('food-11/training', True, True)         # (9866, 128, 128, 3), (9866,)
    valid_x, valid_y = readfile('food-11/validation', True)             # (3430, 128, 128, 3), (3430,)
    test_x = readfile('food-11/testing', False)
    train_val_x = np.concatenate((train_x, valid_x), axis=0)
    train_val_y = np.concatenate((train_y, valid_y), axis=0)

    print('[*] Saving train dataset...')
    with open('preprocessed/train_x.pkl', 'wb') as f:
        pickle.dump(train_x, f)
    with open('preprocessed/train_y.pkl', 'wb') as f:
        pickle.dump(train_y, f)

    print('[*] Saving valid dataset...')
    with open('preprocessed/valid_x.pkl', 'wb') as f:
        pickle.dump(valid_x, f)
    with open('preprocessed/valid_y.pkl', 'wb') as f:
        pickle.dump(valid_y, f)

    print('[*] Saving test dataset...')
    with open('preprocessed/test_x.pkl', 'wb') as f:
        pickle.dump(test_x, f)

    print('[*] Saving train-valid dataset...')
    with open('preprocessed/train_val_x.pkl', 'wb') as f:
        pickle.dump(train_val_x, f)
    with open('preprocessed/train_val_y.pkl', 'wb') as f:
        pickle.dump(train_val_y, f)


if args.do_train or args.do_all:
    print('[*] Loading pickles...')
    with open('preprocessed/train_x.pkl', 'rb') as f:
        train_x = pickle.load(f)
    with open('preprocessed/train_y.pkl', 'rb') as f:
        train_y = pickle.load(f)
    with open('preprocessed/valid_x.pkl', 'rb') as f:
        valid_x = pickle.load(f)
    with open('preprocessed/valid_y.pkl', 'rb') as f:
        valid_y = pickle.load(f)
    with open('preprocessed/train_val_x.pkl', 'rb') as f:
        train_val_x = pickle.load(f)
    with open('preprocessed/train_val_y.pkl', 'rb') as f:
        train_val_y = pickle.load(f)
    

    train_dataset = ImgDataset(train_x, train_y, transform=train_transform)
    valid_dataset = ImgDataset(valid_x, valid_y, transform=test_transform)
    train_val_dataset = ImgDataset(train_val_x, train_val_y, transform=train_transform)

    model = build_model()
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    if not device == 'cpu':
        cudnn.benchmark = True
    
    trainer = Trainer(arch, model, args.batch_size, args.lr, device)

    print('[*] Start training...')
    for epoch in range(args.max_epoch):
        print('Epoch: {}'.format(epoch))
        trainer.run_epoch(epoch, train_dataset, training=True, desc='[Train]')
        trainer.run_epoch(epoch, valid_dataset, training=False, desc='[Valid]')
    
    print('[*] Training with full dataset')
    for epoch in range(30):
        print('Epoch: {}'.format(epoch))
        trainer.run_epoch(epoch, train_val_dataset, training=True, desc='[Total]')
        trainer.save_best()


if args.do_predict or args.do_all:
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.343, 0.451, 0.555], std=[0.239, 0.240, 0.230])
    ])

    with open('preprocessed/test_x.pkl', 'rb') as f:
        test_x = pickle.load(f)
    test_dataset = ImgDataset(test_x, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = build_model()
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    if not device == 'cpu':
        cudnn.benchmark = True
    model.load_state_dict(torch.load(f'{arch}/model.ckpt'))
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


if args.do_plot or args.do_all:
    with open(f'{arch}/history.json', 'r') as f:
        history = json.loads(f.read())

    train_loss = [l['loss'] for l in history['train']]
    valid_loss = [l['loss'] for l in history['valid']]

    plt.figure(figsize=(7, 5))
    plt.title('Loss')
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{arch}/Loss.png')
    print('Lowest Loss ', min([[l['loss'], idx + 1] for idx, l in enumerate(history['valid'])]))

    train_acc = [l['acc'] for l in history['train']]
    valid_acc = [l['acc'] for l in history['valid']]
    plt.figure(figsize=(7, 5))
    plt.title('Acc')
    plt.plot(train_acc, label='train')
    plt.plot(valid_acc, label='valid')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{arch}/Acc.png')
    print('Best acc', max([[l['acc'], idx + 1] for idx, l in enumerate(history['valid'])]))
