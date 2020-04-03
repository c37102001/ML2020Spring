import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import argparse
from ipdb import set_trace as pdb
from model import *
from trainer import Trainer
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('arch', type=str)
parser.add_argument('--show_imgs', action='store_true')
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--do_predict', action='store_true')
parser.add_argument('--plot_lr', action='store_true')
parser.add_argument('--plot_label', action='store_true')
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--max_epoch', type=int, default=200)
args = parser.parse_args()


if args.show_imgs:
    from utils import no_axis_show, show_canny
    
    titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'television', 'dog', 'dolphin', 'spider']
    plt.figure(figsize=(18, 18))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        fig = no_axis_show(plt.imread(f'real_or_drawing/train_data/{i}/{500*i}.bmp'), title=titles[i])

    plt.figure(figsize=(18, 18))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        fig = no_axis_show(plt.imread(f'real_or_drawing/test_data/0/' + str(i).rjust(5, '0') + '.bmp'))

    show_canny()


source_transform = transforms.Compose([
    transforms.Grayscale(),     # gray scale for canny
    # transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),  # convert skimage.Image to np for cv2
    # transforms.ToPILImage(),    # convert np.array back to skimage.Image
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=(0,)),   # fill 0s to empty space
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    transforms.Grayscale(),     # from 3-dim to 1-dim
    transforms.Resize((32, 32)),    # from test-img size 28x28 to training img size 32x32
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=(0,)),
    transforms.ToTensor(),
])

source_dataset = ImageFolder('real_or_drawing/train_data', transform=source_transform)
target_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)

source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)
device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
arch = f'arch/{args.arch}'


if args.do_train:
    trainer = Trainer(arch, device)
    for epoch in range(args.max_epoch):
        trainer.run_epoch(epoch, source_dataloader, target_dataloader, lamb=0.1)


if args.do_predict:
    feature_extractor = FeatureExtractor()
    feature_extractor.load_state_dict(torch.load(f'{arch}/ckpts/extractor.ckpt'))
    feature_extractor.to(device)
    feature_extractor.eval()

    label_predictor = LabelPredictor()
    label_predictor.load_state_dict(torch.load(f'{arch}/ckpts/predictor.ckpt'))
    label_predictor.to(device)
    label_predictor.eval()
    
    result = []
    for i, (test_data, _) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        test_data = test_data.to(device)
        class_logits = label_predictor(feature_extractor(test_data))

        x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        result.append(x)
    
    result = np.concatenate(result)

    # Generate your submission
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv(f'{arch}/{args.arch}_submission.csv',index=False)


if args.plot_lr:
    from utils import plot_lr
    plot_lr(arch)

if args.plot_label:
    from utils import plot_distribution
    prediction = pd.read_csv(f'{arch}/{args.arch}_submission.csv')['label'].to_list()
    plot_distribution(arch, prediction)

print('done!')