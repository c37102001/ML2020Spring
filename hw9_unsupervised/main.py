import numpy as np
from torch.utils.data import DataLoader
from ipdb import set_trace as pdb
import torch
from trainer import Trainer
from utils import *
from model import AE, MyAE
from unet import UNet
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('arch', type=str)
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--do_predict', action='store_true')
parser.add_argument('--plot_cluster', action='store_true')
parser.add_argument('--plot_recon', action='store_true')
parser.add_argument('--plot_loss', action='store_true')
parser.add_argument('--plot_lc', action='store_true')
parser.add_argument('--plot_all', action='store_true')
args = parser.parse_args()
same_seeds(0)

arch = f'arch/{args.arch}'
trainX = np.load('data/trainX.npy')
trainX_preprocessed = preprocess(trainX)
img_dataset = Image_Dataset(trainX_preprocessed)

model = MyAE().cuda()

if args.do_train:
    n_epoch = 200
    if args.resume:
        model.load_state_dict(torch.load(f'{arch}/ckpt/model.ckpt'))
    trainer = Trainer(arch, model)
    if args.resume:
        with open(f'{arch}/history.json', 'r') as f:
            trainer.history = json.load(f)
    
    for epoch in range(n_epoch):
        trainer.run_epoch(epoch, img_dataset)
    print('done!')


if args.do_predict:
    # load model
    print('[*] Loading model...')
    model.load_state_dict(torch.load(f'{arch}/ckpt/model.ckpt'))
    model.cuda()
    model.eval()

    # predict
    latents = inference(X=trainX, model=model)
    pred, X_embedded = predict(latents)

    save_prediction(pred, f'{arch}/prediction.csv')
    save_prediction(invert(pred), f'{arch}/prediction_invert.csv')
    print('done!')


# ========================= Plot cluster results ============================
if args.plot_cluster or args.plot_all:
    valX = np.load('data/valX.npy')
    valY = np.load('data/valY.npy')

    model.load_state_dict(torch.load(f'{arch}/ckpt/model.ckpt'))
    model.cuda()
    model.eval()
    latents = inference(valX, model)
    pred_from_latent, emb_from_latent = predict(latents)
    acc_latent = cal_acc(valY, pred_from_latent)
    print('The clustering accuracy is:', acc_latent)
    print('The clustering result:')
    plot_scatter(emb_from_latent, valY, savefig=f'{arch}/p1_baseline.png')
    print('done!')


# ========================= Plot AE reconstructed results ============================
if args.plot_recon or args.plot_all:
    print('[*] Loading model...')
    model.load_state_dict(torch.load(f'{arch}/ckpt/model.ckpt'))
    model.cuda()
    model.eval()

    # plot origin images
    plt.figure(figsize=(10,4))
    indexes = [1,2,3,6,7,9]
    imgs = trainX[indexes,]
    
    for i, img in enumerate(imgs):
        plt.subplot(2, 6, i+1, xticks=[], yticks=[])
        plt.imshow(img)
        
    # plot reconstruct images
    inp = torch.Tensor(trainX_preprocessed[indexes,]).cuda()
    latents, recs = model(inp)
    recs = ((recs+1)/2 ).cpu().detach().numpy()
    recs = recs.transpose(0, 2, 3, 1)
    for i, img in enumerate(recs):
        plt.subplot(2, 6, 6+i+1, xticks=[], yticks=[])
        plt.imshow(img)
        
    plt.tight_layout()
    plt.savefig(f'{arch}/reconstruct.png')
    plt.clf()
    print('done!')


# ========================= Plot AE learning curve ============================
if args.plot_lc or args.plot_all:
    print('[*] Loading model...')
    model.load_state_dict(torch.load(f'{arch}/ckpt/model.ckpt'))
    model.cuda()
    model.eval()

    import glob
    valX = np.load('data/valX.npy')
    valY = np.load('data/valY.npy')
    ticks = [i for i in range(20, 200 + 1, 20)]
    checkpoints_list = [f'{arch}/ckpt/model_{i}.ckpt' for i in ticks]

    # load data
    dataset = Image_Dataset(trainX_preprocessed)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    points = []
    with torch.no_grad():
        for i, checkpoint in enumerate(checkpoints_list):
            print('[{}/{}] {}'.format(i+1, len(checkpoints_list), checkpoint))
            model.load_state_dict(torch.load(checkpoint))
            model.cuda()
            model.eval()
            err = 0
            n = 0
            for x in dataloader:
                x = x.cuda()
                _, rec = model(x)
                err += torch.nn.MSELoss(reduction='sum')(x, rec).item()
                n += x.flatten().size(0)
            print('Reconstruction error (MSE):', err/n)
            latents = inference(X=valX, model=model)
            pred, X_embedded = predict(latents)
            acc = cal_acc(valY, pred)
            print('Accuracy:', acc)
            points.append((err/n, acc))

    ps = list(zip(*points))
    plt.figure(figsize=(6,6))
    plt.subplot(211, title='Reconstruction error (MSE)').plot(ps[0])
    plt.subplot(212, title='Accuracy (val)').plot(ps[1])
    plt.xticks([i for i in range(len(ticks))], ticks)
    plt.savefig(f'{arch}/learning_curve')


# ========================= Plot AE loss ============================
if args.plot_loss or args.plot_all:
    from misc import plot_loss
    plot_loss(arch)