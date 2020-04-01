import numpy as np
from torch.utils.data import DataLoader
from ipdb import set_trace as pdb
import torch
from trainer import Trainer
from utils import *
from model import AE
from utils import same_seeds
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--do_predict', action='store_true')
parser.add_argument('--plot_cluster', action='store_true')
parser.add_argument('--plot_AE', action='store_true')
args = parser.parse_args()
same_seeds(0)

trainX = np.load('data/trainX.npy')
trainX_preprocessed = preprocess(trainX)
img_dataset = Image_Dataset(trainX_preprocessed)


if args.do_train:
    n_epoch = 100
    trainer = Trainer()
    for epoch in range(n_epoch):
        trainer.run_epoch(epoch, img_dataset)


if args.do_predict:
    # load model
    print('[*] Loading model...')
    model = AE().cuda()
    model.load_state_dict(torch.load('./checkpoints/checkpoint_100.pth'))
    model.eval()

    # predict
    latents = inference(X=trainX, model=model)
    pred, X_embedded = predict(latents)

    save_prediction(pred, 'results/prediction.csv')
    save_prediction(invert(pred), 'results/prediction_invert.csv')


if args.plot_cluster:
    valX = np.load('data/valX.npy')
    valY = np.load('data/valY.npy')

    model = AE().cuda()
    model.load_state_dict(torch.load('./checkpoints/checkpoint_100.pth'))
    model.eval()
    latents = inference(valX, model)
    pred_from_latent, emb_from_latent = predict(latents)
    acc_latent = cal_acc(valY, pred_from_latent)
    print('The clustering accuracy is:', acc_latent)
    print('The clustering result:')
    plot_scatter(emb_from_latent, valY, savefig='result/p1_baseline.png')


if args.plot_AE:
    # load model
    print('[*] Loading model...')
    model = AE().cuda()
    model.load_state_dict(torch.load('./checkpoints/checkpoint_100.pth'))
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
    plt.savefig('results/reconstruct.png')