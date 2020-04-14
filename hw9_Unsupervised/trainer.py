import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from unet import UNet
from utils import same_seeds
from torch.optim.lr_scheduler import StepLR
import os
import json
from tqdm import tqdm
from ipdb import set_trace as pdb

class Trainer:
    def __init__(self, arch, model, lr=1e-5, batch_size=64):
        self.arch = arch
        if not os.path.exists(f'{arch}/ckpt/'):
            os.makedirs(f'{arch}/ckpt/')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.batch_size = batch_size
        self.model = model
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.history = {'loss':[]}
        same_seeds(0)

    def run_epoch(self, epoch, img_dataset):
        self.model.train()

        img_dataloader = DataLoader(img_dataset, batch_size=self.batch_size, shuffle=True)

        trange = tqdm(img_dataloader, total=len(img_dataloader))
        total_loss = 0
        for i, (data, _) in enumerate(trange):
            img = data.to(self.device)

            encoded, output = self.model(img)
            loss = self.criterion(output, img)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            trange.set_postfix(loss=total_loss/(i+1))

            
        print('[epoch {}], loss:{:.5f}'.format(epoch+1, loss.data))
        self.history['loss'].append(total_loss/len(trange))

        torch.save(self.model.state_dict(), f'{self.arch}/ckpt/model.ckpt')
        self.save_hist()
        self.scheduler.step()
        
        if (epoch+1) % 10 == 0:
            torch.save(self.model.state_dict(), f'{self.arch}/ckpt/model_{epoch+1}.ckpt')

    def save_hist(self):
        with open(f'{self.arch}/history.json', 'w') as f:
            json.dump(self.history, f, indent=4)