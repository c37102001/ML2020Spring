import math
import os
import json
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from metrics import Accuracy
from tqdm import tqdm
from ipdb import set_trace as pdb


class Trainer:
    def __init__(self, arch, model, batch_size, lr, device):
        self.arch = arch
        if not os.path.exists(arch):
            os.makedirs(arch)
        self.model = model
        self.batch_size = batch_size
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = StepLR(self.opt, step_size=30, gamma=0.5)
        self.criteria = torch.nn.CrossEntropyLoss()
        self.device = device
        self.history = {'train':[], 'valid':[]}
        self.best_score = math.inf

    def run_epoch(self, epoch, dataset, training, desc=''):
        self.model.train(training)
        shuffle = training

        dataloader = DataLoader(dataset, self.batch_size, shuffle=shuffle)
        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc)
        loss = 0
        acc = Accuracy()
        for i, (imgs, labels) in trange:        # (b, 3, 128, 128), (b, 1)
            labels = labels.view(-1)            # (b,)
            o_labels, batch_loss = self.run_iters(imgs, labels)
            
            if training:
                self.opt.zero_grad()
                batch_loss.backward()
                self.opt.step()
            
            loss += batch_loss.item()
            acc.update(o_labels.cpu(), labels)

            trange.set_postfix(
                loss=loss / (i+1),
                acc=acc.print_score() 
            )
            if i > 50:
                break

        if training:
            self.history['train'].append({'loss': loss / len(trange), 'acc': acc.get_score()})
        else:
            self.history['valid'].append({'loss': loss / len(trange), 'acc': acc.get_score()})
            self.save_hist()
            if loss < self.best_score:
                self.save_best()
            self.scheduler.step()

    def run_iters(self, imgs, labels):
        imgs = imgs.to(self.device)         # (b, 3, 128, 128)
        labels = labels.to(self.device)     # (b,)
        o_labels = self.model(imgs)         # (b, 11)
        batch_loss = self.criteria(o_labels, labels)
        return o_labels, batch_loss

    def save_best(self):
        torch.save(self.model.state_dict(), f'{self.arch}/model.ckpt')

    def save_hist(self):
        with open(f'{self.arch}/history.json', 'w') as f:
            json.dump(self.history, f, indent=4)

