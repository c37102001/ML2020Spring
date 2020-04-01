import math
import os
import json
import torch
import random
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR
from metrics import Accuracy
from tqdm import tqdm
from ipdb import set_trace as pdb
from utils import same_seeds


class Trainer:
    def __init__(self, arch, model, batch_size, lr, accum_steps, device):
        self.arch = arch
        if not os.path.exists(arch):
            os.makedirs(arch)
        self.model = model
        self.batch_size = batch_size
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-2)
        self.scheduler = StepLR(self.opt, step_size=5, gamma=0.5)
        self.accum_steps = accum_steps
        self.criteria = torch.nn.CrossEntropyLoss()
        self.device = device
        self.history = {'train':[], 'valid':[]}
        self.best_score = math.inf
        same_seeds(73)

    def run_epoch(self, epoch, dataset, training, desc=''):
        self.model.train(training)
        shuffle = training

        dataloader = DataLoader(dataset, self.batch_size, shuffle=shuffle)
        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc)
        loss = 0
        acc = Accuracy()
        for i, (imgs, labels) in trange:        # (b, 3, 128, 128), (b, 1)
            pdb()
            labels = labels.view(-1)            # (b,)
            o_labels, batch_loss = self.run_iters(imgs, labels)
            
            if training:
                batch_loss /= self.accum_steps
                batch_loss.backward()
                if (i + 1) % self.accum_steps == 0:
                    self.opt.step()
                    self.opt.zero_grad()
                batch_loss *= self.accum_steps
            
            loss += batch_loss.item()
            acc.update(o_labels.cpu(), labels)

            trange.set_postfix(
                loss=loss / (i+1),
                acc=acc.print_score() 
            )

        if training:
            self.history['train'].append({'loss': loss / len(trange), 'acc': acc.get_score()})
            self.scheduler.step()
        else:
            self.history['valid'].append({'loss': loss / len(trange), 'acc': acc.get_score()})
            if loss < self.best_score:
                self.save_best()
        self.save_hist()
        

    def run_iters(self, imgs, labels):
        imgs = imgs.to(self.device)         # (b, 3, 128, 128)
        labels = labels.to(self.device)     # (b,)
        o_labels = self.model(imgs)         # (b, 11)
        batch_loss = self.criteria(o_labels, labels)
        return o_labels, batch_loss

    def save_best(self, model_name='model'):
        torch.save(self.model.state_dict(), f'{self.arch}/{model_name}.ckpt')

    def save_hist(self):
        with open(f'{self.arch}/history.json', 'w') as f:
            json.dump(self.history, f, indent=4)



