import math
import os
import json
import torch
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from metrics import Accuracy
from tqdm import tqdm
from ipdb import set_trace as pdb
from utils import same_seeds
from model import *


class Trainer:
    def __init__(self, arch, device):
        self.arch = arch
        if not os.path.exists(f'{arch}/ckpts'):
            os.makedirs(f'{arch}/ckpts')

        self.feature_extractor = FeatureExtractor().to(device)
        self.label_predictor = LabelPredictor().to(device)
        self.domain_classifier = DomainClassifier().to(device)

        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCEWithLogitsLoss()

        self.optimizer_F = optim.Adam(self.feature_extractor.parameters())
        self.optimizer_C = optim.Adam(self.label_predictor.parameters())
        self.optimizer_D = optim.Adam(self.domain_classifier.parameters())
        
        self.device = device
        self.history = {'d_loss':[], 'f_loss':[], 'acc':[]}
        self.best_score = math.inf
        same_seeds(73)

    def run_epoch(self, epoch, source_dataloader, target_dataloader, lamb):

        trange = tqdm(zip(source_dataloader, target_dataloader), total=len(source_dataloader), desc=f'[epoch {epoch}]')
        
        total_D_loss, total_F_loss = 0.0, 0.0
        acc = Accuracy()
        for i, ((source_data, source_label), (target_data, _)) in enumerate(trange):
            source_data = source_data.to(self.device)
            source_label = source_label.to(self.device)
            target_data = target_data.to(self.device)
            
            # =========== Preprocess =================
            # mean/var of source and target datas are different, so we put them together for properly batch_norm 
            mixed_data = torch.cat([source_data, target_data], dim=0)
            domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).to(self.device)
            domain_label[:source_data.shape[0]] = 1     # source data label=1, target data lebel=0
            feature = self.feature_extractor(mixed_data)


            # =========== Step 1 : Train Domain Classifier (fix feature extractor by feature.detach()) =================
            domain_logits = self.domain_classifier(feature.detach())
            loss = self.domain_criterion(domain_logits, domain_label)
            total_D_loss+= loss.item()
            loss.backward()
            self.optimizer_D.step()


            # =========== Step 2: Train Feature Extractor and Label Predictor =================
            class_logits = self.label_predictor(feature[:source_data.shape[0]])
            domain_logits = self.domain_classifier(feature)
            loss = self.class_criterion(class_logits, source_label) - lamb * self.domain_criterion(domain_logits, domain_label)
            total_F_loss+= loss.item()
            loss.backward()
            self.optimizer_F.step()
            self.optimizer_C.step()

            self.optimizer_D.zero_grad()
            self.optimizer_F.zero_grad()
            self.optimizer_C.zero_grad()

            acc.update(class_logits, source_label)

            trange.set_postfix(
                D_loss=total_D_loss / (i+1),
                F_loss=total_F_loss / (i+1),
                acc=acc.print_score() 
            )

        self.history['d_loss'].append(total_D_loss / len(trange))
        self.history['f_loss'].append(total_F_loss / len(trange))
        self.history['acc'].append(acc.get_score())
        self.save_hist()

        self.save_model()
        

    def save_model(self):
        torch.save(self.feature_extractor.state_dict(), f'{self.arch}/ckpts/extractor.ckpt')
        torch.save(self.label_predictor.state_dict(), f'{self.arch}/ckpts/predictor.ckpt')

    def save_hist(self):
        with open(f'{self.arch}/history.json', 'w') as f:
            json.dump(self.history, f, indent=4)



