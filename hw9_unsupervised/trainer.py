import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from model import AE
from utils import same_seeds


class Trainer:
    def __init__(self, lr=1e-5, batch_size=64):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.batch_size = batch_size
        self.model = AE()
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        same_seeds(0)

    def run_epoch(self, epoch, img_dataset):
        self.model.train()

        img_dataloader = DataLoader(img_dataset, batch_size=self.batch_size, shuffle=True)

        for data in img_dataloader:
            img = data.to(self.device)

            encoded, output = self.model(img)
            loss = self.criterion(output, img)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if (epoch+1) % 10 == 0:
                torch.save(self.model.state_dict(), './checkpoints/checkpoint_{}.pth'.format(epoch+1))
            
        print('epoch [{}], loss:{:.5f}'.format(epoch+1, loss.data))
        torch.save(self.model.state_dict(), './checkpoints/last_checkpoint.pth')