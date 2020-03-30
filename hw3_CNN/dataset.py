import torch
from torch.utils.data import Dataset
from ipdb import set_trace as pdb


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):       # (9866, 128, 128, 3), (9866,)
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        Y = [self.y[index]] if self.y is not None else []

        if self.transform is not None:
            X = self.transform(X)
        Y = torch.LongTensor(Y)
        
        return X, Y
    