import torch
import numpy as np
from ipdb import set_trace as pdb


class Accuracy:
    def __init__(self):
        self.n_total = 0
        self.n_correct = 0

    def update(self, predict, target):      # (b, 11), (b,)
        self.n_total += predict.shape[0]
        self.n_correct += torch.sum(predict.argmax(dim=1) == target).item()

    def get_score(self):
        score = float(self.n_correct) / self.n_total
        return score

    def print_score(self):
        score = float(self.n_correct) / self.n_total
        return f'{score:.5f}'
