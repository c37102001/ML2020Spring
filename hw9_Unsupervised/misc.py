import json
from matplotlib import pyplot as plt


def plot_loss(arch):
    with open(f'{arch}/history.json', 'r') as f:
        history = json.loads(f.read())

        loss = history['loss']
        plt.figure(figsize=(7, 5))
        plt.title('Loss')
        plt.plot(loss)
        plt.grid(True)
        plt.savefig(f'{arch}/Loss.png')
    print('done!')
        