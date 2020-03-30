import seaborn as sns
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt 
from ipdb import set_trace as pdb

sns.set()
y_true = ["cat", "dog", "cat", "cat", "dog", "rebit"] 
y_pred = ["dog", "dog", "rebit", "cat", "dog", "cat"] 
C2 = confusion_matrix(y_true, y_pred, labels=["dog", "rebit", "cat"]) 


plt.figure(figsize=(10, 10))
ax = sns.heatmap(
    C2,
    xticklabels=["dog", "rebit", "cat"],
    yticklabels=["dog", "rebit", "cat"],
    cmap=sns.diverging_palette(20, 220, n=200),
    vmin=-1, vmax=1, center=0,
    square=True,
    annot=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

plt.savefig('heat_map')