import numpy as np
import matplotlib.pyplot as plt
from utils import *
from ipdb import set_trace as pdb


np.random.seed(0)
X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'
output_fpath = './output_{}.csv'

# Parse csv files to numpy array
print('[Parsing CSV Files]')
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)  # (54256, 510)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)   # (54256,)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)   # (27622, 510)


# Normalize training and testing data
X_train, X_mean, X_std = normalize(X_train, train = True)
X_test, _, _= normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)
train_size = X_train.shape[0]       # 54256
test_size = X_test.shape[0]         # 27622
data_dim = X_train.shape[1]         # 510

# Compute in-class mean
print('[Computing Cov matrix]')
X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])       # (43105, 510)
X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])       # (11151, 510)
mean_0 = np.mean(X_train_0, axis = 0)   # (510,)
mean_1 = np.mean(X_train_1, axis = 0)   # (510,)

# Compute in-class covariance
cov_0 = np.zeros((data_dim, data_dim))
cov_1 = np.zeros((data_dim, data_dim))
for x in X_train_0:
    cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
for x in X_train_1:
    cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

# Shared covariance is taken as a weighted average of individual in-class covariance.
cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])


# Compute inverse of covariance matrix.
# Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
# Via SVD decomposition, one can get matrix inverse efficiently and accurately.
u, s, v = np.linalg.svd(cov, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)

# Directly compute weights and bias
print('[Computimg weights and bias]')
w = np.dot(inv, mean_0 - mean_1)
b =  (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1))\
    + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0]) 

# Compute accuracy on training set
# if P(C0|x) = 1, then pred = 1-1=0, elif P(C0|x) = 0, pred = 1-0=1
Y_train_pred = 1 - predict(X_train, w, b)
print('Training accuracy: {}'.format(accuracy(Y_train_pred, Y_train)))


# Predict testing labels
print('[Predicting]')
predictions = 1 - predict(X_test, w, b)
with open(output_fpath.format('generative'), 'w') as f:
    f.write('id,label\n')
    for i, label in enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

# Print out the most significant weights
ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])