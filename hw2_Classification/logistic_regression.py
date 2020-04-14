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
X_test, _, _= normalize(X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)
    

# Split data into training set and development set
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)


train_size = X_train.shape[0]       # 48830
dev_size = X_dev.shape[0]           # 5426
test_size = X_test.shape[0]         # 27622
data_dim = X_train.shape[1]         # 510
print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))


# Training
print('[Training]')
# Zero initialization for weights ans bias
w = np.zeros((data_dim,)) 
b = np.zeros((1,))

# Some parameters for training    
max_iter = 10
batch_size = 8
learning_rate = 0.2

# Keep the loss and accuracy at every iteration for plotting
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

# Calcuate the number of parameter updates
step = 1

# Iterative training
for epoch in range(max_iter):
    # Random shuffle at the begging of each epoch
    X_train, Y_train = shuffle(X_train, Y_train)
    
    # Mini-batch training
    for idx in range(train_size // batch_size):
        X = X_train[idx*batch_size: (idx+1)*batch_size]
        Y = Y_train[idx*batch_size: (idx+1)*batch_size]

        # Compute the gradient
        w_grad, b_grad = gradient(X, Y, w, b)
            
        # gradient descent update
        # learning rate decay with time
        w = w - learning_rate/np.sqrt(step) * w_grad
        b = b - learning_rate/np.sqrt(step) * b_grad

        step = step + 1
            
    # Compute loss and accuracy of training set and development set
    y_train_pred = logistic_regression(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(accuracy(Y_train_pred, Y_train))
    train_loss.append(cross_entropy_loss(y_train_pred, Y_train) / train_size)

    y_dev_pred = logistic_regression(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc.append(accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)

print('Training loss: {}'.format(train_loss[-1]))
print('Development loss: {}'.format(dev_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))


# Plot
# Loss curve
plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png')
plt.clf()

# Accuracy curve
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')


# Predict testing labels
predictions = predict(X_test, w, b)
with open(output_fpath.format('logistic'), 'w') as f:
    f.write('id,label\n')
    for i, label in enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

# Print out the most significant weights
ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(f'[{features[i]}]', w[i])