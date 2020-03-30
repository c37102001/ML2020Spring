import sys
import math
import pandas as pd
import numpy as np
from ipdb import set_trace as pdb
import csv

# Preprocessing
data = pd.read_csv('data/train.csv', encoding = 'big5')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()      # (4320, 24), want to make it to 12month*(18fts, 20days*24hours)= 12*(18,480)

# Extract features 1
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day*24: (day+1)*24] = raw_data[18*(20*month+day): 18*(20*month+day+1), :]
    month_data[month] = sample

# Extract feature 2
x = np.empty([12 * 471, 18 * 9], dtype=float)   # (5652, 162)
y = np.empty([12 * 471, 1], dtype=float)        # (5652, 1)

# Month_data: (12, (18, 480))
for month in range(12):
    for day in range(20):
        for hour in range(24):
            start_hour = day * 24 + hour
            if start_hour + 1 > 471:
                continue 
            train_x = month_data[month][:, start_hour: start_hour+9].reshape(1, -1)     # (18, 9) -> (18 * 9)
            train_y = month_data[month][9, start_hour+9]                                # (1)

            x[month*471 + start_hour, :] = train_x
            y[month*471 + start_hour, :] = train_y

# Normalize x
mean_x = np.mean(x, axis=0)     # (162)
std_x = np.std(x, axis=0)       # (162)
for i in range(18*9):
    if not std_x[i] == 0:
        x[:, i] = (x[:, i] - mean_x[i]) / std_x[i]

# Split train valid
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]

# Train
dim = 18 * 9 + 1
x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)
w = np.zeros([dim, 1])
adagrad = np.zeros([dim, 1])
lr = 10
iter_time = 10000
eps = 1e-10

for iter in range(iter_time):
    loss = np.sqrt(np.sum(np.power(y - np.dot(x, w), 2)) / (12*471))
    gradient = np.dot(np.transpose(x), (-2 * (y - np.dot(x, w))))
    adagrad += gradient**2
    w = w - lr * gradient / np.sqrt(adagrad + eps)
    if iter % (iter_time//10) == 0:
        print(str(iter) + ":" + str(loss))
np.save('weight.npy', w)


# Testing Preprocessing
testdata = pd.read_csv('data/test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]        # (4320=240days * 18fts, 9hours)
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()

test_x = np.empty([240, 18*9], dtype=float)   # (240days, 18fts * 9hours) = (240, 162)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)   # (18, 9) -> (1, 18*9)

for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

# Predict
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)

sample = pd.read_csv('data/sample_submission.csv')
submit = dict()
submit['id'] = list(sample.id.values)
prediction = [p.item() for p in ans_y]
submit['value'] = list(prediction)
df = pd.DataFrame.from_dict(submit)
df.to_csv('prediction.csv', index=False)
print('Finished!')