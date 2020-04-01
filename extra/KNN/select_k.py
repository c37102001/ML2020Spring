from sklearn.datasets import load_iris
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
iris_X = iris.data
iris_y = iris.target

train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size=0.3, random_state=777)

range = np.arange(1, round(0.2 * train_X.shape[0]) + 1)
accuracies = []

for i in range:
    clf = neighbors.KNeighborsClassifier(n_neighbors = i)
    iris_clf = clf.fit(train_X, train_y)
    test_y_predicted = iris_clf.predict(test_X)
    accuracy = metrics.accuracy_score(test_y, test_y_predicted)
    accuracies.append(accuracy)

# visualize
plt.scatter(range, accuracies)
plt.savefig('k-accuracy.png')
appr_k = accuracies.index(max(accuracies)) + 1
print(appr_k)