from sklearn.datasets import load_iris
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import metrics

iris = load_iris()
iris_X = iris.data
iris_y = iris.target

train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size=0.3, random_state=777)

clf = neighbors.KNeighborsClassifier()      # defult k=5
# clf = neighbors.KNeighborsClassifier(n_neighbors=8)
iris_clf = clf.fit(train_X, train_y)

test_y_predicted = iris_clf.predict(test_X)
accuracy = metrics.accuracy_score(test_y, test_y_predicted)

print('Predict: ', test_y_predicted)
print('Target: ', test_y)
print('KNN (k=5) accuracy: {:.2f} %'.format(accuracy*100))