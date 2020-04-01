from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from ipdb import set_trace as pdb
from sklearn import metrics

iris = load_iris()
# iris.keys(): dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
# iris['data'].shape = (150, 4)
# iris['target'].shape = (150,)
# iris['target_names'].shape = (3,)   # (Setosa, Virginica, Versicolour)
# iris['feature_names'] = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

iris_X = iris.data
iris_y = iris.target

train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size = 0.3)

clf = tree.DecisionTreeClassifier()
iris_clf = clf.fit(train_X, train_y)

test_y_predicted = iris_clf.predict(test_X)
accuracy = metrics.accuracy_score(test_y, test_y_predicted)

print('Predict: ', test_y_predicted)
print('Target: ', test_y)
print('Decision tree accuracy: {:.2f} %'.format(accuracy*100))