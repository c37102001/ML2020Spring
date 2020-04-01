import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble, preprocessing, metrics
from ipdb import set_trace as pdb

seed = 72

titanic_train = pd.read_csv('titanic_train.csv')

age_median = np.nanmedian(titanic_train["Age"]) # calucate the median of array ignoring the NaN value
new_Age = np.where(titanic_train["Age"].isnull(), age_median, titanic_train["Age"])
titanic_train["Age"] = new_Age

label_encoder = preprocessing.LabelEncoder()
encoded_Sex = label_encoder.fit_transform(titanic_train["Sex"])

titanic_X = pd.DataFrame([titanic_train["Pclass"],
                         encoded_Sex,
                         titanic_train["Age"]
]).T
titanic_y = titanic_train["Survived"]
train_X, test_X, train_y, test_y = train_test_split(titanic_X, titanic_y, test_size=0.3, random_state=seed)

forest = ensemble.RandomForestClassifier(n_estimators=100, random_state=seed)
forest_fit = forest.fit(train_X, train_y)

test_y_predicted = forest.predict(test_X)

accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print('Random Forest accuracy: {:.2f} %'.format(accuracy*100))


# Note
# np.where:   https://www.cnblogs.com/massquantity/p/8908859.html
# label_encoder:  https://medium.com/@PatHuang/%E5%88%9D%E5%AD%B8python%E6%89%8B%E8%A8%98-3-%E8%B3%87%E6%96%99%E5%89%8D%E8%99%95%E7%90%86-label-encoding-one-hot-encoding-85c983d63f87
