import numpy as np 
from sklearn import datasets 
from sklearn. model_selection import train_test_split 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from knn import KNN

cmap = ListedColormap (['#FF0000', '#00FF00', '#0000FF'])
iris = datasets.load_iris()
x, y = iris.data, iris.target
x_train, x_test, y_train, _test = train_test_split(x, y, test_size=0.2, random_state=1234)

plt.figure ()
plt.scatter(x[:,2],x[:,3], c=y, cmap=cmap, edgecolor='k', s=20) 
plt.show()

clf = KNN(k=5)
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
print(predictions)
