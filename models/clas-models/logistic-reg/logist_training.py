import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from logist_reg import LogisticRegression # model imported for training
from sklearn.metrics import confusion_matrix
import seaborn as sns

bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

clf = LogisticRegression(lr=0.01)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

def accuracy(y_pred, y_test) :
    return np.sum(y_pred==y_test) /len(y_test)

acc = accuracy (y_pred, y_test)
print (acc)

# Scatter plot
plt.scatter(y_test, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("True Values vs. Predictions")
plt.show()

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
