from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from rand_forest_reg import RandomForest
import matplotlib.pyplot as plt

data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

clf = RandomForest(n_trees=20)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc = accuracy(y_test, predictions)
print(f"Accuracy: {acc:.2f}")

# Assuming you've already trained and predicted using clf
predictions = clf.predict(X_test)

# Scatter plot of predicted vs true values
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs. Predictions')
plt.show()
