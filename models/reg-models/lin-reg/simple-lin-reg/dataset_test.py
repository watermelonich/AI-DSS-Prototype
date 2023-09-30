import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linear_reg import LinearRegression

x, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize= (8,6))
plt.scatter(x[:, 0], y, color = "b", marker = "o", s = 30)
plt. show()

reg = LinearRegression(lr = 0.01)
reg.fit(x_train, y_train)
predictions= reg.predict(x_test)

def mse(y_test, predictions):
    return np.mean((y_test - predictions)**2)

mse = mse(y_test, predictions)
print(mse)

y_pred_line = reg.predict(x)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize= (8,6))
m1 = plt.scatter(x_train, y_train, color=cmap (0.9), s=10)
m2 = plt.scatter(x_test, y_test, color=cmap(0.5), s=10)
plt.plot(x, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()
