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

# Save the results of the initial analysis to a file (e.g., predictions and MSE)
with open('initial_analysis_results.txt', 'w') as results_file:
    results_file.write("Predictions:\n")
    results_file.write(str(predictions) + "\n")
    results_file.write(f"Mean Squared Error: {mse}\n")

# Load the saved results
with open('initial_analysis_results.txt', 'r') as results_file:
    lines = results_file.readlines()
    predictions = [float(val) for val in lines[1].strip().split()[1:]]
    mse = float(lines[2].split()[-1])

# Re-evaluate accuracy
new_mse = mse(y_test, predictions)
print(f"Re-evaluated Mean Squared Error: {new_mse}")
