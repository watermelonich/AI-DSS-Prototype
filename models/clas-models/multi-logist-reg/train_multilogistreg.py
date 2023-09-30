import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from multi_logistreg import MultiLogisticRegression

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultiLogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print the accuracy
accuracy = np.mean(y_pred == y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")

import matplotlib.pyplot as plt

y_pred = model.predict(X_test)

# Plot actual vs. predicted values
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()
