# Test File

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate some example data (replace this with your actual data)
import numpy as np
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = X.dot(np.array([1, 2, 3, 4, 5])) + np.random.normal(0, 1, 100)  # Linear relationship with noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the ElasticNet model
alpha = 0.5  # Regularization strength for both L1 and L2
l1_ratio = 0.5  # Ratio of L1 penalty (0 for Ridge, 1 for Lasso)
enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
enet.fit(X_train, y_train)

# Predict using the trained model
y_pred = enet.predict(X_test)

# Calculate and print the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs. Predictions')
plt.show()
