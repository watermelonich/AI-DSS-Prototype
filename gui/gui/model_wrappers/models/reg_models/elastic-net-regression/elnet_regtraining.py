import numpy as np
from sklearn.model_selection import train_test_split
from elnet_reg import ElasticNetRegressor
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 5)
true_coefficients = np.array([1, 2, 3, 4, 5])
noise = np.random.normal(0, 1, 100)
y = X.dot(true_coefficients) + noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the custom Elastic Net model
enet = ElasticNetRegressor(alpha=0.5, l1_ratio=0.5, learning_rate=0.01, n_iterations=1000)
enet.fit(X_train, y_train)

# Predict using the trained model
y_pred = enet.predict(X_test)

# Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='blue', label='Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Perfect Predictions')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()

# Plot coefficients
plt.figure(figsize=(8, 6))
plt.bar(np.arange(len(enet.coef_)), enet.coef_)
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Elastic Net Coefficients')
plt.show()
