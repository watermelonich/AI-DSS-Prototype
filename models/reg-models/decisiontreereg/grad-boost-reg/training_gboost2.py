from sklearn.model_selection import train_test_split
from gboost2 import SimpleGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)  # For reproducibility

# Generate X with 100 samples and 5 features
X = np.random.rand(100, 5)

# Generate y with a linear relationship plus some noise
true_coefficients = np.array([1, 2, 3, 4, 5])
noise = np.random.normal(0, 1, 100)
y = X.dot(true_coefficients) + noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the custom Gradient Boosting model
custom_gb_reg = SimpleGradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
custom_gb_reg.fit(X_train, y_train)

# Predict using the custom trained model
y_pred_custom = custom_gb_reg.predict(X_test)

# Calculate and print the Mean Squared Error (MSE) for the custom model
mse_custom = mean_squared_error(y_test, y_pred_custom)
print(f"Mean Squared Error (Custom Gradient Boosting): {mse_custom}")

# Plot actual vs. predicted values for the custom model
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_custom, c='blue', label='Custom GB Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2, label='Perfect Predictions')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (Custom Gradient Boosting)')
plt.legend()
plt.show()
