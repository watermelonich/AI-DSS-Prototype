from lasso2_reg import LassoRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 5)
true_coefficients = np.array([1, 2, 3, 4, 5])
noise = np.random.normal(0, 1, 100)
y = X.dot(true_coefficients) + noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the custom Lasso model
lasso = LassoRegressor(alpha=0.01)  # Adjust alpha for stronger/weaker regularization
lasso.fit(X_train, y_train)

# Predict using the trained model
y_pred = lasso.predict(X_test)

# Plot actual vs. predicted values for Lasso
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (Lasso Regression)')
plt.show()
