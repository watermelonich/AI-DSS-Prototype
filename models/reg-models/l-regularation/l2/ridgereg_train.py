from ridge2_reg import RidgeRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load your dataset here or replace this with your data loading code
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the custom Ridge model
ridge = RidgeRegressor(alpha=0.01)  # Adjust alpha for stronger/weaker regularization
ridge.fit(X_train, y_train)

# Predict using the trained model
y_pred = ridge.predict(X_test)

# Calculate and print the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Ridge): {mse}")

# Plot actual vs. predicted values for Ridge
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (Ridge Regression)')
plt.show()
