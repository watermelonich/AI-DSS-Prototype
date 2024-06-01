# Experimental file


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate some example data (replace this with your actual data)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = X.dot(np.array([1, 2, 3, 4, 5])) + np.random.normal(0, 1, 100)  # Linear relationship with noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict using the trained model
y_pred = reg.predict(X_test)

# Calculate and print the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Multiple Linear Regression): {mse}")

import matplotlib.pyplot as plt

# Assuming you have already trained and predicted using reg
y_pred = reg.predict(X_test)

# Plot actual vs. predicted values
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()
