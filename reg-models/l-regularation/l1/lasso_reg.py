# Experimental file


from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Generate some example data (replace this with your actual data)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = X.dot(np.array([1, 2, 3, 4, 5])) + np.random.normal(0, 1, 100)  # Linear relationship with noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Lasso model
lasso = Lasso(alpha=0.01)  # Adjust alpha for stronger/weaker regularization
lasso.fit(X_train, y_train)

# Predict using the trained model
y_pred = lasso.predict(X_test)

# Calculate and print the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Lasso): {mse}")
