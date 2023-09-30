from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load your dataset here or replace this with your data loading code
# Example:
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Ridge model
ridge = Ridge(alpha=0.01)  # Adjust alpha for stronger/weaker regularization
ridge.fit(X_train, y_train)

# Predict using the trained model
y_pred = ridge.predict(X_test)

# Calculate and print the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (Ridge): {mse}")
