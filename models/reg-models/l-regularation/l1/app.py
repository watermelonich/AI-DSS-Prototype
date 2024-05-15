import csv
from flask import Flask, render_template, jsonify
# import json
import numpy as np
from lasso2_reg import LassoRegressor

app = Flask(__name__, template_folder='templates', static_folder='templates')

#      python3 app.py

# Load data from CSV file
def load_data():
    data = []
    with open('data_sample.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            data.append({'x': int(row[0]), 'y': int(row[1])})
    return data

@app.route("/get_data")
def get_data():
    data = load_data()
    return jsonify(data)

# Add a new route to perform logistic regression and return predictions
@app.route("/l1")
def lasso():
    data = load_data()

    # Load your features (X) and labels (y) from your data
    X = np.array([d['x'] for d in data]).reshape(-1, 1)
    y = np.array([d['y'] for d in data])

    # Initialize and fit the logistic regression model
    lr_model = LassoRegressor()
    lr_model.fit(X, y)

    # Use all the data for testing (replace X_test with your actual test data)
    X_test = X

    # Predict using the logistic regression model
    predictions = lr_model.predict(X_test)

    # Prepare the data for the chart
    chart_data = [{'x': int(x) if not np.isnan(x) else None, 'y': int(y) if not np.isnan(y) else None, 'prediction': int(prediction) if not np.isnan(prediction) else None} for x, y, prediction in zip(X.flatten().tolist(), y.tolist(), predictions)]

    return jsonify(chart_data=chart_data)

@app.route("/")
def home():
    data = load_data()
    return render_template("l1.html", data=data)

if __name__ == '__main__':
    app.run(debug=True)
