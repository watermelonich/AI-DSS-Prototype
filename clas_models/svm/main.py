import csv
from flask import Flask, render_template, jsonify
# import json
import numpy as np
from svm import SVM

app = Flask(__name__, template_folder='../../../templates', static_folder='../../../static')

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
@app.route("/svm")
def svm_route():
    data = load_data()

    # Load your features (X) from your data
    X = np.array([d['x'] for d in data])
    y = np.array([d['y'] for d in data])

    # Stack X and y to create a 2D array for SVM
    X_stacked = np.column_stack((X, y))

    # Initialize and fit the SVM model
    svm = SVM()
    svm.fit(X_stacked, y)

    # Predict the labels using the trained SVM model
    y_pred = svm.predict(X_stacked)

    # Prepare the data for the chart
    chart_data = [{'x': float(x[0]), 'y': float(x[1]), 'label': int(label), 'prediction': float(pred)} for x, label, pred in zip(X_stacked, y, y_pred)]

    return jsonify(chart_data=chart_data)

@app.route("/")
def home():
    data = load_data()
    return render_template("../../../templates/main.html", data=data)

# Create a new route to serve the JavaScript for SVM
@app.route("/static/svm.js")
def serve_svm_js():
    return app.send_static_file("svm.js")


if __name__ == '__main__':
    app.run(debug=True)
