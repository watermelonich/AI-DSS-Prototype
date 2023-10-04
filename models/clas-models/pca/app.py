import csv
from flask import Flask, render_template, jsonify
# import json
import numpy as np
from pca import PCA

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
@app.route("/pca")
def pca_route():
    data = load_data()

    # Load your features (X) from your data
    X = np.array([d['x'] for d in data])
    y = np.array([d['y'] for d in data])

    # Stack X and y to create a 2D array for PCA
    X_stacked = np.column_stack((X, y))

    # Initialize and fit the PCA model
    pca = PCA(n_components=2)
    pca.fit(X_stacked)

    # Perform PCA transformation on the data
    X_transformed = pca.transform(X_stacked)

    # Prepare the data for the chart
    chart_data = [{'x': float(x[0]), 'y': float(x[1])} for x in X_transformed]

    return jsonify(chart_data=chart_data)

@app.route("/")
def home():
    data = load_data()
    return render_template("pca.html", data=data)

if __name__ == '__main__':
    app.run(debug=True)
