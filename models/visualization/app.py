import csv
from flask import Flask, render_template, jsonify
import numpy as np
from clasmodels.logist_reg import LogisticRegression
from clasmodels.svm import SVM
from clasmodels.pca.pca import PCA
from clasmodels.multi_logistreg import MultiLogisticRegression 
import plotly.graph_objs as go
from plotly.offline import plot

app = Flask(__name__, template_folder='templates', static_folder='templates')

# Load data from CSV file
def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            data.append({'x': int(row[0]), 'y': int(row[1])})
    return data

# Route for logistic regression visualization
@app.route("/logistic_regression")
def logistic_regression():
    data = load_data('datasets/data_sample.csv')

    # Load your features (X) and labels (y) from your data
    X = np.array([d['x'] for d in data]).reshape(-1, 1)
    y = np.array([d['y'] for d in data])

    # Initialize and fit the logistic regression model
    lr_model = LogisticRegression()
    lr_model.fit(X, y)

    # Use all the data for testing (replace X_test with your actual test data)
    X_test = X

    # Predict using the logistic regression model
    predictions = lr_model.predict(X_test)

    # Prepare the data for the chart
    chart_data = [{'x': int(x), 'y': int(y), 'prediction': int(prediction)} for x, y, prediction in zip(X.flatten().tolist(), y.tolist(), predictions)]

    return jsonify(chart_data=chart_data)

@app.route("/svm")
def svm_route():
    data = load_data('datasets/data_sample.csv')

    # Load your features (X) from your data
    X = np.array([d['x'] for d in data])
    y = np.array([d['y'] for d in data])

    # Stack X and y to create a 2D array for SVM
    X_stacked = np.column_stack((X, y))

    # Initialize and fit the SVM model
    svm_model = SVM()
    svm_model.fit(X_stacked, y)

    # Predict the labels using the trained SVM model
    y_pred = svm_model.predict(X_stacked)

    # Prepare the data for the chart
    chart_data = [{'x': float(x[0]), 'y': float(x[1]), 'label': int(label), 'prediction': float(pred)} for x, label, pred in zip(X_stacked, y, y_pred)]

    return jsonify(chart_data=chart_data)

@app.route("/multilogistic_regression")
def multilogist():
    data = load_data('datasets/data_sample.csv')

    # Load your features (X) and labels (y) from your data
    X = np.array([d['x'] for d in data]).reshape(-1, 1)
    y = np.array([d['y'] for d in data])

    # Initialize and fit the logistic regression model
    lr_model = MultiLogisticRegression()
    lr_model.fit(X, y)

    # Use all the data for testing (replace X_test with your actual test data)
    X_test = X

    # Predict using the logistic regression model
    predictions = lr_model.predict(X_test)

    # Prepare the data for the chart
    chart_data = [{'x': int(x), 'y': int(y), 'prediction': int(prediction)} for x, y, prediction in zip(X.flatten().tolist(), y.tolist(), predictions)]

    return jsonify(chart_data=chart_data)

@app.route("/pca")
def pca():
    data = load_data('clasmodels/pca/data_sample.csv')

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

    # Calculate explained variance ratio manually
    explained_variance = np.var(X_transformed, axis=0) / np.var(X_stacked, axis=0).sum()
    explained_variance_ratio = explained_variance / explained_variance.sum()

    # Return eigenvectors and explained variance along with chart data
    return jsonify(
        chart_data=chart_data, 
        eigenvectors=pca.components.tolist(), 
        explained_variance=explained_variance_ratio.tolist()
    )

# Combine all visualizations into a single HTML file
@app.route("/")
def home():
    # Generate visualizations for all models
    lr_chart = logistic_regression()
    svm_chart = svm_route()
    pca_chart = pca()
    multi_logist_chart = multilogist()

    # Render the main HTML template with all visualizations
    return render_template("main.html", lr_chart=lr_chart, svm_chart=svm_chart, pca_chart=pca_chart, multi_logist_chart=multi_logist_chart)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
