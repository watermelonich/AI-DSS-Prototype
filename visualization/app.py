import csv
from flask import Flask, render_template, jsonify
import numpy as np

# Import classification models
from clasmodels.logist_reg import LogisticRegression
from clasmodels.svm import SVM
from clasmodels.pca.pca import PCA
from clasmodels.multi_logistreg import MultiLogisticRegression 
import plotly.graph_objs as go

# Import regression models
from regmodels.decistree.decis_tree import DecisionTree
from regmodels.elnetreg.elnet_reg import ElasticNetRegressor
from regmodels.knn.knn import KNN
from regmodels.l1.lasso2_reg import LassoRegressor
from regmodels.linreg.linear_reg import LinearRegression
from regmodels.naibe_bayes.nai_bay import NaiveBayes
from regmodels.randomforest.rand_forest_reg import RandomForest
from regmodels.l2.ridge2_reg import RidgeRegressor
from regmodels.xgb.xgb import SimpleGradientBoostingRegressor

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


# ------------------- Regression models

@app.route("/decision_tree")
def decision_tree():
    data = load_data('regmodels/decistree/data_sample.csv')

    # Load your features (X) and labels (y) from your data
    X = np.array([d['x'] for d in data]).reshape(-1, 1)
    y = np.array([d['y'] for d in data])

    # Initialize and fit the logistic regression model
    lr_model = DecisionTree()
    lr_model.fit(X, y)

    # Use all the data for testing (replace X_test with your actual test data)
    X_test = X

    # Predict using the logistic regression model
    predictions = lr_model.predict(X_test)

    # Prepare the data for the chart
    chart_data = [{'x': int(x), 'y': int(y), 'prediction': int(prediction)} for x, y, prediction in zip(X.flatten().tolist(), y.tolist(), predictions)]

    return jsonify(chart_data=chart_data)

@app.route("/elastic_netreg")
def elastic_netreg():
    data = load_data('regmodels/elnetreg/data_sample.csv')

    # Load your features (X) and labels (y) from your data
    X = np.array([d['x'] for d in data]).reshape(-1, 1)
    y = np.array([d['y'] for d in data])

    # Initialize and fit the logistic regression model
    lr_model = ElasticNetRegressor()
    lr_model.fit(X, y)

    # Use all the data for testing (replace X_test with your actual test data)
    X_test = X

    # Predict using the logistic regression model
    predictions = lr_model.predict(X_test)

    # Prepare the data for the chart
    chart_data = [{'x': int(x) if not np.isnan(x) else None, 'y': int(y) if not np.isnan(y) else None, 'prediction': int(prediction) if not np.isnan(prediction) else None} for x, y, prediction in zip(X.flatten().tolist(), y.tolist(), predictions)]

    return jsonify(chart_data=chart_data)

@app.route("/knn")
def knn():
    data = load_data('regmodels/knn/data_sample.csv')

    # Load your features (X) and labels (y) from your data
    X = np.array([d['x'] for d in data]).reshape(-1, 1)
    y = np.array([d['y'] for d in data])

    # Initialize and fit the logistic regression model
    lr_model = KNN()
    lr_model.fit(X, y)

    # Use all the data for testing (replace X_test with your actual test data)
    X_test = X

    # Predict using the logistic regression model
    predictions = lr_model.predict(X_test)

    # Prepare the data for the chart
    chart_data = [{'x': int(x) if isinstance(x, (int, float)) else None, 
                   'y': int(y) if isinstance(y, (int, float)) else None, 
                   'prediction': int(prediction) if isinstance(prediction, (int, float)) else None} 
                   for x, y, prediction in zip(X.flatten().tolist(), y.tolist(), predictions)]

    return jsonify(chart_data=chart_data)

@app.route("/l1")
def lasso_reg():
    data = load_data('regmodels/l1/data_sample.csv')

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

@app.route("/l2")
def ridge_reg():
    data = load_data('regmodels/l2/data_sample.csv')

        # Load your features (X) and labels (y) from your data
    X = np.array([d['x'] for d in data]).reshape(-1, 1)
    y = np.array([d['y'] for d in data])

    # Initialize and fit the logistic regression model
    lr_model = RidgeRegressor()
    lr_model.fit(X, y)

    # Use all the data for testing (replace X_test with your actual test data)
    X_test = X

    # Predict using the logistic regression model
    predictions = lr_model.predict(X_test)

    # Prepare the data for the chart
    chart_data = [{'x': int(x) if not np.isnan(x) else None, 'y': int(y) if not np.isnan(y) else None, 'prediction': int(prediction) if not np.isnan(prediction) else None} for x, y, prediction in zip(X.flatten().tolist(), y.tolist(), predictions)]

    return jsonify(chart_data=chart_data)

@app.route("/linear_regression")
def linear_reg():
    data = load_data('regmodels/linreg/data_sample.csv')
    
    # Load your features (X) and labels (y) from your data
    X = np.array([d['x'] for d in data]).reshape(-1, 1)
    y = np.array([d['y'] for d in data])

    # Initialize and fit the logistic regression model
    lr_model = LinearRegression()
    lr_model.fit(X, y)

    # Use all the data for testing (replace X_test with your actual test data)
    X_test = X

    # Predict using the logistic regression model
    predictions = lr_model.predict(X_test)

    # Prepare the data for the chart
    chart_data = [{'x': int(x) if not np.isnan(x) else None, 'y': int(y) if not np.isnan(y) else None, 'prediction': int(prediction) if not np.isnan(prediction) else None} for x, y, prediction in zip(X.flatten().tolist(), y.tolist(), predictions)]

    return jsonify(chart_data=chart_data)

@app.route("/naive_bayes")
def naive_bayes():
    data = load_data('regmodels/naibe_bayes/data_sample.csv')

    # Load your features (X) and labels (y) from your data
    X = np.array([d['x'] for d in data]).reshape(-1, 1)
    y = np.array([d['y'] for d in data])

    # Initialize and fit the logistic regression model
    lr_model = NaiveBayes()
    lr_model.fit(X, y)

    # Use all the data for testing (replace X_test with your actual test data)
    X_test = X

    # Predict using the logistic regression model
    predictions = lr_model.predict(X_test)

    # Prepare the data for the chart
    chart_data = [{'x': int(x), 'y': int(y), 'prediction': int(prediction)} for x, y, prediction in zip(X.flatten().tolist(), y.tolist(), predictions)]

    return jsonify(chart_data=chart_data)

# @app.route("/random_forest")
# def random_forest():
#     data = load_data('regmodels/randomforest/data_sample.csv')

#     # Load your features (X) and labels (y) from your data
#     X = np.array([d['x'] for d in data]).reshape(-1, 1)
#     y = np.array([d['y'] for d in data])

#     # Initialize and fit the logistic regression model
#     lr_model = RandomForest()
#     lr_model.fit(X, y)

#     # Use all the data for testing (replace X_test with your actual test data)
#     X_test = X

#     # Predict using the logistic regression model
#     predictions = lr_model.predict(X_test)

#     # Prepare the data for the chart
#     chart_data = [{'x': int(x), 'y': int(y), 'prediction': int(prediction)} for x, y, prediction in zip(X.flatten().tolist(), y.tolist(), predictions)]

#     return jsonify(chart_data=chart_data)

@app.route("/xgb")
def xgb():
    data = load_data('regmodels/xgb/data_sample.csv')

    # Load your features (X) and labels (y) from your data
    X = np.array([d['x'] for d in data]).reshape(-1, 1)
    y = np.array([d['y'] for d in data])

    # Initialize and fit the logistic regression model
    lr_model = SimpleGradientBoostingRegressor()
    lr_model.fit(X, y)

    # Use all the data for testing (replace X_test with your actual test data)
    X_test = X

    # Predict using the logistic regression model
    predictions = lr_model.predict(X_test)

    # Prepare the data for the chart
    chart_data = [{'x': int(x), 'y': int(y), 'prediction': int(prediction)} for x, y, prediction in zip(X.flatten().tolist(), y.tolist(), predictions)]

    return jsonify(chart_data=chart_data)

# Combine all visualizations into a single HTML file
@app.route("/")
def home():

    # Generate visualizations for all models

    # Classification Models
    lr_chart = logistic_regression()
    svm_chart = svm_route()
    pca_chart = pca()
    multi_logist_chart = multilogist()

    # Regression Models
    dc_chart = decision_tree()
    elnet_chart = elastic_netreg()
    knn_chart = knn()
    lasso_chart = lasso_reg()
    ridge_chart = ridge_reg()
    linear_chart = linear_reg()
    naibay_chart = naive_bayes()
    # randomforest_chart = random_forest()
    xgb_chart = xgb()


    # Render the main HTML template with all visualizations
    return render_template("main.html", lr_chart=lr_chart, svm_chart=svm_chart, pca_chart=pca_chart, multi_logist_chart=multi_logist_chart, dc_chart=dc_chart, elnet_chart=elnet_chart, knn_chart=knn_chart, lasso_chart=lasso_chart, ridge_chart=ridge_chart, linear_chart=linear_chart, naibay_chart=naibay_chart, xgb_chart=xgb_chart)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
