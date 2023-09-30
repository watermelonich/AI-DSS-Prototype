import csv
from flask import Flask, render_template, jsonify
# import json
# import numpy as np
# from multi_logistreg import MultiLogisticRegression

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

# @app.route("/get_data")
# def get_data():
#     data = load_data()

#     # Load your features (X) and labels (y) from your data
#     X = np.array([d['x'] for d in data]).reshape(-1, 1)  # Adjust the feature extraction as needed
#     y = np.array([d['y'] for d in data])

#     # Initialize and fit the ML model
#     ml_model = MultiLogisticRegression(learning_rate=0.01, num_iterations=1000)
#     ml_model.fit(X, y)

#     # Predict some values
#     sample_predictions = ml_model.predict(X[:10])  # Adjust as needed

#     return jsonify(data=json.dumps(data), sample_predictions=sample_predictions.tolist())

@app.route("/")
def home():
    data = load_data()
    return render_template("l1.html", data=data)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
    print("Server running on http://127.0.0.1:5001/")

