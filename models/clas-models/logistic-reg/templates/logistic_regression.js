function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

class LogisticRegression {
    constructor(lr = 0.001, n_iters = 1000) {
        this.lr = lr;
        this.n_iters = n_iters;
        this.weights = null;
        this.bias = null;
    }

    fit(X, y) {
        const n_samples = X.length;
        const n_features = X[0].length;
        this.weights = Array(n_features).fill(0);
        this.bias = 0;

        for (let iter = 0; iter < this.n_iters; iter++) {
            let linear_pred = X.map(sample => sample.reduce((acc, feature, idx) => acc + feature * this.weights[idx], 0) + this.bias);
            let predictions = linear_pred.map(val => sigmoid(val));

            let dw = Array(n_features).fill(0);
            for (let i = 0; i < n_samples; i++) {
                for (let j = 0; j < n_features; j++) {
                    dw[j] += (1 / n_samples) * (predictions[i] - y[i]) * X[i][j];
                }
            }

            let db = (1 / n_samples) * predictions.reduce((acc, val, idx) => acc + (val - y[idx]), 0);

            for (let i = 0; i < n_features; i++) {
                this.weights[i] -= this.lr * dw[i];
            }
            this.bias -= this.lr * db;
        }
    }

    predict(X) {
        let linear_pred = X.map(sample => sample.reduce((acc, feature, idx) => acc + feature * this.weights[idx], 0) + this.bias);
        let y_pred = linear_pred.map(val => sigmoid(val));
        let class_pred = y_pred.map(val => val <= 0.5 ? 0 : 1);

        return class_pred;
    }
}
