![Example Image](swit.png)

## AI-DSS
A sample of an AI-DSS (AI-Desicison Support System) a system that combines AI, data analysis, and decision support capabilities to help individuals or organizations make informed and data-driven decisions. It can be applied across various domains and industries, including healthcare, finance, marketing, supply chain management, and more...

# Project Structure

```
MODELS:
├── clas-models
│   ├── logistic-reg
│   │   ├── app.py
│   │   ├── data_sample.csv
│   │   ├── logist_reg.py
│   │   ├── logist_training.py
│   │   ├── node_modules
│   │   │   └── ...
│   │   ├── package-lock.json
│   │   ├── package.json
│   │   └── templates
│   │       ├── logist_reg.html
│   │       └── logistic_regression.js
│   ├── multi-logist-reg
│   │   ├── app.py
│   │   ├── data_sample.csv
│   │   ├── multi_logistreg.py
│   │   ├── templates
│   │   │   └── multi_logistreg.html
│   │   └── train_multilogistreg.py
│   ├── pca
│   │   ├── app.py
│   │   ├── data_sample.csv
│   │   ├── pca.py
│   │   └── templates
│   │       └── pca.html
│   └── svm
│       └── SVM
│           ├── app.py
│           ├── data_sample.csv
│           ├── svm.py
│           └── templates
│               └── graph.html
├── datasets
│   └── database.json
└── reg-models
    ├── decisiontreereg
    │   ├── app.py
    │   ├── data_sample.csv
    │   ├── decis_tree.py
    │   ├── decis_tree_output.txt
    │   ├── decis_tree_training.py
    │   ├── grad-boost-reg
    │   │   ├── app.py
    │   │   ├── gboost.py
    │   │   ├── gboost2.py
    │   │   ├── templates
    │   │   │   └── gboost.html
    │   │   └── training_gboost2.py
    │   └── templates
    │       └── decis_tree.html
    ├── elastic-net-regression
    │   ├── app.py
    │   ├── el_netreg.py
    │   ├── elnet_reg.py
    │   ├── elnet_regtraining.py
    │   └── templates
    │       └── elnet.html
    ├── knn
    │   ├── app.py
    │   ├── knn.py
    │   ├── knn_training.py
    │   └── templates
    │       └── knn.html
    ├── l-regularation
    │   ├── l1
    │   │   ├── app.py
    │   │   ├── data_sample.csv
    │   │   ├── lasso2_reg.py
    │   │   ├── lasso_reg.py
    │   │   ├── lasso_reg_training.py
    │   │   └── templates
    │   │       └── l1.html
    │   └── l2
    │       ├── app.py
    │       ├── data_sample.csv
    │       ├── ridge2_reg.py
    │       ├── ridge_reg.py
    │       ├── ridgereg_train.py
    │       └── templates
    │           └── l2.html
    ├── lin-reg
    │   ├── multiple-lin-reg
    │   │   └── ex1.py
    │   └── simple-lin-reg
    │       ├── app.py
    │       ├── data_sample.csv
    │       ├── dataset_test.py
    │       ├── linear_reg.py
    │       └── templates
    │           └── linear_reg.html
    ├── naive_bayes
    │   ├── app.py
    │   ├── data_sample.csv
    │   ├── nai_bay.py
    │   └── templates
    │       └── nai_bay.html
    ├── random-forest-reg
    │   ├── app.py
    │   ├── data_sample.csv
    │   ├── decis_tree.py
    │   ├── rand_forest_reg.py
    │   ├── rand_forestreg.py
    │   └── templates
    │       └── rand_forestreg.html
    ├── support-vector-reg
    │   ├── svm.py
    │   └── svm_train.py
    └── xgboost-reg
        ├── app.py
        ├── templates
        │   └── xgb.html
        └── xgb.py

NIC
├── datasets
├── src
│   └── sentimanalysis
│       ├── labels.txt
│       ├── qusentan.py
│       ├── reference_data.txt
│       ├── testdata.txt
│       └── testdata_output.txt
└── tokenizer
    ├── data.csv
    ├── input_data.txt
    ├── nlptrain.py
    ├── tokenizer.py
    ├── tokens.txt
    ├── vector.txt
    └── vocab.txt

public
├── dashboard
│   ├── app.js
│   ├── assets
│   │   └── dnic.png
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   ├── storage
│   └── style.css
├── databoard
│   ├── index.html
│   ├── script.js
│   └── style.css
├── package-lock.json
└── package.json

src
└── scripts
```

The "MODELS" directory contains various machine learning models and tools for building AI-based decision support systems. Here's a brief description of each model:

- `clas-models`: This directory contains classification models, including logistic regression, multi-class logistic regression, principal component analysis (PCA), and support vector machines (SVM).

- `reg-models`: This directory contains regression models, such as decision tree regression, gradient-boosted regression, elastic net regression, k-nearest neighbors (KNN) regression, L1 and L2 regularization, linear regression (both simple and multiple), naive Bayes regression, random forest regression, support vector regression, and XGBoost regression.

This is a project still in development and any feedback received would be greatly appreciated

Within the models directory, each machine learning model is accompanied by a corresponding .csv dataset, an HTML file for visualizing data, and a Flask API for interaction.

# License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
