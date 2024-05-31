# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from models.clas_models.logistic_reg.logist_reg import LogisticRegression
# # Import other models as needed

# def run_logistic_regression():
#     model = LogisticRegression()
#     data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'models', 'clas_models', 'logistic_reg', 'data_sample.csv'))
#     model.train(data)
#     predictions = model.predict(data)
#     plt.figure()
#     plt.scatter(data['Feature'], data['Target'], label='Actual')
#     plt.scatter(data['Feature'], predictions, label='Predicted')
#     plt.legend()
#     plt.title('Logistic Regression')
#     plt.xlabel('Feature')
#     plt.ylabel('Target')
#     plot_path = os.path.join(os.path.dirname(__file__), '..', 'gui', 'plots', 'logistic_regression.png')
#     plt.savefig(plot_path)
#     return plot_path

# import sys
# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # Add the parent directory to the Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# from models.clas_models.logistic_reg.logist_reg import LogisticRegression
# # Import other models as needed

# def run_logistic_regression():
#     model = LogisticRegression()
#     data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'clas_models', 'logistic_reg', 'data_sample.csv')
#     data = pd.read_csv(data_path)
#     model.train(data)
#     predictions = model.predict(data)
#     plt.figure()
#     plt.scatter(data['Feature'], data['Target'], label='Actual')
#     plt.scatter(data['Feature'], predictions, label='Predicted')
#     plt.legend()
#     plt.title('Logistic Regression')
#     plt.xlabel('Feature')
#     plt.ylabel('Target')
#     plot_path = os.path.join(os.path.dirname(__file__), '..', 'gui', 'plots', 'logistic_regression.png')
#     plt.savefig(plot_path)
#     return plot_path
