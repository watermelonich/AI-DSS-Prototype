# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from models.reg_models.decisiontreereg.decis_tree import DecisionTreeRegressor
# # Import other models as needed

# def run_decision_tree_regressor():
#     model = DecisionTreeRegressor()
#     data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'models', 'reg_models', 'decisiontreereg', 'data_sample.csv'))
#     model.train(data)
#     predictions = model.predict(data)
#     plt.figure()
#     plt.plot(data['Feature'], data['Target'], 'b-', label='Actual')
#     plt.plot(data['Feature'], predictions, 'r-', label='Predicted')
#     plt.legend()
#     plt.title('Decision Tree Regressor')
#     plt.xlabel('Feature')
#     plt.ylabel('Target')
#     plot_path = os.path.join(os.path.dirname(__file__), '..', 'gui', 'plots', 'decision_tree_regressor.png')
#     plt.savefig(plot_path)
#     return plot_path

# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from models.reg_models.decisiontreereg.decis_tree import DecisionTreeRegressor

# def run_decision_tree_regressor():
#     model = DecisionTreeRegressor()
#     data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'reg_models', 'decisiontreereg', 'data_sample.csv')
#     data = pd.read_csv(data_path)
#     model.train(data)
#     predictions = model.predict(data)
#     plt.figure()
#     plt.plot(data['Feature'], data['Target'], 'b-', label='Actual')
#     plt.plot(data['Feature'], predictions, 'r-', label='Predicted')
#     plt.legend()
#     plt.title('Decision Tree Regressor')
#     plt.xlabel('Feature')
#     plt.ylabel('Target')
#     plot_path = os.path.join(os.path.dirname(__file__), '..', '..', 'gui', 'plots', 'decision_tree_regressor.png')
#     plt.savefig(plot_path)
#     return plot_path
