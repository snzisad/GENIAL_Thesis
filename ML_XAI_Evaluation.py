import numpy as np
import pandas as pd
from datetime import datetime
import shap
import lime
from scipy.stats import linregress
import lime.lime_tabular
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import AdaBoostRegressor 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neural_network import MLPRegressor

def get_prediction_from_model(dataset):
    pred = selected_model.predict(dataset)
    # brb = BRBJOPS(param, model_type, num_ref_value=numRefVal, isDebug = False)
    # for x in data:
    #     aqi = brb.runBRB(x)
    #     # print(brb.PM2_5_ref_val)
    #     # print(brb.PM10_ref_val)
    #     # print(brb.NH3_ref_val)
    #     # print(brb.utilityScore)
    #     # return
    #     pred.append(aqi)

    return np.array(pred)

def get_lime_values(dataset):
    lime_values = []
    single_lime_value = []
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(training_set, feature_names=features, class_names=['AQI'], verbose=True, mode='regression')
    

    for test_data in dataset:

      lime_exp = lime_explainer.explain_instance(test_data, get_prediction_from_model, num_features=len(features), labels=(1,))

      coeffs = lime_exp.local_exp[0]
      single_lime_value = [0] * len(features)
      for c in coeffs:
          single_lime_value[c[0]] += c[1]

      lime_values.append(single_lime_value)

    return np.array(lime_values)

def XAIEvaluation(model, model_name):
    global selected_model
    # Create a shap explainer
    selected_model = model
    shap_explainer = shap.KernelExplainer(get_prediction_from_model, training_set)
    
    X_test = dataset_x[num_trained_item:]
    y_test = dataset_y[num_trained_item:]
    
    
    
    num_items = 200
    subset_A = X_test[:num_items]
    subset_B = X_test[num_items:2*num_items]
    
    shap_values_A = shap_explainer.shap_values(subset_A)
    shap_values_B = shap_explainer.shap_values(subset_B)
    
    shap_values = np.array([*shap_values_A, *shap_values_B])
    
    
    """### LIME Model"""
    
    
    num_items = 200
    subset_A = X_test[:num_items]
    subset_B = X_test[num_items:2*num_items]
    
    lime_values_A = get_lime_values(subset_A)
    lime_values_B = get_lime_values(subset_B)
    
    lime_values = np.array([*lime_values_A, *lime_values_B])
    
    # Calculate feature coverage
    shap_coverages = []
    
    nonzero_counts = np.sum(shap_values != 0, axis=0)
    nonzero_percents = nonzero_counts / shap_values.shape[0]
    shap_coverages.extend(nonzero_percents)
    
    
    
    # Calculate feature coverage
    lime_coverages = []
    
    nonzero_counts = np.sum(lime_values != 0, axis=0)
    nonzero_percents = nonzero_counts / lime_values.shape[0]
    lime_coverages.extend(nonzero_percents)
    
    
    
    # calculate the average absolute SHAP value for each feature
    avg_shap = np.abs(shap_values).mean(axis=0)
    
    
    """### LIME Model"""
    
    # calculate the average absolute lime value for each feature
    avg_lime = np.abs(lime_values).mean(axis=0)
    
    
    shap_consistency = np.mean(np.abs(shap_values_A.flatten() - shap_values_B.flatten()))
    
    
    slope, intercept, r_value, p_value, std_err = linregress(shap_values_A.flatten(), shap_values_B.flatten())
    shap_icc = (r_value**2) * (np.sum(shap_values_A.flatten()**2) / len(shap_values_A.flatten()))
    
    """### LIME Model"""
    
    lime_consistency = np.mean(np.abs(lime_values_A.flatten() - lime_values_B.flatten()))    
    
    slope, intercept, r_value, p_value, std_err = linregress(lime_values_A.flatten(), lime_values_B.flatten())
    lime_icc = (r_value**2) * (np.sum(lime_values_A.flatten()**2) / len(lime_values_B.flatten()))
    
    
    # Errors
    
    y_predict = model.predict(X_test)
    
    mae = mean_absolute_error(y_predict, y_test)
    mse = mean_squared_error(y_predict, y_test)
    rmse = math.sqrt(mse)
    
    
    row = {"Model Name": model_name, "MAE": mae, "MSE": mse, "RMSE": rmse, 
           "SHAP Coverage":shap_coverages, "SHAP Avg Coverage": np.mean(shap_coverages), "SHAP Relevence":avg_shap, "SHAP Avg Relevence":np.mean(avg_shap), "SHAP CE":shap_consistency, "SHAP ICC":shap_icc,
            "LIME Coverage":lime_coverages, "LIME Avg Coverage": np.mean(lime_coverages), "LIME Relevence":avg_lime, "LIME Avg Relevence":np.mean(avg_lime), "LIME CE":lime_consistency, "LIME ICC":lime_icc}
    output_file.append(row)
    pd.DataFrame.from_dict(output_file).to_csv(output_file_path, index=False)


root_path = '..'
updated_dataset = root_path+"/UpdatedAirQualityDataset.csv"
df = pd.read_csv(updated_dataset)

features =list(df.columns[:-1])
dataset_x = df.drop("AQI", axis=1).values
dataset_y = df["AQI"].values
training_set = dataset_x[:500]
num_trained_item = 10000

x_train,x_test,y_train,y_test=train_test_split(dataset_x[:num_trained_item], dataset_y[:num_trained_item], test_size=0.2, random_state=42)


file_name = "ml_"+str(datetime.now())+".csv"
file_name = file_name.replace(' ', '-')
file_name = file_name.replace(':', '-')
output_file_path = root_path+"/result/"+file_name
output_file = []

"""# SHAP Explanation"""

start = datetime.now()

svr = SVR(kernel='rbf', C=1, epsilon=10)
svr.fit(x_train, y_train) 
XAIEvaluation(svr, "Support Vector Regressor")


# linear = LinearRegression()
# linear.fit(x_train, y_train)
# XAIEvaluation(linear, "Linear Regression")


# adaboost = AdaBoostRegressor(learning_rate=0.001, random_state=0, n_estimators=500) 
# adaboost.fit(x_train, y_train)
# XAIEvaluation(adaboost, "AdaBoost Regression")


# rf = RandomForestRegressor(n_estimators = 1000, random_state = 42) 
# rf.fit(x_train, y_train)
# XAIEvaluation(rf, "Random Forest Regressor")


# mlp = MLPRegressor(hidden_layer_sizes=(10), activation='relu', max_iter = 1880, solver='lbfgs')
# mlp.fit(x_train, y_train)
# XAIEvaluation(mlp, "MLP Regressor")

# total = datetime.now() - start
# print("Time: ", total)
