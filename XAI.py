from BRBModel import BRBModel, ModelType
import numpy as np
import pandas as pd
from datetime import datetime
import shap
import lime
from scipy.stats import linregress
import lime
import lime.lime_tabular
import math



# model_type = ModelType.DISJUNCTIVE_BRB
model_type = ModelType.CONJUNCTIVE_BRB
# parameter_type = "TRAINED"
parameter_type = "NONTRAINED"


root_path = '..'
updated_dataset = root_path+"/UpdatedAirQualityDataset.csv"
disjunctive_parameters_csv_file = root_path+"/result/disjunctive/2023-03-24-02-52-33.774583.csv"
conjunctive_parameters_csv_file = root_path+"/result/conjunctive/2023-03-25-03-09-19.726411.csv"

 

def process_parameters(parameters_str):
    parameters_str = parameters_str.replace("\n", "")
    parameters_str = parameters_str.replace(". ", ".0,")
    parameters_str = parameters_str.replace(".0 ", ".0,")
    parameters_str = parameters_str.replace(" 0.", ", 0.")
    parameters_str = parameters_str.replace("e-01 ", "e-01, ")
    parameters_str = parameters_str.replace("e-02 ", "e-02, ")
    parameters_str = parameters_str.replace("e-03 ", "e-03, ")
    parameters_str = parameters_str.replace("e-04 ", "e-04, ")
    parameters_str = parameters_str.replace("e+00 ", "e+00, ")
    parameters_str = " ".join(parameters_str.split())
    parameters_str = parameters_str.replace(" ", ",")
    parameters_str = parameters_str.replace(", ,", ",")
    parameters_str = parameters_str.replace("[", "")
    parameters_str = parameters_str.replace("]", "")
    
    param = []
    for a in parameters_str.split(","):
        try:
            param.append(float(a))
        except:
            pass
        
    print(len(param))
    return param
    

def getConjunctiveBRBParam(df_path):
    if parameter_type == "TRAINED":
        df = pd.read_csv(df_path)
        # print(df.sort_values(by=['mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error']))
        
        parameters_str = df.sort_values(by=['root_mean_squared_error', 'mean_absolute_error']).iloc[0]["Parameter"]
    else:
        parameters_str = "[1.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.8, 0.0, 0.2, 0.6, 0.4, 0.0, 0.4, 0.6, 0.0, 0.5, 0.3, 0.2, 0.8, 0.0, 0.2, 0.5, 0.3, 0.2, 0.2, 0.0, 0.8, 0.8, 0.2, 0.0, 0.4, 0.6, 0.0, 0.5, 0.3, 0.2, 0.4, 0.6, 0.0, 0.0, 1.0, 0.0, 0.0, 0.8, 0.2, 0.5, 0.3, 0.2, 0.0, 0.8, 0.2, 0.0, 0.2, 0.8, 0.8, 0.0, 0.2, 0.5, 0.3, 0.2, 0.2, 0.0, 0.8, 0.5, 0.3, 0.2, 0.0, 0.8, 0.2, 0.0, 0.2, 0.8, 0.2, 0.0, 0.8, 0.0, 0.2, 0.8, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]"
    
    return process_parameters(parameters_str)


def getDisjunctiveBRBParam(df_path):
    if parameter_type == "TRAINED":
        df = pd.read_csv(df_path)
        # print(df.sort_values(by=['mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error']))
        
        parameters_str = df.sort_values(by=['root_mean_squared_error', 'mean_absolute_error']).iloc[0]["Parameter"]
    else:
        parameters_str = "[1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]"
    
    return process_parameters(parameters_str)



def predictWithBRB(data):
    pred = []
    brb = BRBModel(param, model_type, isDebug = False)
    for x in data:
        aqi = brb.runBRB(x)
        pred.append(aqi)

    return np.array(pred)


if model_type == ModelType.CONJUNCTIVE_BRB:
    param = getConjunctiveBRBParam(conjunctive_parameters_csv_file)
elif model_type == ModelType.DISJUNCTIVE_BRB:
    param = getDisjunctiveBRBParam(disjunctive_parameters_csv_file)

df = pd.read_csv(updated_dataset)

features =list(df.columns[:-1])
dataset_x = df.drop("AQI", axis=1).values
dataset_y = df["AQI"].values
training_set = dataset_x[:500]

"""# SHAP Explanation"""

start = datetime.now()

# Create a shap explainer
shap_explainer = shap.KernelExplainer(predictWithBRB, training_set)

test_pos = 100
test_data = dataset_x[test_pos]
original_aqi = dataset_y[test_pos]
shap_values = shap_explainer.shap_values(test_data)

duration = datetime.now() - start
print("Total time: "+str(duration))

shap.initjs()
shap.force_plot(shap_explainer.expected_value, shap_values, test_data, feature_names=features)


"""# Text Explanation"""

base_value = shap_explainer.expected_value
total_changed = sum(shap_values)
predicted_aqi = base_value+total_changed
feature_impact = []
explaination = ""

impact_status = []
feature_impact= []

for val in shap_values:
      if val > 0:
          impact_status.append("positive")
      else:
          impact_status.append("negative")

      feature_impact.append(abs(val))

total_impact = sum(feature_impact)
normalized_feature_impact = feature_impact/total_impact

for i in range(len(shap_values)):
    impact = feature_impact[i]
    if impact == 0:
        explaination += str(features[i]) + " has no impact, "
    else:
        explaination += str(features[i]) + " has "+ str(round(normalized_feature_impact[i]*100, 2)) +"% " + impact_status[i] +" impact, "

explaination += "to make the prediction "+str(str(round(predicted_aqi, 2)))

print("Explanation: \n")
print(explaination)

print("Original AQI: "+str(original_aqi))
print("Base value: "+str(base_value))
print("Predicted value: "+str(predicted_aqi))

# Feature Importance
import matplotlib.pyplot as plt

plt.figure()
barchart = plt.barh(features, feature_impact, color="b")

# Set red color for negative value
for i in range(len(feature_impact)):
  if feature_impact[i]<0:
    barchart[i].set_color('r')

plt.title("Feature importances")
plt.xlabel("Importance")
plt.ylabel("Feature name")
plt.show()

"""# LIME Explaination"""


lime_explainer = lime.lime_tabular.LimeTabularExplainer(training_set, feature_names=features, class_names=['AQI'], verbose=True, mode='regression')
lime_exp = lime_explainer.explain_instance(test_data, predictWithBRB, num_features=len(features), labels=(1,))
print(lime_exp)
lime_exp.show_in_notebook(show_table=True)

"""# Text Explaination"""

impact_status = []
feature_impact= []
explaination = ""

for val in lime_exp.as_map().get(1):
    if val[1] > 0:
        impact_status.append("Positive")
    else:
        impact_status.append("Negative")

    feature_impact.append(abs(val[1]))

total_impact = sum(feature_impact)
normalized_feature_impact = feature_impact/total_impact

for i in range(len(feature_impact)):
    impact = feature_impact[i]
    if impact == 0:
        explaination += str(features[i]) + " has no impact, "
    else:
        explaination += str(features[i]) + " has "+ str(round(normalized_feature_impact[i]*100, 2)) +"% " + impact_status[i] +" impact, "

explaination += "to make the prediction "+str(str(round(predicted_aqi, 2)))

print("Explanation: \n")
print(explaination)

# https://github.com/marcotcr/lime/blob/master/lime/explanation.py

print(lime_exp.as_list())
print(lime_exp.as_map())
print(lime_exp.intercept)
print(lime_exp.max_value)
print(lime_exp.min_value)
print(lime_exp.predicted_value)
print(lime_exp.score)
lime_exp.as_pyplot_figure()

"""# Evaluation Metrics

# Errors
"""

from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error

start = datetime.now()

num_trained_item = 10000
X_test = dataset_x[:num_trained_item]
y_test = dataset_y[:num_trained_item]


"""# Feature Coverage
Average number of non zero score

1. Create a shap.Explainer object to explain the model's predictions.
2. Compute SHAP values for each instance in the test set using the shap_values method of the shap.Explainer object.
4. For each feature in the dataset, calculate the percentage of instances in the set from step 3 for which the SHAP value of that feature is nonzero.
6. Calculate the average percentage of nonzero SHAP values

### SHAP Model
"""

num_items = 200
subset_A = X_test[:num_items]
subset_B = X_test[num_items:2*num_items]

shap_values_A = shap_explainer.shap_values(subset_A)
shap_values_B = shap_explainer.shap_values(subset_B)

shap_values = np.array([*shap_values_A, *shap_values_B])


"""### LIME Model"""

def get_lime_values(dataset):
    lime_values = []
    single_lime_value = []

    for test_data in dataset:

      lime_exp = lime_explainer.explain_instance(test_data, predictWithBRB, num_features=len(features), labels=(1,))

      coeffs = lime_exp.local_exp[0]
      single_lime_value = [0] * len(features)
      for c in coeffs:
          single_lime_value[c[0]] += c[1]

      lime_values.append(single_lime_value)

    return np.array(lime_values)

num_items = 200
subset_A = X_test[:num_items]
subset_B = X_test[num_items:2*num_items]

lime_values_A = get_lime_values(subset_A)
lime_values_B = get_lime_values(subset_B)

lime_values = np.array([*lime_values_A, *lime_values_B])

# Calculate feature coverage
coverages = []

nonzero_counts = np.sum(shap_values != 0, axis=0)
nonzero_percents = nonzero_counts / shap_values.shape[0]
coverages.extend(nonzero_percents)

print("SHAP")
for feature, coverage in zip(features, coverages):
    print(f"{feature}: {coverage}")

print(f"\nAverage Coverage: {np.mean(coverages)}")

# Calculate feature coverage
coverages = []

nonzero_counts = np.sum(lime_values != 0, axis=0)
nonzero_percents = nonzero_counts / lime_values.shape[0]
coverages.extend(nonzero_percents)

print("LIME")
for feature, coverage in zip(features, coverages):
    print(f"{feature}: {coverage}")

print(f"\nAverage Coverage: {np.mean(coverages)}")

"""# Relevance

1. Calculate the SHAP values for each feature
2. Once we have the SHAP values, we can calculate the average absolute SHAP value for each feature. This will give an idea of how much each feature is contributing to the model's predictions

### SHAP Model
"""

# calculate the average absolute SHAP value for each feature
avg_shap = np.abs(shap_values).mean(axis=0)


print("SHAP")
# print the feature importance in descending order
for feature, importance in zip(features, avg_shap):
    print(f"{feature}: {importance}")

"""### LIME Model"""

# calculate the average absolute lime value for each feature
avg_lime = np.abs(lime_values).mean(axis=0)

print("LIME")
# print the feature importance in descending order
for feature, importance in zip(features, avg_lime):
    print(f"{feature}: {importance}")

"""# Consistency

1. Load a dataset and train a machine learning model on it.
2. Create a SHAP explainer object using the trained model and the dataset.
Select a test set from the dataset for which to calculate SHAP values and consistency.
3. Calculate the SHAP values for the test set using the explainer object.
4. For each feature in the dataset, shuffle the values of that feature in the test set and calculate the SHAP values for the shuffled data using the explainer object.
5. Calculate the maximum absolute difference of the test set SHAP values and shuffled set SHAP values
6. Compute the mean of the consistency errors across all features and classes to obtain the consistency score.

### SHAP Model
"""

print("SHAP")
consistency = np.mean(np.abs(shap_values_A.flatten() - shap_values_B.flatten()))
print("Consistency Error:", consistency)


slope, intercept, r_value, p_value, std_err = linregress(shap_values_A.flatten(), shap_values_B.flatten())
icc = (r_value**2) * (np.sum(shap_values_A.flatten()**2) / len(shap_values_A.flatten()))
print("Pearson correlation coefficient: "+str(r_value))
print(f"ICC: {icc}")

"""### LIME Model"""

print("LIME")
consistency = np.mean(np.abs(lime_values_A.flatten() - lime_values_B.flatten()))
print("Consistency Error:", consistency)


slope, intercept, r_value, p_value, std_err = linregress(lime_values_A.flatten(), lime_values_B.flatten())
icc = (r_value**2) * (np.sum(lime_values_A.flatten()**2) / len(lime_values_B.flatten()))
print("Pearson correlation coefficient: "+str(r_value))
print(f"ICC: {icc}")

"""# Test-retest reliability

1. Load a dataset and train a machine learning model on it.

2. Create a SHAP explainer object using the trained model and the dataset. Select a test set from the dataset for which to calculate SHAP values and consistency.

3. Calculate the SHAP values for the test set using the explainer object.

4. For each feature in the dataset, shuffle the values of that feature in the test set and calculate the SHAP values for the shuffled data using the explainer object.

5. Calculate the intraclass correlation coefficient (ICC) between the SHAP values generated by the different runs of the model.

6. Interpret the ICC value to determine the test-retest reliability of the SHAP values.


"""

# Errors

y_predict = predictWithBRB(X_test)

print("MAE: "+str(mean_absolute_error(y_predict, y_test)))
print("MSE: "+str(mean_squared_error(y_predict, y_test)))
print("RMSE: "+str(math.sqrt(mean_squared_error(y_predict, y_test))))

duration = datetime.now() - start
print("Total time: "+str(duration))