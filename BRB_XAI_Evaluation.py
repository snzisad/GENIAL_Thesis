from BRBModel import ModelType
from BRB_JOPS import BRBJOPS
import numpy as np
import pandas as pd
from datetime import datetime
import shap
import lime
from scipy.stats import linregress
import lime
import lime.lime_tabular
import math
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error


# model_type = ModelType.DISJUNCTIVE_BRB
model_type = ModelType.CONJUNCTIVE_BRB

parameter_type = "TRAINED"
# parameter_type = "NONTRAINED"


root_path = '..'
disjunctive_parameters_csv_file = root_path+"/result/disjunctive/dl_2023-04-28-16-57-28.037705.csv"
conjunctive_parameters_csv_file = root_path+"/result/conjunctive/dl_2023-04-28-20-04-57.983775.csv"


if model_type == ModelType.DISJUNCTIVE_BRB:
    if parameter_type == "TRAINED":
        df = pd.read_csv(disjunctive_parameters_csv_file)
        selected_row = df.sort_values(by=['Test RMSE', 'Train RMSE']).iloc[0]
        parameters_str = selected_row["Parameter"]
        numRefVal = int(selected_row["Ref Val"])
    else:
        numRefVal = 3 
        parameters_str = '[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.061, 0.0, 1.0, 0.101, 0.0, 1.0, 0.1005, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0]'
else:
    if parameter_type == "TRAINED":
        df = pd.read_csv(conjunctive_parameters_csv_file)
        selected_row = df.sort_values(by=['Test RMSE', 'Train RMSE']).iloc[0]
        parameters_str = selected_row["Parameter"]
        numRefVal = int(selected_row["Ref Val"])
    else:
        numRefVal = 3  
        parameters_str = '[1.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.8, 0.0, 0.2, 0.6, 0.4, 0.0, 0.4, 0.6, 0.0, 0.5, 0.3, 0.2, 0.8, 0.0, 0.2, 0.5, 0.3, 0.2, 0.2, 0.0, 0.8, 0.8, 0.2, 0.0, 0.4, 0.6, 0.0, 0.5, 0.3, 0.2, 0.4, 0.6, 0.0, 0.0, 1.0, 0.0, 0.0, 0.8, 0.2, 0.5, 0.3, 0.2, 0.0, 0.8, 0.2, 0.0, 0.2, 0.8, 0.8, 0.0, 0.2, 0.5, 0.3, 0.2, 0.2, 0.0, 0.8, 0.5, 0.3, 0.2, 0.0, 0.8, 0.2, 0.0, 0.2, 0.8, 0.2, 0.0, 0.8, 0.0, 0.2, 0.8, 0.0, 0.0, 1.0, 1.0, 0.061, 0.0, 1.0, 0.101, 0.0, 1.0, 0.1005, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0]'

def predictWithBRB(data):
    pred = []
    brb = BRBJOPS(param, model_type, num_ref_value=numRefVal, isDebug = False)
    for x in data:
        aqi = brb.runBRB(x)
        # print(brb.PM2_5_ref_val)
        # print(brb.PM10_ref_val)
        # print(brb.NH3_ref_val)
        # print(brb.utilityScore)
        # return
        pred.append(aqi)

    return np.array(pred)

parameters_str = parameters_str.replace("[", "")
parameters_str = parameters_str.replace("]", "")

param = []
for a in parameters_str.split(","):
    try:
        param.append(float(a))
    except:
        pass

print(len(param))

updated_dataset = root_path+"/UpdatedAirQualityDataset.csv"
df = pd.read_csv(updated_dataset)

features =list(df.columns[:-1])
dataset_x = df.drop("AQI", axis=1).values
dataset_y = df["AQI"].values
training_set = dataset_x[:500]

predictWithBRB([dataset_x[0]])

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



lime_explainer = lime.lime_tabular.LimeTabularExplainer(training_set, feature_names=features, class_names=['AQI'], verbose=True, mode='regression')

"""# Evaluation Metrics

# Errors
"""

start = datetime.now()

num_trained_item = 10000
X_test = dataset_x[num_trained_item:]
y_test = dataset_y[num_trained_item:]


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

print(f"\nAverage Relevence: {np.mean(avg_shap)}")

"""### LIME Model"""

# calculate the average absolute lime value for each feature
avg_lime = np.abs(lime_values).mean(axis=0)

print("LIME")
# print the feature importance in descending order
for feature, importance in zip(features, avg_lime):
    print(f"{feature}: {importance}")

print(f"\nAverage Relevence: {np.mean(avg_lime)}")

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

print("\nMAE: "+str(mean_absolute_error(y_predict, y_test)))
print("MSE: "+str(mean_squared_error(y_predict, y_test)))
print("RMSE: "+str(math.sqrt(mean_squared_error(y_predict, y_test))))

duration = datetime.now() - start
print("Total time: "+str(duration))