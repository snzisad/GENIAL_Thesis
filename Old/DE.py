from BRBModel import BRBModel, ModelType
import numpy as np
import pandas as pd
import math
from datetime import datetime
from random import *
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, mean_absolute_percentage_error, r2_score, mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance, d2_tweedie_score, mean_pinball_loss

model_type = ModelType.DISJUNCTIVE_BRB
file_name = "disjunctive/"+str(datetime.now())+".csv"

# model_type = ModelType.CONJUNCTIVE_BRB
# file_name = "conjunctive/"+str(datetime.now())+".csv"



root_path = '../'
updated_dataset = root_path+"/UpdatedAirQualityDataset.csv"
df = pd.read_csv(updated_dataset)

features =list(df.columns[:-1])
dataset_x = df.drop("AQI", axis=1).values
dataset_y = df["AQI"].values

num_param = 150
scaling_rate = 0.5
crossover_rate =0.3
population_size=20
num_iteration = 500
X_train = dataset_x[:100000]
Y_train = dataset_y[:len(X_train)]
parameter_list = np.random.uniform (0, 1, (population_size, num_param))



def tuneBRBParams(param):
    # if model_type == ModelType.CONJUNCTIVE_BRB:
    #     scaling_rate = param[114]
    #     crossover_rate = param[115]
    # else:
    #     scaling_rate = param[18]
    #     crossover_rate = param[19]
        
    predicted_aqi = []
    brb = BRBModel(param, model_type, isDebug = False)
    for i in range(len(X_train)):
        aqi = brb.runBRB(X_train[i])
        predicted_aqi.append(aqi)
    
    return [mean_absolute_error(Y_train, predicted_aqi), mean_squared_error(Y_train, predicted_aqi), math.sqrt(mean_squared_error(Y_train, predicted_aqi)), r2_score(Y_train, predicted_aqi), brb.PM2_5_ref_val[1], brb.PM10_ref_val[1], brb.NH3_ref_val[1]]


start_time = datetime.now()

file_name = file_name.replace(' ', '-')
file_name = file_name.replace(':', '-')
output_file_path = root_path+"/result/"+file_name

output_file = []
parent_metrics= np.asarray([tuneBRBParams(ind) for ind in parameter_list])


for i in range(num_iteration):  
    for j in range(population_size):  
        idxs = [idx for idx in range(population_size) if idx != j]

        # Mutation
        Pa, Pb, Pc = parameter_list[np.random.choice(idxs, 3, replace = False)]
        mutant = np.clip(Pa + scaling_rate * (Pb - Pc), 0, 1) 

        # Cross Over
        cross_points = np.random.rand(num_param) < crossover_rate
        
        # If the whole list consists of false, select a random position and make it true
        if not np.any(cross_points):
            cross_points[np.random.randint(0, num_param)] = True

        # Trail selection
        trial_param = np.where(cross_points, mutant, parameter_list[j])
        
        trial_metrix = tuneBRBParams(trial_param)
        parent_metrix = parent_metrics[j]
        if trial_metrix[0] < parent_metrix[0]:          
            parent_metrics[j] = trial_metrix
            parameter_list[j] = trial_param

    best_idx = np.argmin(np.array(parent_metrics)[:,2])  
    best_param = parameter_list[best_idx]
    row = {"Iteration":(i+1), "Scaling Rate":best_param[18], "Crossover Rate":best_param[19], "Population Size":population_size, "mean_absolute_error":parent_metrics[best_idx][0], "mean_squared_error":parent_metrics[best_idx][1], "root_mean_squared_error":parent_metrics[best_idx][2], "r2_score":parent_metrics[best_idx][3], "Parameter":best_param, "PM2_5_mid":parent_metrics[best_idx][4], "PM10_mid":parent_metrics[best_idx][5], "NH3_mid":parent_metrics[best_idx][6]}
    output_file.append(row)
    pd.DataFrame.from_dict(output_file).to_csv(output_file_path, index=False)


    if ((i+1)%10==0):
        print("Iter-"+str(i+1)+", RMSE: "+str(parent_metrics[best_idx][2]))
        end_time = datetime.now()
        print("Time: "+str(end_time-start_time))
    else:
        print("Iter: "+str(i+1))