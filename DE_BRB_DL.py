from BRBModel import ModelType
from BRB_DL_Model import BRBDL
import numpy as np
import pandas as pd
from random import *
import math
from datetime import datetime
from random import *
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, mean_absolute_percentage_error, r2_score, mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance, d2_tweedie_score, mean_pinball_loss

# model_type = ModelType.DISJUNCTIVE_BRB
model_type = ModelType.CONJUNCTIVE_BRB


def tuneBRBParams(param):
    predicted_aqi = []
    brb = BRBDL(param, model_type, num_ref_value=numRefVal, isDebug = False)
    for i in range(len(X_train)):
        aqi = brb.runBRB(X_train[i])
        predicted_aqi.append(aqi)
    
    return [mean_absolute_error(Y_train, predicted_aqi), mean_squared_error(Y_train, predicted_aqi), math.sqrt(mean_squared_error(Y_train, predicted_aqi)), r2_score(Y_train, predicted_aqi), brb.PM2_5_ref_val[1], brb.PM10_ref_val[1], brb.NH3_ref_val[1]]


def getParameterNumber(numRefVal):    
    numAntAttr = 3
    transformed_input_size = numAntAttr*numRefVal
    num_hidden_layers = 5
    num_neuron_size = 100
    
    if model_type == ModelType.CONJUNCTIVE_BRB:
        num_activation_weight = numRefVal**numAntAttr
        num_param = (numRefVal**numAntAttr)*(numRefVal+1)+5*numRefVal
    else:
        num_activation_weight = numRefVal
        num_param = numRefVal*(numRefVal+6)
        
    num_dl_param = num_neuron_size*(transformed_input_size+num_activation_weight+num_hidden_layers*num_neuron_size)+num_hidden_layers+1
    return num_param+num_dl_param

def runDE():
    global file_name, scaling_rate, crossover_rate
    
    start_time = datetime.now()
    
    num_param = getParameterNumber(numRefVal)
    # parameter_list = np.random.uniform (0, 1, (population_size, num_param))
    parameter_list = np.random.rand(population_size, num_param) 
    min_b, max_b = np.asarray([0,1]).T  
    diff = np.fabs(min_b - max_b)
    parameter_list = min_b + parameter_list * diff 
    
    parent_metrics= np.asarray([tuneBRBParams(ind) for ind in parameter_list])
    
    
    for i in range(num_iteration): 
        # For adaptive BRB
        error_diff = 0.0
        param_diff= 0.0
        #----------------
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
                # For adaptive BRB
                error_diff += (parent_metrix[1] - trial_metrix[1])**2
                param_diff += (parameter_list[j] - trial_param)**2
                #----------------
                parent_metrics[j] = trial_metrix
                parameter_list[j] = trial_param
                
        best_idx = np.argmin(np.array(parent_metrics)[:,2])  
        best_param = list(parameter_list[best_idx])
        
        test_rmse = getTestRMSE(best_param)
        row = {"Iteration":(i+1), "Scaling Rate":scaling_rate, "Crossover Rate":crossover_rate, "Ref Val":numRefVal, "Population Size":population_size, "Test RMSE":test_rmse, "Train MAE":parent_metrics[best_idx][0], "Train MSE":parent_metrics[best_idx][1], "Train RMSE":parent_metrics[best_idx][2], "r2_score":parent_metrics[best_idx][3], "Parameter":best_param, "PM2_5_mid":parent_metrics[best_idx][4], "PM10_mid":parent_metrics[best_idx][5], "NH3_mid":parent_metrics[best_idx][6]}
        output_file.append(row)
        pd.DataFrame.from_dict(output_file).to_csv(output_file_path, index=False)

    
        if ((i+1)%10==0):
            print("Iter-"+str(i+1)+", RMSE: "+str(parent_metrics[best_idx][2]))
            end_time = datetime.now()
            print("Time: "+str(end_time-start_time))
        else:
            print("Iter: "+str(i+1))
            end_time = datetime.now()
            # print("Time: "+str(end_time-start_time))
            
    return best_param       


def getTestRMSE(best_param):
    predicted_aqi = []
    brb = BRBDL(best_param, model_type, num_ref_value=numRefVal, isDebug = False)
    
    for data in X_test:
        aqi = brb.runBRB(data)
        predicted_aqi.append(aqi)
    
    return math.sqrt(mean_squared_error(Y_test, predicted_aqi))

root_path = '../'
updated_dataset = root_path+"/UpdatedAirQualityDataset.csv"
df = pd.read_csv(updated_dataset)

features =list(df.columns[:-1])
dataset_x = df.drop("AQI", axis=1).values
dataset_y = df["AQI"].values


if model_type == ModelType.CONJUNCTIVE_BRB:
    file_path = "conjunctive/dl_"
else:
    file_path = "disjunctive/dl_"
    

file_name = file_path+str(datetime.now())+".csv"
file_name = file_name.replace(' ', '-')
file_name = file_name.replace(':', '-')
output_file_path = root_path+"/result/"+file_name
output_file = []


scaling_rate = 0.5
crossover_rate = 0.3
population_size = 20
num_iteration = 100
num_trained_item = 10000
num_test_item = 400
X_train = dataset_x[:num_trained_item]
Y_train = dataset_y[:len(X_train)]
X_test = dataset_x[num_trained_item:num_trained_item+num_test_item]
Y_test = dataset_y[num_trained_item:num_trained_item+num_test_item]
numRefVal = 3
best_param = runDE()