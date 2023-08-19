from DynamicBRB import ModelType
from JOPSModel import JOPSModel
from BRB_CR_F import BRB_CR_F, ParamType
import numpy as np
import pandas as pd
from random import *
import math
from datetime import datetime
from random import *
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, mean_absolute_percentage_error, r2_score, mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance, d2_tweedie_score, mean_pinball_loss

model_type = ModelType.DISJUNCTIVE_BRB
# model_type = ModelType.CONJUNCTIVE_BRB

is_adaptive = False
is_jops = False


def tuneBRBParams(param):
    predicted_aqi = []
    brb = JOPSModel(modelType = model_type, num_ant_attr = 3, num_ref_values = numRefVal, isDebug = False)
    brb.setAttributes(param, is_jops)
    for i in range(len(X_train)):
        aqi = brb.runBRB(X_train[i])
        predicted_aqi.append(aqi)
    
    print(param)
    return [mean_absolute_error(Y_train, predicted_aqi), mean_squared_error(Y_train, predicted_aqi), math.sqrt(mean_squared_error(Y_train, predicted_aqi)), r2_score(Y_train, predicted_aqi)]


def getParameterNunber(numRefVal):    
    numAntAttr = 3
    if model_type == ModelType.CONJUNCTIVE_BRB:
        num_param = 2*(numRefVal**numAntAttr)*numAntAttr+4*numRefVal
    else:
        num_param = 2*numRefVal*(numAntAttr+2)
        
    return num_param  

def runDE():
    global file_name, scaling_rate, crossover_rate
    
    start_time = datetime.now()
    
    num_param = getParameterNunber(numRefVal)
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
                
        if is_adaptive and error_diff>0.0:
            pc = math.sqrt(np.sum(param_diff)/population_size)
            fc = math.sqrt(error_diff/population_size)
            d11_t = (1 + pc) * (math.exp(-pc))   
            d11 = 1 - d11_t
            d12_t = (1 + fc) * (math.exp(-fc))     
            d12 = 1 - d12_t 
            d21 = 2 * d11
            d22 = 2 * d12
            
            scaling_rate = BRB_CR_F(model_type, ParamType.SCALING_RATIO).runBRB([d21, d22])
            crossover_rate = BRB_CR_F(model_type, ParamType.CROSSOVER_RATE).runBRB([d11, d12])
            
            
        best_idx = np.argmin(np.array(parent_metrics)[:,2])  
        best_param = list(parameter_list[best_idx])
        
        test_rmse = getTestRMSE(best_param)
        row = {"Iteration":(i+1), "Scaling Rate":scaling_rate, "Crossover Rate":crossover_rate, "Ref Val":numRefVal, "Population Size":population_size, "Test RMSE":test_rmse, "Train MAE":parent_metrics[best_idx][0], "Train MSE":parent_metrics[best_idx][1], "Train RMSE":parent_metrics[best_idx][2], "r2_score":parent_metrics[best_idx][3], "Parameter":best_param}
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
    brb = JOPSModel(modelType = model_type, num_ant_attr = 3, num_ref_values = numRefVal, isDebug = False)
    brb.setAttributes(best_param, is_jops)
    
    for data in X_test:
        aqi = brb.runBRB(data)
        predicted_aqi.append(aqi)
    
    return math.sqrt(mean_squared_error(Y_test, predicted_aqi))

def selectNextRefVal(pos):
    if refValList[pos+1] == 0:
        low_val = refValList[pos]+1
    elif msel[pos] >= msel[pos+1] and refValList[pos+1] >= refValList[pos]:
        low_val = refValList[pos]+1
    elif msel[pos] < msel[pos+1] and refValList[pos] >= refValList[pos+1]:
            low_val = refValList[pos+1]+1
    else:
        low_val = 2
    
    if low_val>=6:
        return 6
    
    if low_val<=3:
        low_val = 3
    return np.random.randint(low = low_val, high = 6, size = 1)[0]

root_path = '../'
updated_dataset = root_path+"/UpdatedAirQualityDataset.csv"
df = pd.read_csv(updated_dataset)

features =list(df.columns[:-1])
dataset_x = df.drop("AQI", axis=1).values
dataset_y = df["AQI"].values


if is_adaptive and is_jops:
    if model_type == ModelType.CONJUNCTIVE_BRB:
        file_path = "conjunctive/jops_"
    else:
        file_path = "disjunctive/jops_"
    
elif is_adaptive:
    if model_type == ModelType.CONJUNCTIVE_BRB:
        file_path = "conjunctive/adp_"
    else:
        file_path = "disjunctive/adp_"
    
else:
    if model_type == ModelType.CONJUNCTIVE_BRB:
        file_path = "conjunctive/"
    else:
        file_path = "disjunctive/"
    

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

if is_adaptive and is_jops:
    # JOPS Start
    refValList = [0] * 5
    msel = [0] * 5
    
    
    start_time = datetime.now()
    param_file_name = file_path+"param_"+str(datetime.now())+".csv"
    param_file_name = param_file_name.replace(' ', '-')
    param_file_name = param_file_name.replace(':', '-')
    param_output_file_path = root_path+"/result/"+param_file_name
    param_output_file = []
    
    mse_1 = getTestRMSE(best_param)
    msel[0] =  mse_1
    refValList[0] = numRefVal
    
    row = {"Ref Val": numRefVal, "Test RMSE": mse_1, "Parameter": best_param}
    param_output_file.append(row)
    pd.DataFrame.from_dict(param_output_file).to_csv(param_output_file_path, index=False)
    
    for i in range(1, 5):
        numRefVal = selectNextRefVal(i-1)
        best_param = runDE()
        mse_2 = getTestRMSE(best_param)
        msel[i] =  mse_2
        refValList[i] = numRefVal
        
        row = {"Ref Val": numRefVal, "Test RMSE": mse_2, "Parameter": best_param}
        param_output_file.append(row)
        pd.DataFrame.from_dict(param_output_file).to_csv(param_output_file_path, index=False)
    
    
    min_rmse_pos = np.argmin(msel)
    final_ref_val = refValList[min_rmse_pos]
    
    print("Minimum RMSE: "+str(msel[min_rmse_pos]))
    print("Ref Val: "+str(refValList[min_rmse_pos]))
    
    
    end_time = datetime.now()
    print("Total Time: "+str(end_time-start_time))

