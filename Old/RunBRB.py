from BRBModel import BRBModel, ModelType
from BRB_JOPS import BRBJOPS
import numpy as np
import pandas as pd

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
    df = pd.read_csv(df_path)
    # print(df.sort_values(by=['mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error']))
    
    # parameters_str = df.sort_values(by=['root_mean_squared_error', 'mean_absolute_error']).iloc[0]["Parameter"]
    parameters_str = "[1.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.8, 0.0, 0.2, 0.6, 0.4, 0.0, 0.4, 0.6, 0.0, 0.5, 0.3, 0.2, 0.8, 0.0, 0.2, 0.5, 0.3, 0.2, 0.2, 0.0, 0.8, 0.8, 0.2, 0.0, 0.4, 0.6, 0.0, 0.5, 0.3, 0.2, 0.4, 0.6, 0.0, 0.0, 1.0, 0.0, 0.0, 0.8, 0.2, 0.5, 0.3, 0.2, 0.0, 0.8, 0.2, 0.0, 0.2, 0.8, 0.8, 0.0, 0.2, 0.5, 0.3, 0.2, 0.2, 0.0, 0.8, 0.5, 0.3, 0.2, 0.0, 0.8, 0.2, 0.0, 0.2, 0.8, 0.2, 0.0, 0.8, 0.0, 0.2, 0.8, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]"
    
    return process_parameters(parameters_str)


def getDisjunctiveBRBParam(df_path):
    df = pd.read_csv(df_path)
    # print(df.sort_values(by=['mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error']))
    
    # parameters_str = df.sort_values(by=['root_mean_squared_error', 'mean_absolute_error']).iloc[0]["Parameter"]
    parameters_str = "[1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]"
    
    return process_parameters(parameters_str)




root_path = '..'
disjunctive_parameters_csv_file = root_path+"/result/disjunctive/2023-03-24-02-52-33.774583.csv"
conjunctive_parameters_csv_file = root_path+"/result/conjunctive/2023-03-25-03-09-19.726411.csv"

# param = getDisjunctiveBRBParam(disjunctive_parameters_csv_file)
# param = getConjunctiveBRBParam(conjunctive_parameters_csv_file)
# print(param)

inputs = [104, 500, 9.8]

# brb = BRBJOPS(getDisjunctiveBRBParam(disjunctive_parameters_csv_file), ModelType.DISJUNCTIVE_BRB, 3, isDebug = True)
brb = BRBModel(getDisjunctiveBRBParam(disjunctive_parameters_csv_file), ModelType.DISJUNCTIVE_BRB, isDebug = True)

# brb = BRBModel(getConjunctiveBRBParam(conjunctive_parameters_csv_file), ModelType.CONJUNCTIVE_BRB, isDebug = True)
# brb = BRBJOPS(getConjunctiveBRBParam(conjunctive_parameters_csv_file), ModelType.CONJUNCTIVE_BRB, isDebug = True)

predicted_aqi = brb.runBRB(inputs)
brb.PM10_ref_val
print(predicted_aqi)
# brb.printParameters()
