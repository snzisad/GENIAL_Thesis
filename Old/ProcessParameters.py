from BRBModel import ModelType
import pandas as pd

class ProcessParameters:  
    root_path = '..'
    disjunctive_parameters_csv_file = root_path+"/result/disjunctive/adp_2023-04-06-18-27-55.322862.csv"
    conjunctive_parameters_csv_file = root_path+"/result/conjunctive/jops_2023-04-03-15-06-05.701215.csvparam_2023-04-03-20-24-35.131671.csv"
    

    def getProcessedParameters(self, modelType, parameter_type):  
        self.parameter_type = parameter_type
        if modelType == ModelType.CONJUNCTIVE_BRB:
            return self.getConjunctiveBRBParam(self.conjunctive_parameters_csv_file)
        else:
            return self.getDisjunctiveBRBParam(self.disjunctive_parameters_csv_file) 
    
    def process_parameters(self, parameters_str):
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
        
    
    def getConjunctiveBRBParam(self, df_path):
        if self.parameter_type == "TRAINED":
            df = pd.read_csv(df_path)
            # print(df.sort_values(by=['mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error']))
            parameters_str = df.sort_values(by=['Test RMSE']).iloc[0]["Parameter"]
            
            # parameters_str = df.sort_values(by=['root_mean_squared_error', 'mean_absolute_error']).iloc[0]["Parameter"]
        else:
            parameters_str = "[1.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.8, 0.0, 0.2, 0.6, 0.4, 0.0, 0.4, 0.6, 0.0, 0.5, 0.3, 0.2, 0.8, 0.0, 0.2, 0.5, 0.3, 0.2, 0.2, 0.0, 0.8, 0.8, 0.2, 0.0, 0.4, 0.6, 0.0, 0.5, 0.3, 0.2, 0.4, 0.6, 0.0, 0.0, 1.0, 0.0, 0.0, 0.8, 0.2, 0.5, 0.3, 0.2, 0.0, 0.8, 0.2, 0.0, 0.2, 0.8, 0.8, 0.0, 0.2, 0.5, 0.3, 0.2, 0.2, 0.0, 0.8, 0.5, 0.3, 0.2, 0.0, 0.8, 0.2, 0.0, 0.2, 0.8, 0.2, 0.0, 0.8, 0.0, 0.2, 0.8, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]"
        
        return self.process_parameters(parameters_str)
    
    
    def getDisjunctiveBRBParam(self, df_path):
        if self.parameter_type == "TRAINED":
            df = pd.read_csv(df_path)
            # print(df.sort_values(by=['mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error']))
            parameters_str = df.sort_values(by=['Test RMSE']).iloc[0]["Parameter"]
            # parameters_str = df.sort_values(by=['root_mean_squared_error', 'mean_absolute_error']).iloc[0]["Parameter"]
        else:
            parameters_str = "[1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]"
        
        return self.process_parameters(parameters_str)

