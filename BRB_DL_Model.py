from BRBModel import BRBModel, ModelType
import numpy as np

class BRBDL(BRBModel):
        
    def setAttributes(self):
        self.PM2_5_ref_val.clear()
        self.PM10_ref_val.clear()
        self.NH3_ref_val.clear()
        self.relativeWeight.clear()
        self.initialRuleWeight.clear()
        self.utilityScore.clear()
        
        PM2_5_ref_val_pos = len(self.consequentBeliefDegree) * len(self.consequentBeliefDegree[0])
        for i in range(self.num_ref_value):
            self.PM2_5_ref_val.append(self.param[PM2_5_ref_val_pos+i] * 1000)
        
        PM10_ref_val_pos = PM2_5_ref_val_pos+self.num_ref_value
        for i in range(self.num_ref_value):
            self.PM10_ref_val.append(self.param[PM10_ref_val_pos+i] * 1000)
            
        NH3_ref_val_pos = PM10_ref_val_pos+self.num_ref_value
        for i in range(self.num_ref_value):
            self.NH3_ref_val.append(self.param[NH3_ref_val_pos+i] * 2000)
        
        relativeWeight_pos = NH3_ref_val_pos+self.num_ref_value
        for i in range(self.num_ref_value):
            self.relativeWeight.append(self.param[relativeWeight_pos+i])
            
        initialRuleWeight_pos = relativeWeight_pos+self.num_ref_value
        for i in range(len(self.consequentBeliefDegree)):
            self.initialRuleWeight.append(self.param[initialRuleWeight_pos+i])
        
        utilityScore_pos = initialRuleWeight_pos+len(self.consequentBeliefDegree)
        
        for i in range(self.num_ref_value):
            self.utilityScore.append(self.param[utilityScore_pos+i])
            
        self.initDNN(utilityScore_pos+self.num_ref_value, len(self.initialRuleWeight))
        
        
        
            
    def initDNN(self, last_used_pos, output_size):
        self.input_size = self.numberOfAntAttributes*self.num_ref_value
        self.num_hidden_layers = 5
        self.num_neuron_size = 100
        self.weights = []
        np_param = np.array(self.param)
        
        for i in range(self.num_hidden_layers+2):
            # input layer
            if i == 0:
                total_neuron = self.num_neuron_size*self.input_size
                self.weights.append(np.reshape(np_param[last_used_pos:last_used_pos+total_neuron], (-1, self.input_size)))
            
            # output layer
            elif i == self.num_hidden_layers+1:
                total_neuron = output_size*self.num_neuron_size
                self.weights.append(np.reshape(np_param[last_used_pos:last_used_pos+total_neuron], (-1, self.num_neuron_size)))
                
            # Hidden layers  
            else:
                total_neuron = self.num_neuron_size*self.num_neuron_size
                self.weights.append(np.reshape(np_param[last_used_pos:last_used_pos+total_neuron], (-1, self.num_neuron_size)))
                
            last_used_pos += total_neuron
            
        self.biases = np_param[last_used_pos:last_used_pos+self.num_hidden_layers+1]
        
            
    def runBRB(self, inputs):    
        self.consequentBeliefDegree = self.initRuleBase()
        self.setAttributes()
        transformed_input_PM2_5 = self.transformInput(inputs[0], self.PM2_5_ref_val)
        transformed_input_PM10 = self.transformInput(inputs[1], self.PM10_ref_val)
        transformed_input_NH3= self.transformInput(inputs[2], self.NH3_ref_val)
        
        self.calculateActivationWeight(transformed_input_PM2_5+transformed_input_PM10+transformed_input_NH3)
        self.updateBeliefDegree(transformed_input_PM2_5, transformed_input_PM10, transformed_input_NH3)
        aqi = self.aggregateER()
        self.printf("AQI: "+ str(aqi))
        return aqi
            

    def calculateActivationWeight(self, transformed_input):
        self.activationWeight.clear()
        
        transformed_input = np.array(transformed_input)
        prev_layer_input = self.sigmoid(transformed_input.dot(self.weights[0].T)+self.biases[0])
        for i in range(1, self.num_hidden_layers):
            # print(self.biases[i])
            prev_layer_input = self.sigmoid(prev_layer_input.dot(self.weights[i])+self.biases[i])
        
        activation_weights = self.sigmoid(prev_layer_input.dot(self.weights[-1].T)+self.biases[-1])

        self.activationWeight.extend(activation_weights)
    
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    

