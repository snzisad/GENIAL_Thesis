import numpy as np
from enum import Enum
 
class ModelType(Enum):
    DISJUNCTIVE_BRB = 1
    CONJUNCTIVE_BRB = 2
 

class BRBModel:
    PM2_5_mid_val = 100
    PM10_mid_val = 500
    NH3_mid_val = 1000
    
    PM2_5_ref_val = [1000, 61, 0.0]
    PM10_ref_val = [1000, 101, 0]
    NH3_ref_val = [2000, 201, 0]
    
    utilityScore = [1, 0.5, 0]
    referencesValues_text = ["Severe", "Poor", "Good"]
    matchingDegree = []
    activationWeight = []
    relativeWeight = []
    initialRuleWeight = []
    numberOfAntAttributes = 3
    
    def __init__(self, param, modelType, num_ref_value=3, isDebug=False):
        self.modelType = modelType
        self.param = param
        self.num_ref_value = num_ref_value
        self.isDebug = isDebug
    
    def setAttributes(self):
        self.relativeWeight.clear()
        self.initialRuleWeight.clear()
        
        ref_val_pos = len(self.consequentBeliefDegree) * len(self.consequentBeliefDegree[0])
        self.PM2_5_ref_val[1] = self.PM2_5_mid_val*self.param[ref_val_pos]
        self.PM10_ref_val[1] = self.PM10_mid_val*self.param[ref_val_pos+1]
        self.NH3_ref_val[1] = self.NH3_mid_val*self.param[ref_val_pos+2]
            
        relativeWeight_pos = ref_val_pos+self.numberOfAntAttributes
        for i in range(self.num_ref_value):
            self.relativeWeight.append(self.param[relativeWeight_pos+i])
            
        initialRuleWeight_pos = relativeWeight_pos+self.num_ref_value
        for i in range(len(self.consequentBeliefDegree)):
            self.initialRuleWeight.append(self.param[initialRuleWeight_pos+i])
            
    def initRuleBase(self):   
        consequentBeliefDegree = []
        if self.modelType == ModelType.DISJUNCTIVE_BRB:
            i = 0
            for _ in range(self.num_ref_value):
                rule_wise_belief_degree = []
                rule_wise_sum = np.sum(self.param[i:i+self.num_ref_value])
                for _ in range(self.num_ref_value):
                    rule_wise_belief_degree.append(self.param[i]/rule_wise_sum)
                    i += 1
                consequentBeliefDegree.append(rule_wise_belief_degree)
        
        elif self.modelType == ModelType.CONJUNCTIVE_BRB:
            i = 0
            for _ in range(self.num_ref_value ** self.numberOfAntAttributes):
                rule_wise_belief_degree = []
                rule_wise_sum = np.sum(self.param[i:i+self.num_ref_value])
                for _ in range(self.num_ref_value):
                    rule_wise_belief_degree.append(self.param[i]/rule_wise_sum)
                    i += 1
                consequentBeliefDegree.append(rule_wise_belief_degree)
                
        return np.array(consequentBeliefDegree)
    

    def runBRB(self, inputs):    
        self.consequentBeliefDegree = self.initRuleBase()
        self.setAttributes()
        transformed_input_PM2_5 = self.transformInput(inputs[0], self.PM2_5_ref_val)
        transformed_input_PM10 = self.transformInput(inputs[1], self.PM10_ref_val)
        transformed_input_NH3= self.transformInput(inputs[2], self.NH3_ref_val)
        
        self.printf("PM2_5_ref_val")
        self.printf(self.PM2_5_ref_val)
        
        self.printf("PM10_ref_val")
        self.printf(self.PM10_ref_val)
        
        self.printf("NH3_ref_val")
        self.printf(self.NH3_ref_val)
        
        self.printf("Input Transformation")
        self.printf(transformed_input_PM2_5)
        self.printf(transformed_input_PM10)
        self.printf(transformed_input_NH3)
        
        
        self.printf("\nRule Base")
        self.printf(self.consequentBeliefDegree)
        
        self.calculateActivationWeight(transformed_input_PM2_5, transformed_input_PM10, transformed_input_NH3)
        self.updateBeliefDegree(transformed_input_PM2_5, transformed_input_PM10, transformed_input_NH3)
        aqi = self.aggregateER()
        self.printf("AQI: "+ str(aqi))
        return aqi
    
    
    def printParameters(self):
        self.consequentBeliefDegree = self.initRuleBase()
        self.setAttributes()
        
        print("\nRule Base")
        print(self.consequentBeliefDegree)
        
        print("\nRule Weight")
        print(self.initialRuleWeight)
        
        print("\nRelative Weight")
        print(self.relativeWeight)
        
        print("\nPM2.5 Ref Values")
        print(self.PM2_5_ref_val)
        
        print("\nPM10 Ref Values")
        print(self.PM10_ref_val)
        
        print("\nNH3 Ref Values")
        print(self.NH3_ref_val)
        
        
    
    def printf(self, variable):
        if self.isDebug:
            print(variable)
            
            
    def transformInput(self, input_val, referential_values):            
        transformed_input = [0.0]*len(referential_values)
        
        if (input_val >= referential_values[0]): 
            transformed_input[0] = 1.0
    
        elif (input_val <= referential_values[-1]):
            transformed_input[-1] = 1.0
        else:
            for i in range(0, len(referential_values)-1):
                if (input_val == referential_values[i]): 
                    transformed_input[i] = 1.0
                    break
                    
                elif (input_val <= referential_values[i]) and (input_val >= referential_values[i+1]):
                    M = (referential_values[i] - input_val)/(referential_values[i] - referential_values[i+1])
                    transformed_input[i] = 1 - M
                    transformed_input[i+1] = M
                    break
    
        return transformed_input
    
    def calculateActivationWeight(self, transformed_input_PM2_5, transformed_input_PM10, transformed_input_NH3):
        self.matchingDegree.clear()
        self.activationWeight.clear()
        
        if self.modelType == ModelType.DISJUNCTIVE_BRB:
            for i in range(len(transformed_input_PM2_5)):
                self.matchingDegree.append(pow(transformed_input_PM2_5[i], self.relativeWeight[0]) + pow(transformed_input_PM10[i], self.relativeWeight[1]) + pow(transformed_input_NH3[i], self.relativeWeight[2]))

        elif self.modelType == ModelType.CONJUNCTIVE_BRB:
            for i in range(len(transformed_input_PM2_5)):
                for j in range(len(transformed_input_PM10)):
                  for k in range(len(transformed_input_NH3)):
                      self.matchingDegree.append(pow(transformed_input_PM2_5[i], self.relativeWeight[0]) * pow(transformed_input_PM10[j], self.relativeWeight[1]) * pow(transformed_input_NH3[k], self.relativeWeight[2]))
        
        self.printf("\nMatching degrees of the rules are as follow. ")
        self.printf(self.matchingDegree)
    
        sumMatchingDegree = np.sum(self.matchingDegree)
    
        for i in range(len(self.matchingDegree)):
            self.activationWeight.append(self.initialRuleWeight[i] * self.matchingDegree[i] / sumMatchingDegree)
    
        self.printf("\nActivation Weights of the rules are as follow. ")
        self.printf(self.activationWeight)
    
    
    def updateBeliefDegree(self, transformed_input_PM2_5, transformed_input_PM10, transformed_input_NH3):
        beliefDegreeChangeLevel = (np.sum(transformed_input_PM2_5) + np.sum(transformed_input_PM10) + np.sum(transformed_input_NH3))/self.numberOfAntAttributes
    
        if beliefDegreeChangeLevel<1:
            for x in range(0, self.consequentBeliefDegree.shape[0]):
                for y in range(0, self.consequentBeliefDegree.shape[1]):
                    self.consequentBeliefDegree[x][y] *= beliefDegreeChangeLevel
            
            # self.printf("Updated Consequent Belief Degree for change level : "+str(beliefDegreeChangeLevel))
            self.printf(self.consequentBeliefDegree)
    
        else:
            self.printf("\nNo update needed in belief degree")
    
    
    def aggregateER(self):
        ruleWiseBeliefDegreeSum = []
        part2 = 1
        partD = 1
    
        # calculate rule wise belief degree sum
        for x in range(0, self.consequentBeliefDegree.shape[0]):
            sum = 0
            for y in range(0, self.consequentBeliefDegree.shape[1]):
                sum = sum + self.consequentBeliefDegree[x, y]
    
            ruleWiseBeliefDegreeSum.append(sum)
    
        ruleWiseBeliefDegreeSum = np.array(ruleWiseBeliefDegreeSum)
    
        parts = [1 for _ in range(self.consequentBeliefDegree.shape[1])]
    
        # calculate all parts
        for j in range(len(ruleWiseBeliefDegreeSum)):
            rule_wise_sum = ruleWiseBeliefDegreeSum[j]
            activation_weight = self.activationWeight[j]
    
            for k in range(self.consequentBeliefDegree.shape[1]):
                section_2 = 1 - (activation_weight * rule_wise_sum)
    
                parts[k] *= float(activation_weight * self.consequentBeliefDegree[j, k]+section_2)
    
    
        for k in range(len(self.activationWeight)):
            partD *= 1 - self.activationWeight[k]
            part2 *= 1 - self.activationWeight[k]*ruleWiseBeliefDegreeSum[k]
    
    
        value = np.sum(parts) - (self.numberOfAntAttributes-1)*part2
        meu = 1 / value 
    
    
        numerators = []
        for i in range(len(parts)):
            numerators.append(float(meu * (parts[i] - part2)))
    
        denominator = 1 - meu * partD
    
        # perform aggregration
        aggregatedBeliefDegree = []
        for i in range(len(numerators)):
            aggregatedBeliefDegree.append(float(numerators[i]/denominator))
        
        
        #calculate crisp value
        aggregatedBeliefDegreeSum = np.sum(aggregatedBeliefDegree)
    
        if aggregatedBeliefDegreeSum == 1:
          return self.calculateCrispValue(aggregatedBeliefDegree)
    
        else:
            degreeOfIncompleteness = 1 - aggregatedBeliefDegreeSum
     
            utilityMax, utilityMin = 0, 0
            for i in range(len(aggregatedBeliefDegree)):
                if(i == 0):
                      utilityMax += (aggregatedBeliefDegree[i]+degreeOfIncompleteness)*self.utilityScore[i]
                      utilityMin += (aggregatedBeliefDegree[i])*self.utilityScore[i]
                elif(i == len(aggregatedBeliefDegree)-1):
                      utilityMin += (aggregatedBeliefDegree[i]+degreeOfIncompleteness)*self.utilityScore[i]
                      utilityMax += (aggregatedBeliefDegree[i])*self.utilityScore[i]
                else:
                      utilityMax += (aggregatedBeliefDegree[i])*self.utilityScore[i]
                      utilityMin += (aggregatedBeliefDegree[i])*self.utilityScore[i]
     
            utilityAvg = (utilityMax + utilityMin)/2
    
            revisedBeliefDegree = []
            for belief_degree in aggregatedBeliefDegree:
                revisedBeliefDegree.append(belief_degree/aggregatedBeliefDegreeSum)
              
            aqi = self.calculateAQI(revisedBeliefDegree)
            crisp_value = self.calculateCrispValue(revisedBeliefDegree)
            
            return aqi
    
    def calculateCrispValue(self, aggregatedBeliefDegree):
          crisp_value = 0
          for i in range(len(aggregatedBeliefDegree)):
              crisp_value += aggregatedBeliefDegree[i]*self.utilityScore[i]
    
          return crisp_value
    
    
    def calculateAQI(self, aggregatedBeliefDegree):
        aqi = 0
        
        if (aggregatedBeliefDegree[0] > aggregatedBeliefDegree[1]) and (aggregatedBeliefDegree[0] >  aggregatedBeliefDegree[2]):
            aqi = (201 + 299*aggregatedBeliefDegree[0]) + ((200* aggregatedBeliefDegree[1])/2)
            
        elif (aggregatedBeliefDegree[2] >  aggregatedBeliefDegree[1]) and (aggregatedBeliefDegree[2] > aggregatedBeliefDegree[0]): 
            aqi = (100*(1 - aggregatedBeliefDegree[2])) + ((200* aggregatedBeliefDegree[1])/2) 
    
        elif ( aggregatedBeliefDegree[1] > aggregatedBeliefDegree[0]) and ( aggregatedBeliefDegree[1] > aggregatedBeliefDegree[2]):
            if aggregatedBeliefDegree[0] > aggregatedBeliefDegree[2]:
                aqi = (101 + 99* aggregatedBeliefDegree[1]) + ((500*aggregatedBeliefDegree[0])/2)
      
            elif (aggregatedBeliefDegree[2] > aggregatedBeliefDegree[0]):   
                aqi = (101 + 99* aggregatedBeliefDegree[1]) + ((100*aggregatedBeliefDegree[2])/2)
                
        return aqi
    
    