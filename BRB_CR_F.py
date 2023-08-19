import numpy as np
from enum import Enum
from BRBModel import ModelType

class ParamType(Enum):
    SCALING_RATIO = 1
    CROSSOVER_RATE = 2
 


class BRB_CR_F:    
    first_ref_val = [1.0, 0.5, 0.0]
    second_ref_val = [2.0, 1.0, 0.0]
    
    utilityScore = [1, 0.5, 0]
    referencesValues_text = ["Big", "Medium", "Small"]
    matchingDegree = []
    activationWeight = []
    numberOfAntAttributes = 2
    
    def __init__(self, modelType, paramType):
        if modelType == ModelType.DISJUNCTIVE_BRB:
            self.param = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        else:
            self.param = [1, 0, 0, 0.5, 0.5, 0, 0, 1, 0, 0.5, 0.5, 0, 0, 1, 0, 0, 0.5, 0.5, 0, 1, 0, 0, 0.5, 0.5, 0, 0, 1]
        
        if paramType == ParamType.SCALING_RATIO:
            self.d22_ref_val = [2.0, 1.0, 0.0]
        else:
            self.d22_ref_val = [1.0, 0.5, 0.0]

        self.isDebug = False
        self.modelType = modelType
        
    def initRuleBase(self):   
        consequentBeliefDegree = []
        if self.modelType == ModelType.DISJUNCTIVE_BRB:
            consequentBeliefDegree.extend([
                [self.param[0]/np.sum(self.param[0:3]), self.param[1]/np.sum(self.param[0:3]), self.param[2]/np.sum(self.param[0:3])],
                [self.param[3]/np.sum(self.param[3:6]), self.param[4]/np.sum(self.param[3:6]), self.param[5]/np.sum(self.param[3:6])],
                [self.param[6]/np.sum(self.param[6:9]), self.param[7]/np.sum(self.param[6:9]), self.param[8]/np.sum(self.param[6:9])]
            ])
        
        elif self.modelType == ModelType.CONJUNCTIVE_BRB:
            consequentBeliefDegree.extend([
                [self.param[0]/np.sum(self.param[0:3]), self.param[1]/np.sum(self.param[0:3]), self.param[2]/np.sum(self.param[0:3])],
                [self.param[3]/np.sum(self.param[3:6]), self.param[4]/np.sum(self.param[3:6]), self.param[5]/np.sum(self.param[3:6])],
                [self.param[6]/np.sum(self.param[6:9]), self.param[7]/np.sum(self.param[6:9]), self.param[8]/np.sum(self.param[6:9])],
                [self.param[9]/np.sum(self.param[9:12]), self.param[10]/np.sum(self.param[9:12]), self.param[11]/np.sum(self.param[9:12])],
                [self.param[12]/np.sum(self.param[12:15]), self.param[13]/np.sum(self.param[12:15]), self.param[14]/np.sum(self.param[12:15])],
                [self.param[15]/np.sum(self.param[15:18]), self.param[16]/np.sum(self.param[15:18]), self.param[17]/np.sum(self.param[15:18])],
                [self.param[18]/np.sum(self.param[18:21]), self.param[19]/np.sum(self.param[18:21]), self.param[20]/np.sum(self.param[18:21])],
                [self.param[21]/np.sum(self.param[21:24]), self.param[22]/np.sum(self.param[21:24]), self.param[23]/np.sum(self.param[21:24])],
                [self.param[24]/np.sum(self.param[24:27]), self.param[25]/np.sum(self.param[24:27]), self.param[26]/np.sum(self.param[24:27])],
            ])
        return np.array(consequentBeliefDegree)


    def runBRB(self, inputs):
        transformed_input_first = self.transformInput(inputs[0], self.first_ref_val)
        transformed_input_second = self.transformInput(inputs[1], self.second_ref_val)
                
        self.consequentBeliefDegree = self.initRuleBase()
        
        self.calculateActivationWeight(transformed_input_first, transformed_input_second)
        self.updateBeliefDegree(transformed_input_first, transformed_input_second)
        crisp = self.aggregateER()
        return crisp
    
    
    
    def printf(self, variable):
        if self.isDebug:
            print(variable)
            
            
    def transformInput(self, input_val, referential_values):     
        if (input_val >= referential_values[0]): 
            H = 1 
            M = 0
            L = 0
    
        elif (input_val == referential_values[1]):
            H = 0 
            M = 1
            L = 0
    
        elif (input_val <= referential_values[2]):
            H = 0
            M = 0
            L = 1
           
        elif (input_val <= referential_values[0]) and (input_val >= referential_values[1]):
            M = (referential_values[0] - input_val)/(referential_values[0] - referential_values[1])
            H = 1 - M
            L = 0.0 
    
        elif (input_val <= referential_values[1]) and (input_val >= referential_values[2]):
            L = (referential_values[1] - input_val)/(referential_values[1] - referential_values[2])
            M = 1 - L  
            H = 0.0
    
        return [H, M, L]
    
    def calculateActivationWeight(self, transformed_input_first, transformed_input_second):
        self.matchingDegree.clear()
        self.activationWeight.clear()
        
        if self.modelType == ModelType.DISJUNCTIVE_BRB:
            for i in range(len(transformed_input_first)):
                self.matchingDegree.append(pow(transformed_input_first[i], 1) + pow(transformed_input_second[i], 1))

        elif self.modelType == ModelType.CONJUNCTIVE_BRB:
            for i in range(len(transformed_input_first)):
                for j in range(len(transformed_input_second)):
                    self.matchingDegree.append(pow(transformed_input_first[i], 1) * pow(transformed_input_second[j], 1))
        
        self.printf("\nMatching degrees of the rules are as follow. ")
        self.printf(self.matchingDegree)
    
        sumMatchingDegree = np.sum(self.matchingDegree)
    
        for i in range(len(self.matchingDegree)):
            self.activationWeight.append(1 * self.matchingDegree[i] / sumMatchingDegree)
    
        self.printf("\nActivation Weights of the rules are as follow. ")
        self.printf(self.activationWeight)
    
    
    def updateBeliefDegree(self, transformed_input_first, transformed_input_second):
        beliefDegreeChangeLevel = (np.sum(transformed_input_first) + np.sum(transformed_input_second))/self.numberOfAntAttributes
    
        if beliefDegreeChangeLevel<1:
            for x in range(0, self.consequentBeliefDegree.shape[0]):
                for y in range(0, self.consequentBeliefDegree.shape[1]):
                    self.consequentBeliefDegree[x][y] *= beliefDegreeChangeLevel
            
            self.printf(self, "Updated Consequent Belief Degree for change level : ", beliefDegreeChangeLevel)
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
        aggregatedBeliefDegreeSum = 0
        for belief_degree in aggregatedBeliefDegree:
            aggregatedBeliefDegreeSum += belief_degree
    
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
    
          crisp_value = self.calculateCrispValue(revisedBeliefDegree)
          
          return crisp_value
    
    def calculateCrispValue(self, aggregatedBeliefDegree):
          crisp_value = 0
          for i in range(len(aggregatedBeliefDegree)):
              crisp_value += aggregatedBeliefDegree[i]*self.utilityScore[i]
    
          return crisp_value
    