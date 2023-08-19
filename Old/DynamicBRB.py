#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 15:30:03 2023

@author: snzisad
"""

import numpy as np
from enum import Enum
 
class ModelType(Enum):
    DISJUNCTIVE_BRB = 1
    CONJUNCTIVE_BRB = 2
 

class BRBModel:
    ant_attr_referential_values = []
    consequentBeliefDegree = np.array([])
    utilityScore = []
    matchingDegree = []
    activationWeight = []
    relativeWeight = []
    initialRuleWeight = []
    transformed_input_list = []
    
    def __init__(self, modelType, num_ant_attr, num_ref_values, isDebug=False):
        self.modelType = modelType
        self.num_ant_attr = num_ant_attr
        self.num_ref_values = num_ref_values
        self.isDebug = isDebug

    """
    Set referential values for all antecident attributes. 
    Here, ref_value_list is a 2D array with a shape of num_ant_attr*num_ref_values
    """
    def setRefValues(self, ref_value_list):
        self.ant_attr_referential_values.clear()

        if len(ref_value_list) != self.num_ant_attr or len(ref_value_list[0]) != self.num_ref_values:
            raise Exception('ref_value_list should be a 2D array with a shape of num_ant_attr*num_ref_values')

        self.ant_attr_referential_values.extend(ref_value_list)
    
    """
    Set initial relative weight and rule weight. 
    Here, relative_weight_list is a 1D array with a size equal to num_ref_values. 
    rule_weight_list is also a 1D array. it's size would be equal to num_ref_values for disjunctive brb and num_ref_values**num_ant_attr for conjunctive brb
    """
    def setIntitalWeights(self, relative_weight_list, rule_weight_list):
        self.relativeWeight.clear()
        self.initialRuleWeight.clear()

        if len(relative_weight_list) != self.num_ant_attr:
            raise Exception('relative_weight_list should be a 1D array with a size euqal to num_ant_attr')
        if self.modelType == ModelType.DISJUNCTIVE_BRB and len(rule_weight_list) != self.num_ref_values:
            raise Exception('rule_weight_list should be a 1D array with a size euqal to num_ref_values')
        elif self.modelType == ModelType.CONJUNCTIVE_BRB and len(rule_weight_list) != self.num_ref_values**self.num_ant_attr:
            raise Exception('rule_weight_list should be a 1D array with a size euqal to num_ref_values**num_ant_attr')

        self.relativeWeight.extend(relative_weight_list)
        self.initialRuleWeight.extend(rule_weight_list)
    
    """
    Set utility score
    Here, utility_score_list is a 1D array with a size equal to num_ref_values
    """
    def setUtilityScore(self, utility_score_list):
        self.utilityScore.clear()

        if len(utility_score_list) != self.num_ref_values:
            raise Exception('utility_score_list should be a 1D array with a size euqal to num_ref_values')

        self.utilityScore.extend(utility_score_list)


    """
    Set initial rule base
    Here,  rule_base_list is also a 1D array. it's size would be equal to num_ref_values*num_ref_values for disjunctive brb 
    and (num_ref_values**num_ant_attr)*num_ref_values for conjunctive brb
    """
    def setRuleBase(self, rule_base_list):  
        consequentBeliefDegrees = []
        
        if self.modelType == ModelType.DISJUNCTIVE_BRB:
            if len(rule_base_list) != self.num_ref_values*self.num_ref_values:
                raise Exception('rule_base_list should be a 1D array with a size euqal to num_ref_values*num_ref_values')

            i = 0
            for _ in range(self.num_ref_values):
                rule_wise_belief_degree = []
                rule_wise_sum = np.sum(rule_base_list[i:i+self.num_ref_values])
                for _ in range(self.num_ref_values):
                    rule_wise_belief_degree.append(rule_base_list[i]/rule_wise_sum)
                    i += 1
                consequentBeliefDegrees.append(rule_wise_belief_degree)
        
        elif self.modelType == ModelType.CONJUNCTIVE_BRB:
            if len(rule_base_list) != (self.num_ref_values**self.num_ant_attr)*self.num_ref_values:
                raise Exception('rule_base_list should be a 1D array with a size euqal to (num_ref_values**num_ant_attr)*num_ref_values')

            i = 0
            for _ in range(self.num_ref_values ** self.num_ant_attr):
                rule_wise_belief_degree = []
                rule_wise_sum = np.sum(rule_base_list[i:i+self.num_ref_values])
                for _ in range(self.num_ref_values):
                    rule_wise_belief_degree.append(rule_base_list[i]/rule_wise_sum)
                    i += 1
                consequentBeliefDegrees.append(rule_wise_belief_degree)
                
        self.consequentBeliefDegree = np.array(consequentBeliefDegrees)
    

    def runBRB(self, input_list):    
        if len(input_list) != self.num_ant_attr:
            raise Exception('input_list should be a 1D array with a size euqal to num_ant_attr')

        self.transformed_input_list.clear()

        for i in range(self.num_ant_attr):
            self.transformed_input_list.append(self.transformInput(input_list[i], self.ant_attr_referential_values[i]))
        
        self.printf("Initial Rule Base")
        for rule_base in self.consequentBeliefDegree:
            self.printf(rule_base)
        
        self.printf("\nTransformed Input")
        for t_input in self.transformed_input_list:
            self.printf(t_input)
                
        self.calculateActivationWeight()
        self.updateBeliefDegree()
        crisp_value = self.aggregateER()
        return crisp_value
    
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
    
    def calculateActivationWeight(self):
        self.matchingDegree.clear()
        self.activationWeight.clear()
        
        if self.modelType == ModelType.DISJUNCTIVE_BRB:
            for i in range(self.num_ref_values):
                self.matchingDegree.append(np.sum(self.transformed_input_list[k][i]**self.relativeWeight[k] for k in range(self.num_ant_attr)))

        elif self.modelType == ModelType.CONJUNCTIVE_BRB:
            self.matchingDegree.extend([1]*(self.num_ref_values**self.num_ant_attr))

            for j in range(self.num_ant_attr):
                temp_matchingDegree = []
                for i in range(self.num_ref_values):
                    temp_matchingDegree.extend([self.transformed_input_list[j][i]**self.relativeWeight[j]] * pow(self.num_ref_values, self.num_ant_attr-j-1))
                    
                for k in range(len(self.matchingDegree)):
                    self.matchingDegree[k] *= temp_matchingDegree[k%len(temp_matchingDegree)]
    
        sumMatchingDegree = np.sum(self.matchingDegree)
    
        for i in range(len(self.matchingDegree)):
            self.activationWeight.append(self.initialRuleWeight[i] * self.matchingDegree[i] / sumMatchingDegree)
    
        self.printf("\nMatching Degrees")
        self.printf(self.matchingDegree)
        
        self.printf("\nActivation Weights ")
        self.printf(self.activationWeight)
    
    
    def updateBeliefDegree(self):
        beliefDegreeChangeLevel = np.sum(self.transformed_input_list)/self.num_ant_attr
    
        if beliefDegreeChangeLevel<1:
            for x in range(0, self.consequentBeliefDegree.shape[0]):
                for y in range(0, self.consequentBeliefDegree.shape[1]):
                    self.consequentBeliefDegree[x][y] *= beliefDegreeChangeLevel
            
            self.printf("Updated Consequent Belief Degree for change level : "+str(beliefDegreeChangeLevel))
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
    
    
        value = np.sum(parts) - (self.num_ant_attr-1)*part2
        meu = 1 / value 
    
    
        numerators = []
        for i in range(len(parts)):
            numerators.append(float(meu * (parts[i] - part2)))
    
        denominator = 1 - meu * partD
    
        # perform aggregration
        aggregatedBeliefDegree = []
        for i in range(len(numerators)):
            aggregatedBeliefDegree.append(float(numerators[i]/denominator))
        
        
        self.printf("\nER Aggregated Belief Degree")
        self.printf(aggregatedBeliefDegree)
        
        #calculate crisp value
        aggregatedBeliefDegreeSum = 0
        for belief_degree in aggregatedBeliefDegree:
            aggregatedBeliefDegreeSum += belief_degree
    
        if aggregatedBeliefDegreeSum == 1:
          return self.calculateCrispValue(aggregatedBeliefDegree)
    
        else:
          degreeOfIncompleteness = 1 - aggregatedBeliefDegreeSum
          self.printf("\nUnassigned degree of belief: "+str(degreeOfIncompleteness))
    
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
          
          
          self.printf("\nMaximum expected utility: "+str(utilityMax))
          self.printf("Minimum expected utility: "+str(utilityMin))
          self.printf("Average expected utility: "+str(utilityAvg))
    
          revisedBeliefDegree = []
          for belief_degree in aggregatedBeliefDegree:
              revisedBeliefDegree.append(belief_degree/aggregatedBeliefDegreeSum)
    
          self.printf("\nRevised ER Aggregated Belief Degree")
          self.printf(revisedBeliefDegree)
            
          crisp_value = self.calculateCrispValue(revisedBeliefDegree)
          
          return crisp_value
    
    def calculateCrispValue(self, aggregatedBeliefDegree):
          crisp_value = 0
          for i in range(len(aggregatedBeliefDegree)):
              crisp_value += aggregatedBeliefDegree[i]*self.utilityScore[i]
    
          return crisp_value    