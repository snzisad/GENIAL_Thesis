from BRBModel import BRBModel, ModelType
import numpy as np
import pandas as pd

class BRBJOPS(BRBModel):
        
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
        
        # For non trained
        # utilityScore_pos = initialRuleWeight_pos+len(self.consequentBeliefDegree)
        # For trained
        utilityScore_pos = initialRuleWeight_pos+self.num_ref_value
        
        for i in range(self.num_ref_value):
            self.utilityScore.append(self.param[utilityScore_pos+i])
            
            
            
            
            



