import sys
import os

import json
import csv

import pandas as pd
import numpy as np
from collections import OrderedDict



# ==========================================================================



class read:
    def __init__ (self, filename):
        data = csv.DictReader(open(filename))
        data = list(data)
        # Splitting file in train_set (80%) and test_set (20%) 
        size = int(len(data) * 0.8)
        self.train_set, self.test_set = data[0:size], data[size:len(data)]

        self._learn_init_prob()    # Learn initial probabilities, train and test data
     
    # ----------------------------------------------------------------------    

    def _learn_init_prob(self):
        self.x_lookUpTable = {}
        self.y_lookUpTable = {}       

        # Train data
        self.x_train, self.y_train = self._normalize_data(self.train_set)
        # Test data
        self.x_test, self.y_test = self._normalize_data(self.test_set)

    # ---------------------------------------------------------------------- 

    def _normalize_data(self, dataset):
        x_dataset = []  
        y_dataset = []    
 
        for i, pkt in enumerate(dataset):
            x = str(pkt['Length']) + '-' + str(pkt['Ratio'])        
            if x not in self.x_lookUpTable:
               self.x_lookUpTable[x] = float(len(self.x_lookUpTable))
            x = self.x_lookUpTable[x]
            x_dataset.append(x)

            y = int(pkt['ActionType'])
            if y not in self.y_lookUpTable:
               self.y_lookUpTable[y] = float(len(self.y_lookUpTable))
            y = self.y_lookUpTable[y]
            y_dataset.append(y)       

        return x_dataset, y_dataset


    # ----------------------------------------------------------------------
 
    # Return HMM parameters, train_data and test_data
    def get_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test




   
# ==========================================================================


























