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

        self.training_file = 'train.csv' 
        self.testing_file = 'test.csv'    

        # Train data
        headers, rows = self._normalize_data(self.train_set)
        csvfile = open(self.training_file, 'w')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(rows)
        # Test data
        headers, rows = self._normalize_data(self.test_set)
        csvfile = open(self.testing_file, 'w')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headers)
        csvwriter.writerows(rows)
        

    # ---------------------------------------------------------------------- 

    def _normalize_data(self, dataset):
      headers = ['Features', 'Labels']
      rows = []      
 
      for i, pkt in enumerate(dataset):
          x = str(pkt['Length'])+'-'+str(pkt['Ratio'])        
          if x not in self.x_lookUpTable:
             self.x_lookUpTable[x] = float(len(self.x_lookUpTable))
          x = self.x_lookUpTable[x]

          y = int(pkt['ActionType'])
          if y not in self.y_lookUpTable:
             self.y_lookUpTable[y] = float(len(self.y_lookUpTable))
          y = self.y_lookUpTable[y]
          rows.append([x,y])       

      return headers, rows


    # ----------------------------------------------------------------------
 
    # Return HMM parameters, train_data and test_data
    def get_data(self):
      return self.training_file, self.testing_file


   
# ==========================================================================


























