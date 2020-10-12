import sys
import os

import json
import csv

import pandas as pd
import numpy as np
from collections import OrderedDict



# ==========================================================================


class read:
    def __init__ (self, filename=None):

        #data = pd.read_csv(filename)
        data = csv.DictReader(open(filename))
        data = list(data)
        # Splitting file in train_set (80%) and test_set (20%) 
        size = int(len(data) * 0.8)
        self.train_set, self.test_set = data[0:size], data[size:len(data)]

        self._learn_init_prob()    # Learn initial probabilities, train and test data
     
    # ----------------------------------------------------------------------    

    def _learn_init_prob(self):
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []          

        self.obs_lookUpTable = {}
        
        # Train data
        for i, pkt in enumerate(self.train_set):
          obs = pkt['Source']+'-'+pkt['Destination']+'-'+pkt['Length']
          
          tcp_payload_len = self._get_state(pkt['Info'])
          if tcp_payload_len != 0:                  # Only packet with payload
            if obs not in self.obs_lookUpTable:
               self.obs_lookUpTable[obs] = len(self.obs_lookUpTable)
            obs = self.obs_lookUpTable[obs]
            
            self.x_train.append([obs])
            self.y_train.append(tcp_payload_len)


        # Test data
        for i, pkt in enumerate(self.test_set):
          obs = pkt['Source']+'-'+pkt['Destination']+'-'+pkt['Length']

          tcp_payload_len = self._get_state(pkt['Info'])
          if tcp_payload_len != 0:                    # Only packet with
            if obs not in self.obs_lookUpTable:
               self.obs_lookUpTable[obs] = len(self.obs_lookUpTable)
            obs = self.obs_lookUpTable[obs]

            self.x_test.append([obs])
            self.y_test.append(tcp_payload_len)

    # ---------------------------------------------------------------------- 
 
    # Return HMM parameters, train_data and test_data
    def get_data(self):
      return (self.x_train, self.y_train, self.x_test, self.y_test)
       
    # ----------------------------------------------------------------------
    
    # Compute payload length (amount of bytes transfered)
    def _get_state(self, info):
      for tmp in info.split():
        if 'Len' in tmp:
          return int(tmp[tmp.index('=')+len('='):])

   
# ==========================================================================













