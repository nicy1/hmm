import pandas as pd
import numpy as np
import csv
from collections import OrderedDict

# ==========================================================================



class read:
    def __init__ (self, filename=None):

        data = pd.read_csv(open(filename))
        target = data['Protocol']
        X = data.drop('Protocol', axis=1)
        
        # Convert field 'Protocol'into float
        index_table = {'TCP':0, 'TLSv1.2':1}
        Y = []
        for t in target:
          Y.append(index_table[t])
              
        # Splitting file in train_set (80%) and test_set (20%) 
        x_train, x_test = np.split(X, [int(0.8 * len(X))])
        self.y_train, self.y_test = np.split(Y, [int(0.8 * len(Y))])
   
        # Save train and test set
        pd.DataFrame(x_train).to_csv("x_train.csv",encoding='utf-8',index=False)
        pd.DataFrame(x_test).to_csv("x_test.csv",encoding='utf-8',index=False)


        self._learn_init_prob()    # Learn initial probabilities, train and test data
     
    # ----------------------------------------------------------------------    

    def _learn_init_prob(self):
        
        self.x_train = []
        self.x_test = [] 
        self.lookUpTable = {}  
        
        # Train data
        csv_reader = csv.DictReader(open("x_train.csv", mode='r'))
        for row in csv_reader:
          obs = str(row['Length'])
          info = str(row['Info'])
          if 'ACK]' in info:
             info = info[:info.index('ACK]')+len('ACK]')]
          elif '[SYN]' in info:
             info = info[:info.index('[SYN]')+len('[SYN]')]

          obs = obs + '_' + info
          if obs not in self.lookUpTable:
            self.lookUpTable[obs] = float(len(self.lookUpTable))
          obs = self.lookUpTable[obs]
          # Add training data
          self.x_train.append([obs]) 


        # Test data
        csv_reader = csv.DictReader(open("x_test.csv", mode='r'))
        for row in csv_reader:
          obs = str(row['Length'])
          info = str(row['Info'])
          if 'ACK]' in info:
             info = info[:info.index('ACK]')+len('ACK]')]
          elif '[SYN]' in info:
             info = info[:info.index('[SYN]')+len('[SYN]')]

          obs = obs + '_' + info
          if obs not in self.lookUpTable:
            self.lookUpTable[obs] = float(len(self.lookUpTable))
          obs = self.lookUpTable[obs]
          # Add testing data
          self.x_test.append([obs])

    # ---------------------------------------------------------------------- 
 
    # Return HMM parameters, train and test
    def get_data(self):
      return (self.x_train,self.y_train,self.x_test,self.y_test)
       
  
   
# ==========================================================================





















