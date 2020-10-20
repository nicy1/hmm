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
        
        # Save train and test set
        #pd.DataFrame(train).to_csv("train.csv",encoding='utf-8',index=False)
        #pd.DataFrame(test).to_csv("test.csv",encoding='utf-8',index=False)

        self._learn_init_prob()    # Learn initial probabilities, train and test data
     
    # ----------------------------------------------------------------------    

    def _learn_init_prob(self):
        
        self.states = []
        self.symbols = []
        self.train_data = []
        self.test_data = []
        self.targets = []           # Hidden states (target for prediction)

        self.trans_prob = {}
        self.emis_prob = {}
        self.obs_lookUpTable = {}
        
        # Train data
        for i, pkt in enumerate(self.train_set):
            obs = str(pkt['Length'])
            if obs not in self.obs_lookUpTable:
               self.obs_lookUpTable[obs] = 'obs'+str(len(self.obs_lookUpTable))
            obs = self.obs_lookUpTable[obs]
            #obs = self._normalize(obs)          # Normalize data 

            from_state = str(pkt['ActionType'])
            # Add training data, state and symbol
            self.train_data.append(obs) 
            self.states.append(from_state)
            self.symbols.append(obs)
          
            # Emission probabilities
            if from_state not in self.emis_prob.keys():
               self.emis_prob[from_state] = {}
            if obs not in self.emis_prob[from_state]:
               self.emis_prob[from_state][obs] = 0
            self.emis_prob[from_state][obs] += 1

            if i >= len(list(self.train_set))-1:     # EOF (no transiion)
               break

            # Transition probabilities
            if from_state not in self.trans_prob.keys():
               self.trans_prob[from_state] = {}
            to_state = str(list(self.train_set)[i+1]['ActionType'])   
            if to_state not in self.trans_prob[from_state]:
               self.trans_prob[from_state][to_state] = 0
            self.trans_prob[from_state][to_state] += 1


        # Test data
        for i, pkt in enumerate(self.test_set):
            obs = str(pkt['Length'])
            if obs not in self.obs_lookUpTable:
               self.obs_lookUpTable[obs] = 'obs'+str(len(self.obs_lookUpTable))
            obs = self.obs_lookUpTable[obs]
            #obs = self._normalize(obs)          # Normalize data

            state = str(pkt['ActionType'])
            # Add testing data, state and symbol
            self.test_data.append(obs)
            self.targets.append(state)           # Hidden states (target for prediction)
            self.states.append(state)
            self.symbols.append(obs)

        self.states = set(self.states)         # Eliminate duplicate
        self.symbols = set(self.symbols)

    # ---------------------------------------------------------------------- 
 
    # Return HMM parameters, train_data and test_data
    def get_data(self):
      return (self.states, self.symbols, self.trans_prob, self.emis_prob, \
              self.train_data, self.test_data, self.targets)
    
    # ----------------------------------------------------------------------

    def _to_lowercase(self, words):
        """Convert all characters to lowercase from list of words"""
        new_words = ''
        for word in words:
            new_words +=word.lower()
        return new_words

    # ----------------------------------------------------------------------

    def _to_uppercase(self, words):
        """Convert all characters to uppercase from list of words"""
        new_words = ''
        for word in words:
            new_words +=word.upper()
        return new_words

    # ----------------------------------------------------------------------

    def _replace_numbers(self, words):
        """Replace all interger occurrences in list of words"""

        number_to_word = {'0':'zero', '1':'one', '2':'two', '3':'three', '4':'four', \
                          '5':'five', '6':'six', '7':'seven', '8':'eight', '9':'nine'} 
        new_words = ''
        for word in words:
            if word.isdigit():
               new_word = number_to_word[word]
               new_words += new_word
            else:
               new_words += word
        return new_words

    # ----------------------------------------------------------------------

    def _normalize (self, words):
      if isinstance(words, list):
        w = []
        for word in words:
          word = self._to_lowercase(word)
          word = self._replace_numbers(word)
          w.append(word)
        return w
      else:
        words = self._to_lowercase(words)
        words = self._replace_numbers(words)
        return words


   
# ==========================================================================





















