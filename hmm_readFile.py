import sys
import os

import json
import csv

import pandas as pd
from sklearn.model_selection import train_test_split
from collections import OrderedDict

# ==========================================================================

class readFile:
    def __init__ (self, filename=None):
        """This class accepts only csv and json files"""
       
        data = pd.read_csv(open(filename))
        # Splitting file in train_file and test_file
        Y = data['Caesarian']                          # Target values (dependent variables)
        X = data.drop('Caesarian', axis=1)             # Independent features
        X = X.drop('Age', axis=1)
        
        # Train_data: 80%, Test_data: 20%
        x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2, random_state=0)   

        # Save Train and Test data
        pd.DataFrame(x_train).to_csv("x_train.csv",encoding='utf-8',index=False)     # Train_data observ.
        pd.DataFrame(y_train).to_csv("y_train.csv",encoding='utf-8',index=False)     # Train_data states
        pd.DataFrame(x_test).to_csv("x_test.csv",encoding='utf-8',index=False)       # Test_data observ.
        pd.DataFrame(y_test).to_csv("y_test.csv",encoding='utf-8',index=False)       # Test_data states

    # ----------------------------------------------------------------------------------------    
  
    def get_parameters(self):
        '''
        Return HMM parameters, train_data and test_data
        '''
        # Put 'y_train' states in vector 'y_train'
        y_train = []
        csv_reader_y = csv.reader(open("y_train.csv", mode='r'))
        for i,row in enumerate(csv_reader_y):
          if i != 0:
            y_train.append(row[0])
        
        # Put 'x_train' obs in vector 'train_data'
        train_data = []
        compute_transition_prob = {}
        compute_emission_prob = {}

        csv_reader_x = csv.DictReader(open("x_train.csv", mode='r'))
        rows = list(csv_reader_x)
        length_x = len(rows)
        
        for i,row in enumerate(rows):
          obs_seq = [row['Delivery_number'],row['Delivery_time'],row['Blood_of_Pressure'],row['Heart_Problem']] 
          obs_seq = readFile.look_up_table (self, obs_seq)          # Check if the data is null and convert it 
          obs_seq = readFile.normalize(self, obs_seq)               # Normalize data

          train_data.append(obs_seq)     
          state = y_train[i]                                  # Get the state in row i
          state = readFile.normalize(self, state).upper()     # Normalize data

          if state not in compute_emission_prob.keys():
             compute_emission_prob[state] = {}
          for obs in obs_seq:
              if obs not in compute_emission_prob[state]:
                 compute_emission_prob[state][obs] = 0
              compute_emission_prob[state][obs] += 1
                  
          if i == length_x-1:                # EOF
             break
          
          if state not in compute_transition_prob.keys():
            compute_transition_prob[state] = {}
          next_state = y_train[i+1].upper()                 # Next state

          if next_state not in compute_transition_prob[state]:
            compute_transition_prob[state][next_state] = 0
          compute_transition_prob[state][next_state] += 1
          
          
        # Retrieve test_data
        test_data = []
        csv_reader_x = csv.DictReader(open("x_test.csv", mode='r'))

        for i,row in enumerate(csv_reader_x):
          obs_seq = [row['Delivery_number'].upper(),row['Delivery_time'].upper(), \
                    row['Blood_of_Pressure'].upper(),row['Heart_Problem'].upper()] 
          obs_seq = readFile.look_up_table (self, obs_seq)             # Check if the data is null and convert it 
          obs_seq = readFile.normalize(self, obs_seq)                  # Normalize data
          test_data.append(obs_seq)
        
        return (compute_transition_prob,compute_emission_prob,train_data,test_data)
             
    # ----------------------------------------------------------------------

    def look_up_table(self, words):                # Gestion of a null value
        w = []
        for word in words:
          if word == '':
            w.append('no_value')
          else: 
            w.append(word)
        return w

    # ----------------------------------------------------------------------

    def to_lowercase(self, words):
        """Convert all characters to lowercase from list of words"""
        new_words = ''
        for word in words:
            new_words +=word.lower()
        return new_words

    # ----------------------------------------------------------------------

    def replace_numbers(self, words):
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

    def normalize (self, words):
      if isinstance(words, list):
        w = []
        for word in words:
          word = readFile.to_lowercase(self, word)
          word = readFile.replace_numbers(self, word)
          w.append(word)
        return w
      else:
        words = readFile.to_lowercase(self, words)
        words = readFile.replace_numbers(self, words)
        return words
   
# ==========================================================================





















