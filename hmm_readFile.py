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
       
        data = pd.read_csv(open(filename))
        # Splitting file in train_file and test_file
        Y = data['quality']                          # Target values (dependent variables)
        X = data.drop('quality', axis=1)         
        
        # Train_data: 70%, Test_data: 30%
        x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.3, random_state=0)   

        # Save Train and Test data
        pd.DataFrame(x_train).to_csv("x_train.csv",encoding='utf-8',index=False)     # Train_data observ.
        pd.DataFrame(y_train).to_csv("y_train.csv",encoding='utf-8',index=False)     # Train_data states
        pd.DataFrame(x_test).to_csv("x_test.csv",encoding='utf-8',index=False)       # Test_data observ.
        pd.DataFrame(y_test).to_csv("y_test.csv",encoding='utf-8',index=False)       # Test_data states

    # ----------------------------------------------------------------------    
    
    # Return HMM parameters, train_data and test_data
    def get_parameters(self):
        # Put 'y_test' states in vector 'y_test'
        y_test = []
        csv_reader_y = csv.reader(open("y_test.csv", mode='r'))
        for i,row in enumerate(csv_reader_y):
          if i != 0:
            y_test.append(row[0].upper())

        # Put 'y_train' states in vector 'y_train'
        y_train = []
        csv_reader_y = csv.reader(open("y_train.csv", mode='r'))
        for i,row in enumerate(csv_reader_y):
          if i != 0:
            y_train.append(row[0].upper())
        
        # Retrieve train_data
        x_train = []
        csv_reader_x = csv.DictReader(open("x_train.csv", mode='r'))
        for row in csv_reader_x:
          obs_seq = [row['from'],row['to']]
          obs_seq = readFile.look_up_table (self, obs_seq)          # Check if the data is null and convert it 
          obs_seq = readFile.normalize(self, obs_seq)               # Normalize data
          x_train.append(obs_seq)
          
          
        # Retrieve test_data
        x_test = []
        csv_reader_x = csv.DictReader(open("x_test.csv", mode='r'))
        for row in csv_reader_x:
          obs_seq = [row['from'],row['to']]
          obs_seq = readFile.look_up_table (self, obs_seq)             # Check if the data is null and convert it 
          obs_seq = readFile.normalize(self, obs_seq)                  # Normalize data
          x_test.append(obs_seq)
        
        return (x_train,x_test,y_train,y_test)
       
    # ----------------------------------------------------------------------

    def look_up_table(self, words):                # Gestion of a null value
        w = []
        for word in words:
          if word == '':
            w.append('unknown')
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





















