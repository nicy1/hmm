import sys
import os

import json
import csv
from collections import OrderedDict


# ==========================================================================


class readFile:
    def __init__ (self, filename=None):
        """This class accepts only csv and json files"""

        if filename == None:
            print ('Error, set the file name')
            os.exit ()

        if '.csv' in filename:                                 # Convert dataset from .csv to .json
            csv_file = open(filename, 'r')
            reader = csv.DictReader(csv_file)

            self.filename = os.path.splitext(filename)[0]       # Get the name of file without extension
            self.filename += '.json'                            # Add extension
            jsonfile = open(self.filename, 'w+')
            d = []
            for r in reader: 
               d.append(r)
            jsonfile.write(json.dumps(d, indent=4))
           
        elif '.json' in self.filename:
            self.filename = filename
        else:
            print ('The program accepts only .csv or .json file')
            raise Exception    

        self.number_to_word = {'0':'zero', '1':'one', '2':'two', '3':'three', '4':'four', \
                               '5':'five', '6':'six', '7':'seven', '8':'eight', '9':'nine'}             
             
    # ----------------------------------------------------------------------

    def get_hmm_parameters (self):  
        """Return states and observations lists"""
 
        data = json.loads(open(self.filename).read())
 
        self.compute_transition_prob = {}
        self.compute_emission_prob = {}      

        data = json.loads(open(self.filename).read())
        
        num_testing = 20           # Number of observations for each sequence in test_data
        num_training = 860          # Number of data for training
        self.train_data = []        # Contains data for training
        self.test_data = []         # Contains data for testing
           
        for i,record in enumerate(data):
            if i == len(data) - 1:       # End of dataset
               break
            
            st = record['Proposed_Management_with_RCM'].upper()           # Temporary variable
            next_st = data[i+1]['Proposed_Management_with_RCM'].upper()     # Next state
            if st not in self.compute_transition_prob.keys():
               self.compute_transition_prob[st] = {}
            if next_st not in self.compute_transition_prob[st]:
               self.compute_transition_prob[st][next_st] = 0
            self.compute_transition_prob[st][next_st] += 1 

            obs = record['Confidence_for_RCM_Management']  
            obs = readFile.look_up_table (self, obs)        # Check if the data is null and convert it 
            obs = readFile.normalize(self, obs)             # Normalize data
        
            if st not in self.compute_emission_prob.keys():
               self.compute_emission_prob[st] = {}
            if obs not in self.compute_emission_prob[st]:
               self.compute_emission_prob[st][obs] = 0
            self.compute_emission_prob[st][obs] += 1
        
            if num_training != 0:              # Insert Obs for training (max 500)
               self.train_data.append(obs)   
               num_training -= 1
            else:
                self.test_data.append(obs)
                num_testing -= 1

            if num_testing == 0:       
               break                    

        return (self.compute_transition_prob,self.compute_emission_prob,self.train_data,self.test_data)
    
          
    # ----------------------------------------------------------------------

    def look_up_table(self, word):                            # Gestion of a null value
        if word == '':
           return 'no_value'
        else: 
           return word

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
        new_words = ''
        for word in words:
            if word.isdigit():
               new_word = self.number_to_word[word]
               new_words += new_word
            else:
               new_words += word
        return new_words

    # ----------------------------------------------------------------------

    def normalize (self, words):
        words = readFile.to_lowercase(self, words)
        words = readFile.replace_numbers(self, words)
        return words


   
# ==========================================================================





















