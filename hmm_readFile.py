import sys
import os

import json
import csv
import inflect
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
             
    # ----------------------------------------------------------------------

    def get_hmm_parameters (self):  
        """Return states and observations lists"""
 
        data = json.loads(open(self.filename).read())

        self.states = []
        self.observations = []

        data = json.loads(open(self.filename).read())

        max_seq = 30
        self.test_data = [[] for i in range(max_seq)]         # Observation sequences (one per line) for testing
        n_obs_seq = 0                                         # Number of obs sequence list in 'test_data' (MAX: 300)
        l = 3                                                 # Length of each obs sequence in 'test_data'
        self.train_data = []
        num_rec_to_select = 500                       # Number of records to select for trainning
           
        for record in data:
            st = record['Statecode']                          # Temporary variable
            self.states.append(st.upper())       
            obs = record['Patient_Status']  
            obs = readFile.look_up_table (self, obs)        # Check if the record is null and convert it 
            obs = readFile.normalize(obs)                   # Normalize data
            self.observations.append(obs)
            if num_rec_to_select != 0:                      # Putting Obs in train_data (max 100)
               self.train_data.append(obs)   
               num_rec_to_select -= 1
            else:
               if l != 0: 
                  self.test_data[n_obs_seq].append(obs)           # Get records from index 100
                  l -= 1
               else:
                  l = 3                                    # Set again the length
                  n_obs_seq += 1                           # Next sequence index           
          
            if n_obs_seq == max_seq: break

        self.states = list(dict.fromkeys(self.states))                                   # Remove duplicated states
        self.observations = list(dict.fromkeys(self.observations))                       # Remove duplicated states
        self.test_data = OrderedDict((tuple(x), x) for x in self.test_data).values()     # Remove duplicated lists

        return (self.states,self.observations,self.train_data,self.test_data)
    
          
    # ----------------------------------------------------------------------

    def look_up_table(self, word):                            # Gestion of a null value
        if word!='Recovered' and word!='Deceased':
           return 'Treatment'
        else: 
           return word

    # ----------------------------------------------------------------------

    def to_lowercase(words):
        """Convert all characters to lowercase from list of words"""
        new_words = ''
        for word in words:
            new_words +=word.lower()
        return new_words

    # ----------------------------------------------------------------------

    def replace_numbers(words):
        """Replace all interger occurrences in list of words"""
        p = inflect.engine()
        new_words = ''
        for word in words:
            if word.isdigit():
               new_word = p.number_to_words(word)
               new_words += new_word
            else:
               new_words += word
        return new_words

    # ----------------------------------------------------------------------

    def normalize (words):
        words = readFile.to_lowercase(words)
        words = readFile.replace_numbers(words)
        return words


   
# ==========================================================================





















