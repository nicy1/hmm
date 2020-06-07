import sys
import os

import json
import csv
import requests
from collections import OrderedDict

import hmm
# ======================================================================

class hmmTest:

    def __init__ (self, filename=None):

        # Get dataset in .csv and convert it in .json
        if filename == None:
           print ('Error, set the file name')
           os.exit ()
        
        csvfile = open(filename + '.csv', 'r')
        reader = csv.DictReader(csvfile)

        jsonfile = open(filename + '.json', 'w+')
        d = []
        for r in reader:
            d.append(r)
        jsonfile.write(json.dumps(d, indent=4))

   # ----------------------------------------------------------------------

    def look_up_table (self, p_s):   
        if p_s=='NotCountedbyAnyState#' or p_s=='':
            return 'TREATMENT'
        elif p_s=='Recovered':
            return 'RECOVERED'
        elif p_s=='Deceased':
            return 'DECEASED'

# ====================================================================================

# TESTI STARTING
if __name__ == '__main__':
    
   filename = 'death_and_recovered'           # Name of dataset
   test = hmmTest(filename)

   # ----------------------------------------------------------------------
   # Define HMM state list and observation list

   """
   RECOVERED
   DECEASED
   TRAITEMENT: if the value of the field 'Patient_Status' is empty or in case of 'NotCountedbyAnyState#'
   """
   hmm_states = []
   hmm_obs = ['RECOVERED', 'DECEASED', 'TREATMENT']


   data = json.loads(open(filename+'.json').read())

   max_seq = 300
   test_data = [[] for i in range(max_seq)]       # Observation sequences (one per line) for testing
   n_obs_seq = 0                                  # Number of obs sequences (MAX: 300)
   l = 3                                          # Length of each obs sequence in 'test_data'
   train_data = []
   num_rec_to_select = 400                    # Number of records to select for trainning
   
   for record in data:
       hmm_states.append(record['Statecode'])       
       O = test.look_up_table(record['Patient_Status'])    
       if num_rec_to_select != 0:             # Inserting Observations in train_data (max 100)
          train_data.append(O)   
          num_rec_to_select -= 1
       else:
          if l != 0: 
             test_data[n_obs_seq].append(O)           # Get records from index 100
             l -= 1
          else:
             l = 3                                    # Set again the length
             n_obs_seq += 1                           # Next sequence index           
          
       if n_obs_seq == max_seq: break

   hmm_states = list(dict.fromkeys(hmm_states))                           # Remove duplicated states
   test_data = OrderedDict((tuple(x), x) for x in test_data).values()     # Remove duplicated lists
    
   # ------------------------------------------------------------------------

   # Initialise a new HMM and train it
   init_prob = {}
   trans_prob = {}
   obs_prob = {}
   for s in hmm_states:
       init_prob[s] = 1/len(hmm_states)
       trans_prob[s] = {}
       obs_prob[s] = {}
       for s1 in hmm_states:
           trans_prob[s][s1] = 1/len(hmm_states)
       for o in hmm_obs:
           obs_prob[s][o] = 1/len(hmm_obs)
   
   test_hmm = hmm.hmm(hmm_states, hmm_obs, init_prob, trans_prob, obs_prob)
   test_hmm.train(train_data)  # Train the HMM

   test_hmm.check_prob()  # Check its probabilities
   test_hmm.print_hmm()   # Print it out

   # -------------------------------------------------------------

   # Apply the Viterbi algorithm to each sequence of the test data
   print(test_data)
   for test_rec in test_data:
       [state_sequence, sequence_probability] = test_hmm.viterbi(test_rec)
       print ('VITERBI: Find the best hidden state sequence')
       print ('  State sequence: ', state_sequence)
       print ('  Sequence prob.: ', sequence_probability)
   print ('')

   # -------------------------------------------------------------
   
   # Get likelihood prob (forward)
   prob = test_hmm.forward(train_data)
   print ('FORWARD: Likelihood probability')
   print ('  probability: ', prob)
   print ('')

   # -------------------------------------------------------------
   
   # Get likelihood prob (backward)
   prob = test_hmm.backward(train_data)
   print ('BACKWARD: Likelihood probability')
   print ('  probability: ', prob)
   print ('')



# ===============================================================================================






















