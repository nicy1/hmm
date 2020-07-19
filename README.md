# hmm
Implementation of Hidden Markov Model (HMM) using python3

File 'hmm_scaled.py' 
--------------------
HMM's Functions:
 - _set_log_model(self): allows to set the log model
 - check_prob(self): checks the probabilities distribution 
 - _forward_scaled(self, O): computes the likelihood P(O|Model)
 - _backward_scaled(self, O): time-reversed version of the Forward algorithm
 - _forward_backward_multi_scaled(self, obs_sequences): learn the HMM parameters A, B and pi (HMM training)
 - _viterbi_log(self, O, show='yes'): given an observation sequence O, computes the best hidden state sequence Q (Decoding)
 - predict(self, t, curr_state): given current state at time t, predict the next hidden state at t+1
 
 
File 'random_model.json' 
------------------------
It is the initial model (pi, A and B).
  
  
File 'hmm_readFile.py'
----------------------
This file splits the dataset in train (x_train: independent features, y_train: dependent targets) and test data (x_test: independent features, y_test: dependent targets), 70% and 30% respectively.


File 'main.py'
--------------
The main file that tests HMM functionalities.
