# hmm
Implementation of Hidden Markov Model (HMM) using python3

File 'hmm_scaled.py' 
--------------------
HMM's Functions:
 - _set_log_model(self): allows to set the log model
 - check_prob(self): checks the probabilities distribution 
 - _forward_scaled(self, O): computes the likelihood P (O|Model)
 - _backward_scaled(self, O): time-reversed version of the Forward algorithm
 - _forward_backward_multi_scaled(self, obs_sequences): learn the HMM parameters A, B and pi (HMM training)
 - _viterbi_log(self, O, show='yes'): given an observation sequence O, computes the best hidden state sequence Q (Decoding)
 - predict_next_state(self, t, curr_state): given current state at time t, predict the next hidden state at t+1
 
 
File 'initial_model.json' 
------------------------
It is the HMM initial model (pi, A and B).
  
  
File 'readfile.py'
----------------------
This file selects the features to be used as observations and hidden states. It also splits the dataset in train (x_train: observations, y_train: hidden states) and test data (x_test: observations, y_test: hidden states as targets), 80% and 20% respectively.


File 'main.py'
--------------
The main file that tests HMM functionalities.
