import json
import csv
import requests

import numpy as np

import sys
import os


# A:  states transitions probabilities matrix
# B:  states emissions probabilities matrix
# pi: states initial probabilities


class Hmm:
    
  def __init__ (self, model_file):
    #initialization
    
    if model_file == None:
      print ('Error! Insert the file name.')
      os.exit ()

    data = json.loads (open(model_file).read())

    if 'localdataset' in model_file:
      self.hmm  = data["hmm"]
      self.A = self.hmm["A"]
      self.states = self.A.keys()
      self.B = self.hmm["B"]
      self.symbols = list(self.B.values())[0].keys()
      self.pi = self.hmm["pi"]
        
    else:  
      self.states = ['s1', 's2']   

      #set observations or symbols
      self.symbols = []
      for record in data:
        self.symbols.append(record['Patient_Status'])
      self.symbols = list (dict.fromkeys(self.symbols))     #distinct symbols (= Patient_Status)   

      #set pi, A and B
      #pi = np.ones(len(states))/len(states)
      #self.A = Hmm.random_normalized(len(self.states), len(self.states))
      #self.B = Hmm.random_normalized(len(self.states), len(self.symbols))
      self.pi = {'s1':0.5, 's2':0.5} 

      self.A  = {'s1':{'s1':0.3, 's2':0.7},
                 's2':{'s1':0.5, 's2':0.5}}
      
      self.B = {}
      for s in self.states:
        self.B[s] = {}
        for o in self.symbols:
          self.B[s][o] = 1.0 / len(self.symbols)
        


  def random_normalized (d1, d2):
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)
     

  def forward (self, obs):
    print ('The likelihood P(O|λ) - FORWARD ALGORITHM')
      
    #create a probability matrix forward[obs_len, sta_len]
    self.fwd_matrix = [{}]
    
    #initialization (t == 0)
    for s in self.states:
      self.fwd_matrix[0][s] = self.pi[s] * self.B[s][obs[0]]
      
    #recursion
    for t in range(1, len(obs)):
      self.fwd_matrix.append({})
      
      for curr in self.states:
        self.fwd_matrix[t][curr] = sum ((self.fwd_matrix[t - 1][old] * self.A[old][curr] * self.B[curr][obs[t]]) for old in self.states) 
    
    self.fwd_prob = sum ((self.fwd_matrix[len(obs) - 1][s]) for s in self.states)
    
    return (self.fwd_matrix, self.fwd_prob)
    
  
  
  def viterbi (self, obs):
    print ('The best hidden state sequence Q (Decoding) - VITERBI ALGORITHM:')
    
    #create a probability matrix vertebi[obs_len, sta_len] and paths
    self.viterbi_matrix = [{}]
    self.paths = {}
    
    #initialization
    for s in self.states:
      self.viterbi_matrix[0][s] = self.pi[s] * self.B[s][obs[0]]
      self.paths[s] = [s]         #create a list for each state entry
      
    #recursion
    for t in range(1, len(obs)):
      self.viterbi_matrix.append({})
      
      for curr in self.states:
        (prob, state) = max ((self.viterbi_matrix[t - 1][old] * self.A[old][curr] * self.B[curr][obs[t]], old) for old in self.states)  
        
        self.viterbi_matrix[t][curr] = prob
        #update the path
        self.paths[curr] = self.paths[state] + [curr]
    
    #find the best path with the max prob    
    (prob, start_state) = max ((self.viterbi_matrix[len(obs) - 1][s], s) for s in self.states)
    
    self.bestpath_prob = prob
    self.bestpath = self.paths[start_state]
    
    return (self.bestpath, self.bestpath_prob)
    
    
  def backward (self, obs):
    print ('THE BACKWARD ALGORITHM ')
    
    #create a probability matrix backward[obs_len, sta_len]
    self.bwd_matrix = [{} for t in range(len(obs))]
    
    #initialization (t == len(obs)-1)
    for s in self.states:
      self.bwd_matrix[len(obs) - 1][s] = 1 
    
    #recursion in reverse direction (from len(obs)-2 to 0)
    for t in reversed(range(len(obs) - 1)):
      for curr in self.states:
        self.bwd_matrix[t][curr] = sum ((self.bwd_matrix[t + 1][old] * self.A[curr][old] * self.B[old][obs[t + 1]]) for old in self.states)
          
    self.bwd_prob = sum ((self.pi[s] * self.B[s][obs[0]] * self.bwd_matrix[0][s]) for s in self.states)
    
    return (self.bwd_matrix, self.bwd_prob)
        
        
        
  def forward_backward (self, obs):
    print ('Learn the HMM parameters A, B and Pi (Learning) - FORWARD_BACKWARD ALGORITHM: ')
    
    gamma = [{} for t in range(len(obs))]
    zi    = [{} for t in range(len(obs) - 1)]
    
    # get alpha and beta tables computes
    Hmm.forward (self, obs)
    Hmm.backward (self, obs)
    
    # compute gamma values
    for t in range(len(obs)):
      for s in self.states:
        gamma[t][s] = (self.fwd_matrix[t][s] * self.bwd_matrix[t][s]) / self.fwd_prob
        if t == 0:
          self.pi[s] = gamma[t][s]
          
        #compute zi values up to len(obs) - 1
        if t == len(obs) - 1:
          continue
        
        zi[t][s] = {}
        for s1 in self.states:
         zi[t][s][s1] = self.fwd_matrix[t][s] * self.A[s][s1] * self.B[s1][obs[t + 1]] * self.bwd_matrix[t + 1][s1] / self.fwd_prob
          
    # now that we have gamma and zi, let us re-estimate
    for s in self.states:
      for s1 in self.states:
        val = sum ((zi[t][s][s1]) for t in range(len(obs) - 1))
        val /= sum ((gamma[t][s]) for t in range(len(obs) - 1))
        self.A[s][s1] = val
        
    # re-estimate gamma
    for s in self.states:
      for symbol in self.symbols: 
        val = 0.0
        for t in range (len(obs)):
          if obs[t] == symbol :
            val += gamma[t][s]  
        val /= sum ((gamma[t][s]) for t in range(len(obs)))
        self.B[s][symbol] = val
    
    return (self.A, self.B, self.pi)
    
  
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
