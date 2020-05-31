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
  
  def __init__ (self):
    
    self.A = {}
    self.states = []
    self.B = {}
    self.symbols = []
    self.pi = {}

    #default dataset
    #'https://raw.githubusercontent.com/nicy1/hmm/master/death_recovered.csv'
    path = 'https://api.covid19india.org/csv/latest/death_and_recovered.csv'
    text = requests.get(path).text
    lines = text.splitlines()
    reader = csv.DictReader(lines)    

    self.f = open('dataset.json', 'w+')
    d = []
    for r in reader:
      d.append(r)
    self.f.write(json.dumps(d, indent=4))
    
    
  def get_dataset (self):
    return json.loads (open('dataset.json').read())
     
     
  def set_parameters (self, model_file):
    #initialization
  
    if model_file != None:
      data = json.loads (open(model_file).read())
      self.hmm  = data["hmm"]
      
      if self.hmm == None:
        print ('Error on the syntax of model file ')
        os.exit()
      else:
        self.A = self.hmm["A"]
        self.states = self.A.keys()
        self.B = self.hmm["B"]
        self.symbols = list(self.B.values())[0].keys()
        self.pi = self.hmm["pi"]
        
    else:
      self.states = []
      dataset = Hmm.get_dataset (self)
      
      #set states 
      for record in dataset:
        self.states.append(record['State'])
      self.states = list (dict.fromkeys(self.states))       #distinct states (= State)
      
      #set observations or symbols
      for record in dataset:
        self.symbols.append(record['Patient_Status'])
      self.symbols = list (dict.fromkeys(self.symbols))     #distinct symbols (= Patient_Status)
      
      #set pi and A
      for s in self.states:
        self.pi[s] = 1.0 / len(self.states)
        self.A[s] = {}
        for s2 in self.states:
          self.A[s][s2] = 1.0 / len(self.states)
     
      #set B
      for s in self.states:
        self.B[s] = {}
        for symbol in self.symbols:
          self.B[s][symbol] = 1.0 / len(self.symbols)


  
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
    
  
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
