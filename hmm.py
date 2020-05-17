
import sys


class Hmm:
  
  def __init__ (self):
    #initialization

    self.states = ['Rainy', 'Sunny']
    self.stat_len = len (self.states)
    self.obs = ['Walk', 'Shop', 'Clean']
    self.obs_len = len (self.obs)
    self.trans = {
      'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
      'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6}
    }
    self.emission =  {
      'Rainy' : {'Walk': 0.1, 'Shop': 0.4, 'Clean': 0.5},
      'Sunny' : {'Walk': 0.6, 'Shop': 0.3, 'Clean': 0.1}
    }
    self.start = {'Rainy': 0.6, 'Sunny': 0.4}
    
  
  
  def forward (self):
    print ('The likelihood P(O|λ) ')
      
    #create a probability matrix forward[obs_len, sta_len]
    self.fwd_matrix = [{}]
    
    #initialization (t == 0)
    for s in self.states:
      self.fwd_matrix[0][s] = self.start[s] * self.emission[s][self.obs[0]]
      
    #recursion
    for t in range(1, self.obs_len):
      self.fwd_matrix.append({})
      
      for curr in self.states:
        self.fwd_matrix[t][curr] = sum ((self.fwd_matrix[t - 1][old] * self.trans[old][curr] * self.emission[curr][self.obs[t]]) for old in self.states)  
        
    self.fwd_prob = sum ((self.fwd_matrix[self.obs_len - 1][s]) for s in self.states)
    
    return (self.fwd_matrix, self.fwd_prob)
    
  
  
  def viterbi (self):
    print ('The best hidden state sequence Q (Decoding) ')
    
    #create a probability matrix vertebi[obs_len, sta_len] and paths
    self.viterbi_matrix = [{}]
    self.paths = {}
    
    #initialization
    for s in self.states:
      self.viterbi_matrix[0][s] = self.start[s] * self.emission[s][self.obs[0]]
      self.paths[s] = [s]         #create a list for each state entry
      
    #recursion
    for t in range(1, self.obs_len):
      self.viterbi_matrix.append({})
      
      for curr in self.states:
        (prob, state) = max ((self.viterbi_matrix[t - 1][old] * self.trans[old][curr] * self.emission[curr][self.obs[t]], old) for old in self.states)  
        
        self.viterbi_matrix[t][curr] = prob
        #update the path
        self.paths[curr] = self.paths[state] + [curr]
    
    #find the best path with the max prob    
    (prob, start_state) = max ((self.viterbi_matrix[self.obs_len - 1][s], s) for s in self.states)
    
    self.bestpath_prob = prob
    self.bestpath = self.paths[start_state]
    
    return (self.bestpath, self.bestpath_prob)
    
    
  def backward (self):
    print ('The backward algorithm ')
    
    #create a probability matrix backward[obs_len, sta_len]
    self.bwd_matrix = [{} for t in range(self.obs_len)]
    
    #initialization (t == obs_len-1)
    for s in self.states:
      self.bwd_matrix[self.obs_len - 1][s] = 1 
    
    #recursion in reverse direction (from obs_len-2 to 0)
    for t in reversed(range(self.obs_len - 1)):
      self.bwd_matrix.append({})
      
      for curr in self.states:
        self.bwd_matrix[t][curr] = sum ((self.bwd_matrix[t + 1][old] * self.trans[curr][old] * self.emission[old][self.obs[t + 1]]) for old in self.states)
          
    self.bwd_prob = sum ((self.start[s] * self.emission[s][self.obs[0]] * self.bwd_matrix[0][s]) for s in self.states)
    
    return (self.bwd_matrix, self.bwd_prob)
        
        
        
  def forward_backward (self):
    print ('Finding probability of states at each time step (Learning): ')
    
    Hmm.forward (self)
    Hmm.backward (self)
    
    self.fwd_bwd = []
    
    for t in range(self.obs_len):
      self.fwd_bwd.append({s: self.fwd_matrix[t][s] * self.bwd_matrix[t][s] / self.fwd_prob for s in self.states})
    
    return self.fwd_bwd  
    
  
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      