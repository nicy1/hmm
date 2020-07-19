import os
import sys
import math
import json
import numpy as np

class HmmScaled:

    # Base class for HMM model - implements Rabiner's algorithm for scaling to avoid underflow
    # Model is (A, B, pi) where A = Transition probs, B = Emission Probs, pi = initial distribution
    # A model can be initialized to random parameters
    def __init__(self, model_name):

        if model_name == None:
          print ('Fatal Error: You should provide the model file name')
          sys.exit()
        
        self.model = json.loads(open(model_name).read())["hmm"]
        self.pi = self.model["pi"]
        self.A = self.model["A"]
        self.B = self.model["B"]

        self.states = list(self.A.keys())
        self.symbols = list(self.B.values())[0].keys() 
        self.symbols = list(self.symbols)
        self.N = len(self.states)     # Number of states of the model
        self.M = len(self.symbols)    # Number of symbols of the model
      
        # Assign index at each state and symbol
        self.S_index = {}
        self.O_index = {}
        for i,s in enumerate(self.states):
            self.S_index[s] = i
        for i,obs in enumerate(self.symbols):
            self.O_index[obs] = i
        
        # The following are defined to support log version of viterbi
        # We assume that the forward and backward functions use the scaled model
        self.logA = {}
        self.logB = {}
        self.logpi = {}
        self._set_log_model()

    # --------------------------------------------------------------------------- 

    def _set_log_model(self):   

        for s in self.states:
          self.logA[s] = {}
          for s1 in self.A[s].keys():
              self.logA[s][s1] = math.log(self.A[s][s1])
          self.logB[s] = {}

          for sym in self.B[s].keys():
            if self.B[s][sym] == 0:
               self.logB[s][sym] =  sys.float_info.min   # This is to handle symbols that never appear in the dataset
            else:
               self.logB[s][sym] = math.log(self.B[s][sym])

          if self.pi[s] == 0:
             self.logpi[s] =  sys.float_info.min   # This is to handle symbols that never appear in the dataset
          else:
             self.logpi[s] = math.log(self.pi[s])

    # ---------------------------------------------------------------------------

    # Check probabilities in HMM for validity
    def check_prob(self):
        ret = 0
        delta = 0.0000000000001  # Account for floating-point rounding errors

        sum = 0.0
        for s in self.states:
            sum += self.pi[s]
        if (abs(sum - 1.0) > delta):
            print('HMM initial state probabilities sum is not 1: %f' % (sum))
            ret -= 1

        for s1 in self.states:
            sum = 0.0
            for s2 in self.states:
                sum += self.A[s1][s2]
            if (abs(sum - 1.0) > delta):
                print('HMM state "%s" has transition ' % (self.states[self.S_index[s1]]) + \
                      'probabilities sum not 1.0: %f' % (sum))
                ret -= 1

        for s in self.states:
            sum = 0.0
            for o in self.symbols:
                sum += self.B[s][o]
            if (abs(sum - 1.0) > delta):
                print('HMM state "%s" has observation ' % (self.states[self.S_index[s]]) + \
                      'probabilities sum not 1.0: ' + str(sum))
                ret -= 1
        return ret

    # ---------------------------------------------------------------------------

    # Compute c values given the pointer to alpha values
    def _compute_cvalue(self, alpha, states):
        alpha_sum = 0.0
        for s in states:
          alpha_sum += alpha[s]
        if alpha_sum == 0:
          print ('Critical Error, sum of alpha values is zero')
        cval = 1.0 / alpha_sum
        if cval == 0:
          print ('ERROR cval is zero, alpha = ", alpha_sum')

        return cval

    # ---------------------------------------------------------------------------
    
    # This function implements the forward algorithm from Rabiner's paper
    # This implements scaling as per the paper and the errata
    # Given an observation sequence (a list of symbols) and Model, compute P(O|Model)
    # Likelihood problem
    def _forward_scaled(self, obs):   
        self.fwd = [{}]
        local_alpha = {}       # this is the alpha double caret in Rabiner
        self.clist = []        # list of constants used for scaling
        self.fwd_scaled = [{}] # fwd_scaled is the variable alpha_caret in Rabiner book

        # Initialize base case (t == 0)
        for s in self.states:
            self.fwd[0][s] = self.pi[s] * self.B[s][obs[0]]
       
        # Get c1 for base case
        c1 = self._compute_cvalue(self.fwd[0], self.states)
        self.clist.append(c1)

        # Initialize scaled alpha (t == 0)
        for s in self.states:
            self.fwd_scaled[0][s] = c1 * self.fwd[0][s]
       
        # Run Forward algorithm for t > 0
        for t in range(1, len(obs)):   
          self.fwd_scaled.append({})     
          for s in self.states:
            local_alpha[s] = sum((self.fwd_scaled[t-1][s0] * self.A[s0][s] * self.B[s][obs[t]]) for s0 in self.states)
            if (local_alpha[s] == 0):
              print ('ERROR local alpha is zero: s = ', s, '  s0 = ', s0)
              print ('fwd = %3f, A = %3f, B = %3f, obs = %s' % (self.fwd_scaled[t - 1][s0], self.A[s0][s], self.B[s][obs[t]], obs[t]))
          
          c1 = self._compute_cvalue(local_alpha, self.states)
          self.clist.append(c1)
          # Create scaled alpha values
          for s in self.states:
            self.fwd_scaled[t][s] = c1 * local_alpha[s]

        log_p = -sum([math.log(c) for c in self.clist])
        # NOTE: if log probabilty is very low, prob can turn out to be zero
        #prob = math.exp(log_p) #sum((self.fwd[len(obs) - 1][s]) for s in self.states)        
        return log_p #prob
        
    # ---------------------------------------------------------------------------
    
    # Uses the clist created during forward_scaled function
    # This assumes that forward_scaled is already execued and clist is set up properly
    def _backward_scaled(self, obs):  
        self.bwk_scaled = [{} for t in range(len(obs))]
        
        # Initialize base cases (t == T)
        T = len(obs)
        for s in self.states:
          try:
              self.bwk_scaled[T-1][s] = self.clist[T-1] * 1.0 
          except:
              print ('EXCEPTION OCCURED in backward_scaled, T - 1 = ', T-1)
            
        for t in reversed(range(T-1)):
          beta_local = {}
          for s in self.states:
            beta_local[s] = sum((self.bwk_scaled[t+1][s0] * self.A[s][s0] * self.B[s0][obs[t+1]]) for s0 in self.states)
                
          for s in self.states:
            self.bwk_scaled[t][s] = self.clist[t] * beta_local[s]
        
        log_p = -sum([math.log(c) for c in self.clist])
        # NOTE: if log probabilty is very low, prob can turn out to be zero
        #prob = math.exp(log_p) 
        return log_p #prob

    # ---------------------------------------------------------------------------
    
    # Function to compute xi probabilities   
    def _xi(self, obs_sequence):
        xi_t = []    # This holds the gamma for Tk - 1

        for t in range(len(obs_sequence)-1):
          xi_t.append({})
          for s in self.states:
            xi_t[t][s] = {}
            for s1 in self.states:
              xi_t[t][s][s1] = (self.fwd_scaled[t][s] * self.bwk_scaled[t + 1][s1] * \
                                self.A[s][s1] * self.B[s1][obs_sequence[t + 1]]) 
        return xi_t

    # ---------------------------------------------------------------------------
    
    # Function to find gamma probabilities
    def _gamma(self, obs_sequence):
        gamma_t = []   # This holds the gamma for Tk - 1    

        for t in range(len(obs_sequence) - 1):
          gamma_t.append({})
          for s in self.states:
            gamma_t[t][s] = self.fwd_scaled[t][s] * self.bwk_scaled[t][s] / float(self.clist[t])
            if gamma_t[t][s] == 0:
               pass      # To handle any error situation arising due to gamma = 0
        return gamma_t
    
    # ---------------------------------------------------------------------------  

    # Compute aij for a given (i, j) pair of states
    def _compute_aij(self, xi_table, gamma_table, i, j):
        numerator = 0.0
        denominator = 0.0
        
        for k in range(len(xi_table)):     # Sum over all observations in the multi sequence
          for t in range(len(xi_table[k])):     
            denominator += gamma_table[k][t][i]     # gamma value for i, j
            numerator += xi_table[k][t][i][j]       # xi value for i, j
        aij = numerator / denominator
        return aij

    # ---------------------------------------------------------------------------
    
    # Compute the emission probabilities of a given state i emitting symbol
    def _compute_bj(self, xi_table, gamma_table, obs_sequences, i, symbol):
        numerator =  0.0 
        denominator = 0.0
        
        for k in range(len(gamma_table)):     # Sum over all observations in the multi list
          for t in range(len(gamma_table[k])):     
            denominator += gamma_table[k][t][i]        # gamma value for i, j
            if obs_sequences[k][t] == symbol:
               numerator += gamma_table[k][t][i]       # xi value for i, j
        bj = numerator / denominator
        return bj

    # ---------------------------------------------------------------------------

    def _compute_xi_gamma_tables(self, obs_sequences):
        xi_table = []  # Each element in this is for a given obs in obs_sequences, obs is a vector of symbols
        gamma_table = []  # Each element in this is for a given obs in obs_sequences, obs is a vector of symbols
        
        for obs in obs_sequences:   # Do for every observation sequence from the multi observation sequence
          self._forward_scaled(obs)
          self._backward_scaled(obs)

          xi_t = self._xi(obs)   
          gamma_t = self._gamma(obs)
          xi_table.append(xi_t)
          gamma_table.append(gamma_t)

        return (xi_table, gamma_table)

    # ---------------------------------------------------------------------------
    
    # Given 'obs_sequences', learn the HMM parameters A, B and pi - (LEARNING) 
    # Returns new model (A, B and pi) given the initial model
    # Using Forward-Backwardh algorithm
    def _forward_backward_multi_scaled(self, obs_sequences):
        T = len(obs_sequences)
        count = 40

        for iteration in range(count):
          temp_pi = {}
          temp_aij = {}
          temp_bjk = {}
          # xi_table: Each element in this is for a given obs in obs_sequences, obs is a vector of symbols
          # gamma_table: Each element in this is for a given obs in obs_sequences, obs is a vector of symbols
          (xi_table, gamma_table) =  self._compute_xi_gamma_tables(obs_sequences)
          
          # Update self.pi
          for s in self.states:
            temp_pi[s] = 0.0
            for k in range(len(gamma_table)):       # Sum over all observations in the multi sequence
              temp_pi[s] += gamma_table[k][0][s]

          # Update self.A 
          for s in self.states:
            temp_aij[s] = {}
            for s1 in self.states:
              temp_aij[s][s1] = self._compute_aij(xi_table, gamma_table, s, s1)

          # Update self.B
          for s in self.states:
            temp_bjk[s] = {}
            for sym in self.symbols:
              temp_bjk[s][sym] = self._compute_bj(xi_table, gamma_table, obs_sequences, s, sym)

          normalizer = 0.0
          for v in temp_pi.values():
              normalizer += v
          for k, v in temp_pi.items():
              temp_pi[k] = v / normalizer

          self.A = temp_aij
          self.B = temp_bjk
          self.pi = temp_pi

        return (self.A, self.B, self.pi)
     
    # ---------------------------------------------------------------------------

    def train(self, obs_sequences):
        return self._forward_backward_multi_scaled(obs_sequences)

    # ---------------------------------------------------------------------------
    
    # Find the best hidden state sequence Q for the given observation sequence - (DECODING)
    # Using "Viterbi algorithm"
    # Returns Q and it's probability
    def _viterbi_log(self, obs, show='yes'):
        V = [{}]
        path = {}
        # Initialize base cases (t == 0)
        for s in self.states:
          V[0][s] = self.pi[s] * self.B[s][obs[0]]
          path[s] = [s]

        t = 0
        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
          V.append({})
          newpath = {}
          for s in self.states:
            (prob, state) = max((V[t - 1][s0] * self.A[s0][s] * self.B[s][obs[t]], s0) for s0 in self.states)
            V[t][s] = prob
            newpath[s] = path[state] + [s]               
          # Don't need to remember the old paths
          path = newpath
        
        if show == 'yes':
          self._printDptable(V)
        n = 0
        if len(obs)!=1:
          n = t
        (prob, state) = max((V[n][s], s) for s in self.states)

        return (prob, path[state])
        
    #   ---------------------------------------------------------------------------

    def decode(self, obs, show='yes'):
      return self._viterbi_log(obs, show)

    # ---------------------------------------------------------------------------
    
    # Helps visualize the steps of Viterbi (Decode)
    def _printDptable(self, V):
      s = '    ' + ' '.join(('%7d' % i) for i in range(len(V))) + '\n'
      for y in V[0]:
        s += '%.3s: ' % y
        s += ' '.join('%.7s' % ('%f' % v[y]) for v in V)
        s += '\n'
      print(s)

    # ---------------------------------------------------------------------------

    def predict(self, t, curr_state):

        if t == 0:
          self.trellis = [{}]      # Create trellis diagram
          self.trellis[t][curr_state] = self.pi[curr_state]
        else:
          self.trellis.append({})
          posterior_prob = list(self.trellis[t-1].values())[0]
          posterior_state = list(self.trellis[t-1].keys())[0]
          self.trellis[t][curr_state] = posterior_prob * self.A[posterior_state][curr_state]  
      
        # Prediction
        (prob, next_state) = max((self.trellis[t][curr_state] * self.A[curr_state][s0], s0) for s0 in self.states)

        return next_state

    # ---------------------------------------------------------------------------

    # Print a HMM
    # Only probabilities with values larger than 0.0 are printed
    def print_hmm(self):
        state_list = self.states[:]  # Make a copy
        state_list.sort()
        obs_list = self.symbols[:]  # Make a copy
        obs_list.sort()

        print('Hidden Markov Model')
        print('  States:       %s' % (str(state_list)))
        print('  Observations: %s' % (str(obs_list)))

        print('')
        print('  Inital state probabilities:')
        for s in self.states:
            if (self.pi[s] > 0.0):
                print('    State: ' + s + ': ' + str(self.pi[s]))

        print('')
        print('  Transition probabilities:')
        for s in self.states:
            print('    From state: ' + s)
            for s1 in self.states:
                if (self.A[s][s1] > 0.0):
                    print('      to state: ' + s1 + ': ' + str(self.A[s][s1]))

        print('')
        print('  Observation probabilities:')
        for s in self.states:
            print('    In state: ' + s)
            for sym in self.symbols:
                if (self.B[s][sym] > 0.0):
                    print('      Symbol: ' + sym + ': ' + str(self.B[s][sym]))
        print('')

    # =============================================================================
