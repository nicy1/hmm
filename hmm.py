import os
import sys
import json
import math
import numpy as np

class HmmScaled:

    # Base class for HMM model - implements Rabiner's algorithm for scaling to avoid underflow
    # Model is (A, B, pi) where A = Transition probs, B = Emission Probs, pi = initial distribution
    # A model can be initialized to random parameters
    def __init__(self, states, symbols, compute_trans, compute_emis, model_name=None):
        
        if (not isinstance(compute_trans, dict)):
            print('Argument "compute_trans" is not a dict')
            raise Exception
        if (not isinstance(compute_emis, dict)):
            print('Argument "compute_emis" is not a dict')
            raise Exception
        
        self.states = list(states)
        self.symbols = list(symbols)
        self.N = len(self.states)
        self.M = len(self.symbols)

        # Assign index at each state and symbol
        self.S_index = {}
        self.O_index = {}
        for i,s in enumerate(self.states):
            self.S_index[s] = i
        for i,sym in enumerate(self.symbols):
            self.O_index[sym] = i
        
        if model_name != None:
           self._set_init_model(model_name)
        
        else:          
           # Dict for initial state probabilities
           self.pi = {}
           count  = 0
           for s in compute_trans:
               count += sum(compute_trans[s].values())   # Tot. number of data
           for s in self.states:
               val = sum(compute_trans[s].values()) 
               self.pi[s] = val / count
        
           # Matrix for transition probabilities
           self.A = {}
           for s1 in self.states:
             self.A[s1] = {}
             count = sum(compute_trans[s1].values())
             for s2 in self.states:
                 if s2 in compute_trans[s1]:
                    self.A[s1][s2] = compute_trans[s1][s2] / count
                 else:
                    self.A[s1][s2] = sys.float_info.min
        
           # Matrix for observation probabilities
           self.B = {}
           for s in self.states:
             self.B[s] = {}
             count = sum(compute_emis[s].values())
             for sym in self.symbols:
                 if sym in compute_emis[s]:
                    self.B[s][sym] = compute_emis[s][sym] / count
                 else:
                    self.B[s][sym] = sys.float_info.min
        
        # Generate data randomly
        '''
        # Matrix for transition probabilities
        self.A = {}
        tmp = self._random_normalized(self.N, self.N)  # Temporary matrix
        for s1 in self.states:
            self.A[s1] = {}
            for s2 in self.states:
                self.A[s1][s2] = tmp[self.S_index[s1]][self.S_index[s2]]

        # Dict for initial state probabilities
        self.pi = {}
        tmp = self._random_normalized(1, self.N)  # Temporary array
        for s in self.states:
            self.pi[s] = tmp[0][self.S_index[s]]

        # Matrix for observation probabilities
        self.B = {}
        tmp = self._random_normalized(self.N, self.M)  # Temporary matrix
        for s in self.states:
            self.B[s] = {}
            for o in self.symbols:
                self.B[s][o] = tmp[self.S_index[s]][self.O_index[o]]
        '''
        # The following are defined to support log version of viterbi
        # We assume that the forward and backward functions use the scaled model
        self.logA = {}
        self.logB = {}
        self.logpi = {}
        self._set_log_model()
        
        #self.print_hmm()
    # --------------------------------------------------------------------------- 
    
    def _set_init_model(self, model_name):

        self.model = json.loads(open(model_name).read())['hmm']
        self.pi = self.model['pi']
        self.A = self.model['A']
        tmp = self.model['B']
        self.B = {}
        for s in self.states:
          self.B[s] = {}
          for sym in self.symbols:
            if sym in tmp[s].keys():
                self.B[s][sym] = tmp[s][sym]
            else:
                self.B[s][sym] = 0.0

    # ---------------------------------------------------------------------------   

    def _random_normalized(self, d1, d2):            # Generate randomly probabilities
        x = np.random.random((d1, d2))
        return x / x.sum(axis=1, keepdims=True)

    # ---------------------------------------------------------------------------

    def _set_log_model(self):   

        for s in self.states:
          self.logA[s] = {}
          for s1 in self.A[s].keys():
            if self.A[s][s1] == 0.0:
              self.logA[s][s1] = sys.float_info.min   # This is to handle symbols that never appear in the dataset
            else:
              self.logA[s][s1] = math.log(self.A[s][s1])
          self.logB[s] = {}

          for sym in self.B[s].keys():
            if self.B[s][sym] == 0.0:
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
    
    # This function implements the forward algorithm from Rabiner's paper
    # This implements scaling as per the paper and the errata
    # Given an observation sequence (a list of symbols) and Model, compute P(O|Model)
    # Likelihood problem
    def _forward_scaled(self, obs_seq):
        
        self.fwd = [{}]
        self.clist = []                  # List of coefficients used for scaling
        local_alpha = {}                 # This is the alpha double caret in Rabiner
        self.fwd_scaled = [{}]           # fwd_scaled is the variable alpha_caret in Rabiner book
        # Initialize base cases (t == 0)
        for s in self.states:
            self.fwd[0][s] = self.pi[s] * self.B[s][obs_seq[0]]
        # Compute scaling coefficient (t == 0)
        c = 1 / sum(self.fwd[0].values())
        if c == 0:
           c = 1.0     # Set c's to 1s to avoid NaNs
        self.clist.append(c)
        # Create scaled alpha values (t == 0)
        for s in self.states:
            self.fwd_scaled[0][s] = c * self.fwd[0][s]
        # Recursion
        for t in range(1, len(obs_seq)):
          self.fwd.append({})
          self.fwd_scaled.append({})
          for s in self.states:
            #self.fwd[t][s] = sum((self.fwd[t-1][s0] * self.A[s0][s] * self.B[s][obs_seq[t]]) for s0 in self.states)
            local_alpha[s] = sum((self.fwd_scaled[t-1][s0] * self.A[s0][s] * self.B[s][obs_seq[t]]) for s0 in self.states)
          c  = 1 / sum(local_alpha.values())
          if c == 0:
             c = 1.0     # Set c's to 1s to avoid NaNs
          self.clist.append(c)
          for s in self.states:
              self.fwd_scaled[t][s] = c * local_alpha[s]

        log_p = -sum([math.log(c) for c in self.clist])
        # NOTE: if log probabilty is very low, prob can turn out to be zero
        #prob = math.exp(log_p)
        return log_p   
    # ---------------------------------------------------------------------------

    # Uses the clist created during forward_scaled function
    # This assumes that forward_scaled is already execued and clist is set up properly
    def _backward_scaled(self, obs_seq):  
        
        T = len(obs_seq)
        self.bwk = [{} for t in range(len(obs_seq))]
        self.bwk_scaled = [{} for t in range(len(obs_seq))]
        
        # Initialize base cases (t == T-1)
        for s in self.states:
            self.bwk[T-1][s] = 1 
            try:
                # Create scaled beta values (t == T-1)
                self.bwk_scaled[T-1][s] = self.clist[T-1] * 1.0 
            except:
                print ('EXCEPTION OCCURED in backward_scaled, T-1 = ', T-1)

        for t in reversed(range(T-1)):
          local_beta = {}
          for s in self.states:
            local_beta[s] = sum((self.bwk_scaled[t+1][s0] * self.A[s][s0] * self.B[s0][obs_seq[t+1]]) for s0 in self.states) 
               
          for s in self.states:
              self.bwk_scaled[t][s] = self.clist[t] * local_beta[s]
        
        log_p = -sum([math.log(c) for c in self.clist])
        # NOTE: if log probabilty is very low, prob can turn out to be zero
        #prob = math.exp(log_p)
        return log_p 

    # ---------------------------------------------------------------------------

    # Function to compute xi probabilities   
    def _xi(self, obs_seq):
        xi_t = []    # This holds the gamma for Tk - 1

        for t in range(len(obs_seq)-1):
          xi_t.append({})
          for s in self.states:
            xi_t[t][s] = {}
            for s1 in self.states:
              xi_t[t][s][s1] = (self.fwd_scaled[t][s] * self.bwk_scaled[t + 1][s1] * \
                                self.A[s][s1] * self.B[s1][obs_seq[t + 1]]) 
        return xi_t

    # ---------------------------------------------------------------------------
    
    # Function to find gamma probabilities
    def _gamma(self, obs_seq):
        gamma_t = []   # This holds the gamma for Tk - 1    
        for t in range(len(obs_seq) - 1):
          gamma_t.append({})
          for s in self.states:
            gamma_t[t][s] = self.fwd_scaled[t][s] * self.bwk_scaled[t][s] / float(self.clist[t])
            if gamma_t[t][s] == 0:
               pass      # To handle any error situation arising due to gamma = 0
        return gamma_t
    
    # ---------------------------------------------------------------------------  

    # Compute aij for a given (i, j) pair of states
    def _compute_aij(self, i, j):
        numerator = 0.0
        denominator = 0.0        
        for t in range(len(self.xi_table)):     
            denominator += self.gamma_table[t][i]     # gamma value for i, j
            numerator += self.xi_table[t][i][j]       # xi value for i, j
        aij = numerator / denominator
        return aij

    # ---------------------------------------------------------------------------
    
    # Compute the emission probabilities of a given state i emitting symbol
    def _compute_bj(self, obs_seq, i, symbol):
        numerator =  0.0 
        denominator = 0.0      
        for t in range(len(self.gamma_table)):     
            denominator += self.gamma_table[t][i]        # gamma value for i, j
            if obs_seq[t] == symbol:
               numerator += self.gamma_table[t][i]       # xi value for i, j
        bj = numerator / denominator
        return bj

    # ---------------------------------------------------------------------------
    
    # Given 'obs_sequences', learn the HMM parameters A, B and pi - (LEARNING) 
    # Returns new model (A, B and pi) given the initial model
    # Using Forward-Backwardh algorithm
    def train(self, obs_seq, iterations = 1, verbose=True):
        
        for i in range(iterations):
          if verbose:
            print("Iteration: {}".format(i + 1))

          self._expectation(obs_seq)
          self._maximization(obs_seq)

    # ---------------------------------------------------------------------------

    def _expectation(self, obs_seq):
        '''
        Executes expectation step.
        '''
        self._forward_scaled(obs_seq)
        self._backward_scaled(obs_seq)

        self.xi_table = self._xi(obs_seq)   
        self.gamma_table = self._gamma(obs_seq)

    # ---------------------------------------------------------------------------

    def _maximization(self, obs_seq):
        '''
        Executes maximization step.
        '''
        temp_pi = {} 
        temp_aij = {} 
        temp_bjk = {}

        # Update self.pi
        for s in self.states:
            temp_pi[s] = self.gamma_table[0][s]
        normalizer = 0.0
        for v in temp_pi.values():
            normalizer += v
        for k, v in temp_pi.items():
            temp_pi[k] = v / normalizer
 
        for s in self.states:
          temp_bjk[s] = {}
          temp_aij[s] = {}
          # Update self.A
          for s1 in self.states:
            temp_aij[s][s1] = self._compute_aij(s, s1)
          # Update self.B
          for sym in self.symbols:
            temp_bjk[s][sym] = self._compute_bj(obs_seq, s, sym)

        self.A = temp_aij
        self.B = temp_bjk
        self.pi = temp_pi
        self._set_log_model()

    # ---------------------------------------------------------------------------
  
    # Find the best hidden state sequence Q for the given observation sequence - (DECODING)
    # Using "Viterbi algorithm"
    # Returns Q and it's probability
    def _viterbi_log(self, obs_seq, verbose):
        V = [{}]
        path = {}
        # Initialize base cases (t == 0)
        for s in self.states:
          V[0][s] = self.pi[s] * self.B[s][obs_seq[0]]
          path[s] = [s]

        # Run Viterbi when t > 0
        for t in range(1, len(obs_seq)):
          V.append({})
          newpath = {}
          for s in self.states:
            (prob, state) = max((V[t - 1][s0] * self.logA[s0][s] * self.logB[s][obs_seq[t]], s0) for s0 in self.states)
            V[t][s] = prob
            newpath[s] = path[state] + [s]               
          # Don't need to remember the old paths
          path = newpath
        
        if verbose:
          self._printDptable(V)

        (prob, state) = max([(V[len(obs_seq)-1][s0], s0) for s0 in self.states])

        if len(obs_seq) == 1:          # for only one element
           return (prob, state)
        else:
           return (prob, path[state])
    
    #   ---------------------------------------------------------------------------

    def predict(self, obs_seq, verbose=True):
      return self._viterbi_log(obs_seq, verbose)

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
                    print('      Symbol: ' + str(sym) + ': ' + str(self.B[s][sym]))
        print('')

    # =============================================================================
