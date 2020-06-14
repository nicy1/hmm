import os
import numpy as np


class hmm:

    def __init__(self, states, obs):
       #Initialise a new Hidden Markov Model

       if (not isinstance(states, list)):
           print('Argument "states" is not a list')
           raise Exception
       if (not isinstance(obs, list)):
           print('Argument "obs" is not a list')
           raise Exception

       self.N = len(states)
       self.M = len(obs)
       self.states = states
       self.obs = obs

       # Assign index at each state and observation
       self.S_index = {}
       self.O_index = {}
       i = 0
       for s in self.states:
           self.S_index[s] = i
           i += 1
       i = 0
       for o in self.obs:
           self.O_index[o] = i
           i += 1

       # Matrix for transition probabilities
       self.A = {}
       tmp = hmm.random_normalized (self.N, self.N)       # Temporary matrix
       for s1 in self.states:
           self.A[s1] = {}
           for s2 in self.states:
               self.A[s1][s2] = tmp[self.S_index[s1]][self.S_index[s2]]
        
       # Dict for initial state probabilities
       self.pi = {}
       tmp = hmm.random_normalized (1, self.N)       # Temporary array
       for s in self.states:
           self.pi[s] = tmp[0][self.S_index[s]]

       # Matrix for observation probabilities
       self.B = {}
       tmp = hmm.random_normalized (self.N, self.M)       # Temporary matrix
       for s in self.states: 
           self.B[s] = {}
           for o in self.obs:
               self.B[s][o] = tmp[self.S_index[s]][self.O_index[o]]   
       
 
    # ---------------------------------------------------------------------------

    # Generate multiple random numbers to equal a value
    # Used to generate initial probabilities, trans_prob and emission_prob

    def random_normalized (d1, d2):
        x = np.random.random((d1, d2))
        return x / x.sum(axis=1, keepdims=True)

    # --------------------------------------------------------------------------- 

    def check_prob(self):
        # Check probabilities in HMM for validity

        ret = 0
        delta = 0.0000000000001        # Account for floating-point rounding errors
        
        sum = 0.0
        for s in self.states:
            sum += self.pi[s]
        if (abs(sum - 1.0) > delta):
            print ('HMM initial state probabilities sum is not 1: %f' % (sum))
            ret -= 1

        for s1 in self.states:
            sum  = 0.0
            for s2 in self.states:
                sum += self.A[s1][s2]
            if (abs(sum - 1.0) > delta):
                print ('HMM state "%s" has transition ' % (self.states[s1])  + \
                     'probabilities sum not 1.0: %f' % (sum))
                ret -= 1

        for s in self.states:
            sum  = 0.0
            for o in self.obs:
                sum += self.B[s][o]
            if (abs(sum - 1.0) > delta):
                print ('HMM state "%s" has observation ' % (self.states[s]) + \
                     'probabilities sum not 1.0: ' + str(sum))
                ret -= 1
        return ret

    # ---------------------------------------------------------------------------

    def forward (self, obs):
        # Compute the likelihood P(O|λ) of sequence of observations - (LIKELIHOOD)
        # Return the likelihood P(O|λ)
      
        # Create a probability matrix forward[obs_len, sta_len]
        self.fwd_matrix = [{}]
    
        # Initialization (t == 0)
        for s in self.states:
            self.fwd_matrix[0][s] = self.pi[s] * self.B[s][obs[0]]
        
        # Recursion
        for t in range(1, len(obs)):
            self.fwd_matrix.append({})
            for curr in self.states:
                self.fwd_matrix[t][curr] = sum ((self.fwd_matrix[t - 1][old] * self.A[old][curr] * \
                                                 self.B[curr][obs[t]]) for old in self.states)
               
        self.fwd_prob = sum ((self.fwd_matrix[len(obs) - 1][s]) for s in self.states)
        return self.fwd_prob

    # ---------------------------------------------------------------------------
    
    def backward (self, obs):
        # The time-reversed version of the Forward Algorithm
        # Return the likelihood P(O|λ)
    
        # Create a probability matrix backward[obs_len, sta_len]
        self.bwd_matrix = [{} for t in range(len(obs))]
    
        # Initialization (t == len(obs)-1)
        for s in self.states:
            self.bwd_matrix[len(obs) - 1][s] = 1 
    
        # Recursion in reverse direction (from len(obs)-2 to 0)
        for t in reversed(range(len(obs) - 1)):
            for curr in self.states:
                self.bwd_matrix[t][curr] = sum ((self.bwd_matrix[t + 1][old] * self.A[curr][old] * \
                                                  self.B[old][obs[t + 1]]) for old in self.states)
          
        self.bwd_prob = sum ((self.pi[s] * self.B[s][obs[0]] * self.bwd_matrix[0][s]) for s in self.states)
    
        return self.bwd_prob

    # ---------------------------------------------------------------------------

    def train (self, obs):
        # Train the HMM using forward-backward algorithm
        # Learn the HMM parameters A, B and Pi - (LEARNING) 
    
        gamma = [{} for t in range(len(obs))]
        zi    = [{} for t in range(len(obs) - 1)]
    
        # Get alpha and beta tables computes
        hmm.forward (self, obs)
        hmm.backward (self, obs)

        # Compute gamma values
        for t in range(len(obs)):
            for s in self.states:
                gamma[t][s] = (self.fwd_matrix[t][s] * self.bwd_matrix[t][s]) / self.fwd_prob
                if t == 0:
                   self.pi[s] = gamma[t][s]
          
                # Compute zi values up to len(obs) - 1
                if t == len(obs) - 1:
                   continue
        
                zi[t][s] = {}
                for s1 in self.states:
                    zi[t][s][s1] = self.fwd_matrix[t][s] * self.A[s][s1] * self.B[s1][obs[t + 1]] * \
                            self.bwd_matrix[t + 1][s1] / self.fwd_prob
    
        # Now that we have gamma and zi, let us re-estimate
        for s in self.states:
            for s1 in self.states:
                val = sum ((zi[t][s][s1]) for t in range(len(obs) - 1))
                val /= sum ((gamma[t][s]) for t in range(len(obs) - 1))
                self.A[s][s1] = val
        
        # Re-estimate gamma
        for s in self.states:
            for o in self.obs: 
                val = 0.0
                for t in range (len(obs)):
                    if obs[t] == o :
                        val += gamma[t][s]  
                val /= sum ((gamma[t][s]) for t in range(len(obs)))
                self.B[s][o] = val
    
        return self.A, self.B, self.pi
    
    # ---------------------------------------------------------------------------

    def viterbi (self, obs):
        # Find the best hidden state sequence Q for the given observation sequence - (DECODING)
        # Returns Q and it's probability.
    
        # Create a probability matrix vertebi[obs_len, sta_len]
        V=[{}]
        for s in self.states:
            V[0][s]=self.pi[s]*self.B[s][obs[0]]

        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
            V.append({})
            for s in self.states:
                (prob, state) = max((V[t-1][s0] * self.A[s0][s] * self.B[s][obs[t]], s0) for s0 in self.states)
                V[t][s] = prob
            for i in hmm.dptable(V):
                print (i)

            bestpath = []
            for i in V:
                for x,y in i.items():
                    if i[x] == max(i.values()):
                       bestpath.append(x)

        # The highest probability
        bestpath_prob = max(V[-1].values())

        return bestpath, bestpath_prob

    # ---------------------------------------------------------------------------

    def predict_state (self, states):
        # Given a list of states
        # Predict the next state at 't+1' based on the current state at 't'
        
        current_st = states[0] 
        max = 0.0000
        for s in self.A[current_st]:
            if max < self.A[current_st][s]: 
               max = self.A[current_st][s]
               state = s  
        return state

    # ---------------------------------------------------------------------------

    def dptable (V):
        yield " ".join(("%10d" % i) for i in range(len(V)))
        for y in V[0]:
            yield "%.7s: " % y+" ".join("%.7s" % ("%f" % v[y]) for v in V)

    # ---------------------------------------------------------------------------

    def print_hmm(self):
        # Print a HMM
        # Only probabilities with values larger than 0.0 are printed.

        state_list = self.states[:]  # Make a copy
        state_list.sort()
        obs_list = self.obs[:]       # Make a copy
        obs_list.sort()

        print ('Hidden Markov Model')
        print ('  States:       %s' % (str(state_list)))
        print ('  Observations: %s' % (str(obs_list)))

        print ('')
        print ('  Inital state probabilities:')
        for s in self.states:
            if (self.pi[s] > 0.0):
                print ('    State: '+s+': '+str(self.pi[s]))

        print ('')
        print ('  Transition probabilities:')
        for s1 in self.states:
            print ('    From state: '+s1)
            for s2 in self.states:
                if (self.A[s1][s2] > 0.0):
                    print ('      to state: '+s2+': '+str(self.A[s1][s2]))

        print ('')
        print ('  Observation probabilities:')
        for s in self.states:
            print ('    In state: '+s)
            for o in self.obs:
                if (self.B[s][o] > 0.0):
                    print ('      Symbol: '+o+': '+str(self.B[s][o]))
        print ('')

     # =============================================================================

         
         





       