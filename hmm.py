import os
import numpy as np


class hmm:

    def __init__(self, compute_trans, compute_emis):
        '''Initialise a new Hidden Markov Model'''

        if (not isinstance(compute_trans, dict)):
            print('Argument "compute_trans" is not a dict')
            raise Exception
        if (not isinstance(compute_emis, dict)):
            print('Argument "compute_emis" is not a dict')
            raise Exception

        self.states = list(compute_trans.keys())
        self.obs = list(compute_emis[self.states[0]])
        self.N = len(self.states)
        self.M = len(self.obs)

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

        # Dict for initial state probabilities
        self.pi = {}
        count  = 0
        for s in compute_trans:
            count += sum(compute_trans[s][s0] for s0 in compute_trans[s])   # Tot. number of data
        for s in self.states:
            val =  sum(compute_trans[s][s0] for s0 in compute_trans[s])
            self.pi[s] = val / count
        
        # Matrix for transition probabilities
        self.A = {}
        for s1 in self.states:
            self.A[s1] = {}
            count = sum(compute_trans[s1][s] for s in compute_trans[s1])
            for s2 in self.states:
                self.A[s1][s2] = compute_trans[s1][s2] / count

        # Matrix for observation probabilities
        self.B = {}
        for s in self.states:
            self.B[s] = {}
            count = sum(compute_emis[s][ob] for ob in compute_emis[s])
            for ob in self.obs:
                self.B[s][ob] = compute_emis[s][ob] / count

    # --------------------------------------------------------------------------- 

    def check_prob(self):
        '''Check probabilities in HMM for validity'''
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
            for o in self.obs:
                sum += self.B[s][o]
            if (abs(sum - 1.0) > delta):
                print('HMM state "%s" has observation ' % (self.states[self.S_index[s]]) + \
                      'probabilities sum not 1.0: ' + str(sum))
                ret -= 1
        return ret

    # ---------------------------------------------------------------------------

    def forward(self, obs_sequence):
        ''' 
        Use the 'forward algorithm <http://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm>'
        to evaluate the given sequence.

        Return the likelihood P(O|λ) - (LIKELIHOOD)
        '''
        # Create a probability matrix forward[obs_len, sta_len]
        alpha = [{}]

        # Initialization (t == 0)
        for s in self.states:
            alpha[0][s] = self.pi[s] * self.B[s][obs_sequence[0]]

        # Recursion
        for t in range(1, len(obs_sequence)):
            alpha.append({})
            for curr in self.states:
                alpha[t][curr] = sum((alpha[t - 1][old] * self.A[old][curr] * \
                                    self.B[curr][obs_sequence[t]]) for old in self.states)

        alpha_prob = sum((alpha[len(obs_sequence) - 1][s]) for s in self.states)               
        return alpha_prob,alpha

    # ---------------------------------------------------------------------------

    def backward(self, obs_sequence):
        '''
        The time-reversed version of the Forward Algorithm
        Return the likelihood P(O|λ)
        '''
        # Create a probability matrix backward[obs_len, sta_len]
        beta = [{} for t in range(len(obs_sequence))]

        # Initialization (t == len(obs)-1)
        for s in self.states:
          beta[len(obs_sequence) - 1][s] = 1

        # Recursion in reverse direction (from len(obs)-2 to 0)
        for t in reversed(range(len(obs_sequence) - 1)):
          for curr in self.states:
            beta[t][curr] = sum((beta[t + 1][old] * self.A[curr][old] * \
                                self.B[old][obs_sequence[t + 1]]) for old in self.states)

        beta_prob = sum((self.pi[s] * self.B[s][obs_sequence[0]] * beta[0][s]) for s in self.states)
        return beta_prob,beta

    # ---------------------------------------------------------------------------
    
    def train(self, obs_sequence):
        '''
        Given 'obs_sequence', learn the HMM parameters A, B and Pi - (LEARNING) 
        Using Forward-backward algorithm
        ''' 
        gamma = [{} for t in range(len(obs_sequence))]
        zi = [{} for t in range(len(obs_sequence) - 1)]

        # Get alpha and beta tables computes
        alpha_prob,alpha = hmm.forward(self, obs_sequence)
        beta_prob,beta = hmm.backward(self, obs_sequence)

        # Compute gamma values
        for t in range(len(obs_sequence)):
          for s in self.states:
            gamma[t][s] = (alpha[t][s] * beta[t][s]) / alpha_prob
            if t == 0:
              # Update emission probability
              self.pi[s] = gamma[t][s]

            if t == len(obs_sequence) - 1:
              continue

            # Compute zi values
            zi[t][s] = {}
            for s1 in self.states:
              zi[t][s][s1] = alpha[t][s] * self.A[s][s1] * self.B[s1][obs_sequence[t+1]] * \
                              beta[t+1][s1] / alpha_prob

        # Update transition probability
        for s in self.states:
          for s1 in self.states:
            zi_sum = sum((zi[t][s][s1]) for t in range(len(obs_sequence) - 1))
            gamma_sum = sum((gamma[t][s]) for t in range(len(obs_sequence) - 1)) 
            self.A[s][s1] = zi_sum / gamma_sum

        # Update emission probability
        for s in self.states:
          for obs in self.obs:
            gamma_sum = 0.0
            for t in range(len(obs_sequence)):
              if obs_sequence[t] == obs:
                gamma_sum += gamma[t][s]
            emit_gamma_sum = sum((gamma[t][s]) for t in range(len(obs_sequence)))
            self.B[s][obs] = gamma_sum / emit_gamma_sum

        return self.A, self.B, self.pi

    # ---------------------------------------------------------------------------

    def decode(self, obs):
        '''
        Find the best hidden state sequence Q for the given observation sequence - (DECODING)
        Using "Viterbi algorithm"
        Returns Q and it's probability
        '''
        # Create a probability matrix virtebi[obs_len, sta_len]
        V = [{}]
        for s in self.states:
          V[0][s] = self.pi[s] * self.B[s][obs[0]]

        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
          V.append({})
          for s in self.states:
            (prob, state) = max((V[t - 1][s0] * self.A[s0][s] * self.B[s][obs[t]], s0) for s0 in self.states)
            V[t][s] = prob
          #for i in hmm.dptable(V):
             #print(i)

        bestpath = []
        for i in V:
          for x, y in i.items():
            if i[x] == max(i.values()):
              bestpath.append(x)

        # The highest probability
        bestpath_prob = max(V[-1].values())

        return (bestpath, bestpath_prob)
    
    # ---------------------------------------------------------------------------

    def predict_hidden_states(self, obs_sequence):
        '''
        Given the observations_sequence 
        For each observaion at "t", where t=0,...,len(obs_sequence)-1
        Predict the next_state at "t", where t=0,...,len(obs_sequence)-1
        '''
        # Given a obs_sequence, find be the probability that the Hidden Markov Model 
        # will be in a particular hidden state "s" at a particular time step "t"
        # Trellis Diagram to solve this problem
        alpha_prob,trellis_diagram = hmm.forward(self, obs_sequence)

        predicted_states_seq = []
        for t in range(len(obs_sequence)):
            (prob, next_state) = max((trellis_diagram[t][s], s) for s in trellis_diagram[t])
            predicted_states_seq.append(next_state)

        return predicted_states_seq

    # ---------------------------------------------------------------------------

    def dptable(V):
        yield " ".join(("%10d" % i) for i in range(len(V)))
        for y in V[0]:
            yield "%.7s: " % y + " ".join("%.7s" % ("%f" % v[y]) for v in V)

    # ---------------------------------------------------------------------------

    def print_hmm(self):
        '''
        Print a HMM
        Only probabilities with values larger than 0.0 are printed
        '''
        state_list = self.states[:]  # Make a copy
        state_list.sort()
        obs_list = self.obs[:]  # Make a copy
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
        for s1 in self.states:
            print('    From state: ' + s1)
            for s2 in self.states:
                if (self.A[s1][s2] > 0.0):
                    print('      to state: ' + s2 + ': ' + str(self.A[s1][s2]))

        print('')
        print('  Observation probabilities:')
        for s in self.states:
            print('    In state: ' + s)
            for o in self.obs:
                if (self.B[s][o] > 0.0):
                    print('      Symbol: ' + o + ': ' + str(self.B[s][o]))
        print('')

    # =============================================================================
