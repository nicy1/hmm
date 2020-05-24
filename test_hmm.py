
import sys
import hmm



h = hmm.Hmm("dataset.json")

#create a sequence of observations for testing
obs = ('Walk', 'Shop', 'Clean')

# test forward algorithm
print (h.forward(obs))
print ('----------------------------\n')

# test backward algorithm
print (h.backward(obs))
print ('----------------------------\n')

# test viterbi algorithm
print (h.viterbi(obs))
print ('----------------------------\n')

# test forward_backward algorithm
A, B, pi = h.forward_backward(obs)
print
print ('Transition probabilities--> ', A)
print
print ('Emission probabilities--> ', B)
print
print ('Initial probabilities--> ', pi)


