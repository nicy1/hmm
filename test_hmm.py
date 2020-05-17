
import sys
import hmm

h = hmm.Hmm() 
print (h.forward())

print (h.viterbi())

print (h.forward_backward())


