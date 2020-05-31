import sys
import hmm



h = hmm.Hmm()
obs = []
choice = int (input('Choose local(0) or external(1) dataset: '))

if choice == 0:                                #TEST WITH MY LOCAL DATASET
  h.set_parameters('localdataset.json')

  #create a sequence of observations for testing
  obs = ['Walk', 'Shop', 'Clean']

else:                                         #TEST FOR THE TELEMEDICINE OBSERVATIONS DATASET
  h.set_parameters(None)
  dataset = h.get_dataset()

  for record in dataset:
    obs.append(record['Patient_Status'])
 
  

# test forward_backward algorithm
A, B, pi = h.forward_backward(obs)
print ('\nTRANSITION PROBABILITIES--> ', A)
print ('\nEMISSION PROBABILITIES--> ', B)
print ('\nINITIAL PROBABILITIES--> ', pi)
print ('----------------------------\n')
# test forward algorithm
print (h.forward(obs))
print ('----------------------------\n')
# test backward algorithm
print (h.backward(obs))
print ('----------------------------\n')
# test viterbi algorithm
print (h.viterbi(obs))

  
  
