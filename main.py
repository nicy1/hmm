import hmm_scaled
import hmm_readFile



  # ===============================================================


filename = 'android_communication.csv' 
# Get HMM parameters, train_data (for training) and test_data (for decoding)
readFile = hmm_readFile.readFile(filename)
(x_train,x_test,y_train,y_test) = readFile.get_parameters()

model = hmm.HmmScaled('random_model.json')

# Train the HMM
model.train(x_train)

model.check_prob()  # Check its probabilities
model.print_hmm()   # Print it out
print('')

  # -------------------------------------------------------------

# Find the best hidden state sequence Q for the given observation sequence - (DECODING)
# Using the Viterbi algorithm

obs_seq = []         
for i in range(2):           
  obs_seq += x_test[i]

(prob, states) = model.decode(obs_seq, show='yes')
print('Test Decoding:')
print('    Input:', obs_seq)
print('    Output:', states)
print('')

  # -------------------------------------------------------------

# Given an observation at time t, predict the next state at t+1
# Using Trellis diagram
count = 0
success_prediction = 0

for t, sequence in enumerate(x_test):
  # Find the corresponding states which generate obs_sequence
  (prob, states) = model.decode(sequence, show='no')
  
  if t < len(x_test)-1:
    next_state = model.predict(t, curr_state=states[-1])
    
    if next_state == y_test[t+1]:
      success_prediction += 1
    count += 1
    print('time %d: predicted %s - real %s' % (t+1, next_state, y_test[t+1]))


print('')
prediction_accuracy = (success_prediction/count) * 100
print('PREDICTION ACCURACY: %.2f' % (prediction_accuracy), '%')
print('')



  # ===============================================================






