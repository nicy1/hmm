import hmm_scaled
import hmm_readFile



# ===============================================================


filename = 'skype_packets.csv' 
# Get HMM parameters, train_data (for training) and test_data (for decoding)
readFile = hmm_readFile.readFile(filename)
(compute_transition_prob,compute_emission_prob,x_train,x_test,y_train,y_test) = readFile.get_data()

model = hmm_scaled.HmmScaled(compute_transition_prob,compute_emission_prob,'init_model.json')

# Train the HMM
model.train(x_train)

model.check_prob()  # Check its probabilities
model.print_hmm()   # Print it out
print('')

  # -------------------------------------------------------------

# Find the best hidden state sequence Q for the given observation sequence - (DECODING)
# Using the Viterbi algorithm
(prob,states) = model.decode(x_test[0]+x_test[1]+x_test[2], show='yes')
print('Test Decoding:')
print('    Input:', x_test[0]+x_test[1]+x_test[2])
print('    Output:', states)
print('')

  # -------------------------------------------------------------

# Given an observation at time t, predict the next state at t+1
# Using Trellis diagram
count = 0
success_prediction = 0

for t,obs in enumerate(x_test):
  # Find the corresponding states which generate obs_sequence
  (prob, state) = model.decode(obs, show='no')
  if t < len(x_test)-1:
     next_state = model.predict(t, curr_state=state)
    
     if next_state == y_test[t+1]:
        success_prediction += 1
     count += 1
     print('time %d: predicted %s - real %s' % (t+1, next_state, y_test[t+1]))


print('')
prediction_accuracy = (success_prediction/count) * 100
print('PREDICTION ACCURACY: %.2f' % (prediction_accuracy), '%')
print('')



# ===============================================================





