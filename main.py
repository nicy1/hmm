import hmm
import hmm_readFile

# ===============================================================================================


filename = 'caesarian.csv' 
# Get HMM parameters, train_data (for training) and test_data (for decoding)
readFile = hmm_readFile.readFile(filename)
(compute_trans, compute_emis, sequences, test_data) = readFile.get_parameters()

model = hmm.hmm(compute_trans, compute_emis)

# Train the HMM
train_data = []
i=130
for seq in sequences:
   if i ==0: break
   train_data += seq
   i-=1
model.train(train_data)

model.check_prob()  # Check its probabilities
model.print_hmm()   # Print it out
print('')

# -------------------------------------------------------------

# Find the best hidden state sequence Q for the given observation sequence - (DECODING)
# Apply the Viterbi algorithm
# Prediction of the next hidden state starting by the "sequences of observations"

count = 0
success_prediction = 0

for obs_seq in test_data:
  # Get the real best hidden states of the "sequence of obs"
  (states_seq, prob) = model.decode(obs_seq) 

  predicted_states_seq = model.predict_hidden_states(obs_seq)
    
  for i in range(len(obs_seq)):
    if predicted_states_seq[i] == states_seq[i]:
      success_prediction += 1
    count += 1
    print('Predicted state: %s, Real state: %s' % (predicted_states_seq[i], states_seq[i]))
  print('')  

print('')
prediction_accuracy = (success_prediction/count) * 100
print('PREDICTION ACCURACY: %.2f' % (prediction_accuracy), '%')
print('')


# ===============================================================================================