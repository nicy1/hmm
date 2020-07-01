import hmm
import hmm_readFile

# ===============================================================================================


filename = 'RCM_diagnostic.csv'  # Name of dataset
read = hmm_readFile.readFile(filename)

# Get HMM parameters, train_data (for training) and test_data (for viterbi)
(compute_trans, compute_emis, train_data, test_data) = read.get_hmm_parameters()

test_hmm = hmm.hmm(compute_trans, compute_emis)
test_hmm.train(train_data)  # Train the HMM

test_hmm.check_prob()  # Check its probabilities
test_hmm.print_hmm()  # Print it out
print('')

# -------------------------------------------------------------

# Find the best hidden state sequence Q for the given observation sequence - (DECODING)
# Apply the Viterbi algorithm
# Prediction of the next state starting by the 'observation_sequence'

count = 0
success_prediction = 0
# Find the hidden states of the 'observation_sequence'

(state_sequence, sequence_probability) = test_hmm.viterbi(test_data)    

for i, real_state in enumerate(state_sequence):
    predicted_state = test_hmm.predict_state(real_state)
    
    if i == len(state_sequence)-1:        # End of sequence
       break
    if predicted_state == state_sequence[i+1]:
        success_prediction += 1
    count += 1
    print('%d => Predicted state: %s, Real state: %s' % (i+1, predicted_state, state_sequence[i+1]))
  
print('')   
viterbi_accuracy = (success_prediction/count) * 100
print('PREDICTION ACCURACY: %.2f' % (viterbi_accuracy), '%')
print('')

# ===============================================================================================


























