import hmm
import hmm_readFile

# ===============================================================================================


filename = 'advertisement_bandits.csv'  # Name of dataset
read = hmm_readFile.readFile(filename)

# Get HMM states and observations lists, train_data (for training) and test_data (for viterbi)
(hmm_states, hmm_obs, train_data, test_data, sequence_states) = read.get_hmm_parameters()

test_hmm = hmm.hmm(hmm_states, hmm_obs)
test_hmm.train(train_data)  # Train the HMM

test_hmm.check_prob()  # Check its probabilities
test_hmm.print_hmm()  # Print it out
print('')

# -------------------------------------------------------------

# Apply the Viterbi algorithm to each sequence of the test_data variable
for test_rec in test_data:
    [state_sequence, sequence_probability] = test_hmm.viterbi(test_rec)

    print('VITERBI: Find the best hidden state sequence')
    print('  Given obs sequence: ', test_rec)
    print('  State sequence: ', state_sequence)
    print('  Sequence prob.: ', sequence_probability)
    print('')

# -------------------------------------------------------------

print('Prediction of the next state:')

count = 0
success_prediction = 0
current_state = test_hmm.get_current_state() 

for real_state in sequence_states:
    predicted_state = test_hmm.predict_state(current_state)

    if predicted_state == real_state:
       success_prediction += 1
    count += 1
    current_state = predicted_state
    print('Predicted state: %s, Real state: %s' % (predicted_state, real_state))

    
print('')   
viterbi_accuracy = (success_prediction/count) * 100
print('PREDICTION ACCURACY: ', viterbi_accuracy, '%')
print('')

# ===============================================================================================


























