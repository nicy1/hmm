import hmm
import hmm_readFile

# =============================================================================================


filename = 'death_and_recovered.csv'             # Name of dataset
read = hmm_readFile.readFile(filename)
                                                
# Get HMM states and observations lists, train_data (for trainning) and test_data (for viterbi)
(hmm_states,hmm_obs,train_data,test_data) = read.get_hmm_parameters()

test_hmm = hmm.hmm(hmm_states, hmm_obs)
test_hmm.train(train_data)  # Train the HMM

test_hmm.check_prob()  # Check its probabilities
test_hmm.print_hmm()   # Print it out
print ('')

# -------------------------------------------------------------

# Apply the Viterbi algorithm to each sequence of the test data
prob = []           # It will contain the prob of current and next obs for computing the viterbi_accuracy
i = 0
for test_rec in test_data:
    [state_sequence, sequence_probability] = test_hmm.viterbi(test_rec)
    if (i < 2):
        prob.append(sequence_probability)
        i += 1
    print ('VITERBI: Find the best hidden state sequence')
    print ('  State sequence: ', state_sequence)
    print ('  Sequence prob.: ', sequence_probability)
print ('')

viterbi_accuracy = prob[0] - prob[1]
print ('VITERBI ACCURACY: ', (viterbi_accuracy))
print ('')



# ===============================================================================================






















