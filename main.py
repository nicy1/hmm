import hmm
import readfile
from sklearn.metrics import accuracy_score

# ===============================================================


filename = 'telepathology.csv' 
# Get HMM parameters, train_data (for training) and test_data (for decoding)
readFile = readfile.read(filename)
(states, symbols, trans_prob, emis_prob, train_data, test_data, targets) = readFile.get_data()

model = hmm.HmmScaled(states, symbols, trans_prob, emis_prob,'init_model.json')
# Train the HMM
model.train(train_data)

model.check_prob()  # Check its probabilities
model.print_hmm()   # Print it out
print('')
 

  # -------------------------------------------------------------

# Find the best hidden state sequence Q for the given observation sequence - (DECODING)
# Given an observation at time t, predict the next state at t+1
# Method: Viterbi algorithm 

p, y_pred = model.predict(test_data, verbose=False)
accuracy = accuracy_score(targets, y_pred)
print ("Accuracy: %.2f" % (accuracy*100), '%')


# ===============================================================





