import hmm
import readfile
from sklearn.metrics import accuracy_score

# ===============================================================
import hmm
import readfile
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score

# ===============================================================


filename = 'telepathology1.csv' 
# Get HMM parameters, train_data (for training) and test_data (for decoding)
readFile = readfile.read(filename)
(states, symbols, trans_prob, emis_prob, x_train, x_test, y_test) = readFile.get_data()

model = hmm.HmmScaled(states, symbols, trans_prob, emis_prob, 'init_model.json')
# Train the HMM
model.train(x_train)

model.check_prob()  # Check its probabilities
model.print_hmm()   # Print it out
print('')
 

  # -------------------------------------------------------------

# Find the best hidden state sequence Q for the given observation sequence - (DECODING)
# Given an observation at time t, predict the next state at t+1
# Method: Viterbi algorithm 

p, y_pred = model.predict(x_test, verbose=False)
accuracy = accuracy_score(y_test, y_pred)
precision, recall, fscore, support = score(y_test, y_pred)
print('accuracy = %.2f' % (accuracy*100), '%')
print('Precision:', precision)
print('Recall:', recall)
print('F-Measure:', fscore)
print('Support-score:', support)



