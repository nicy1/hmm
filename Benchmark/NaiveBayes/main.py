import readfile
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# ===============================================================


print ("-- Naive Bayes Classifier --")

filename = 'telepathology2.csv' 
reader = readfile.read(filename)
X_train, y_train, X_test, y_test = reader.get_data()

# training a Naive Bayes classifier 
gnb = GaussianNB().fit(X_train, y_train) 
y_pred = gnb.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred)
precision, recall, fscore, support = score(y_test, y_pred)
print('accuracy = %.2f' % (accuracy*100), '%')
print('Precision:', precision)
print('Recall:', recall)
print('F-Measure:', fscore)
print('Support-score:', support)