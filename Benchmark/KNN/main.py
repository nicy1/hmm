import svm
import readfile
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score

# ===============================================================


print ("-- KNN Classifier --")

filename = 'telepathology2.csv' 
reader = readfile.read(filename)
X_train, y_train, X_test, y_test = reader.get_data()

# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 

# creating a confusion matrix 
y_pred = knn.predict(X_test) 
accuracy = accuracy_score(y_test, y_pred)
precision, recall, fscore, support = score(y_test, y_pred)
print('accuracy = %.2f' % (accuracy*100), '%')
print('Precision:', precision)
print('Recall:', recall)
print('F-Measure:', fscore)
print('Support-score:', support)
