from sklearn.svm import SVC
import readfile
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score


# ===============================================================


filename = 'telepathology1.csv' 
reader = readfile.read(filename)
(x_train,y_train,x_test,y_test) = reader.get_data()

svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train, y_train)

y_pred = svclassifier.predict(x_test)
accuracy = accuracy_score(x_test, y_pred)
precision, recall, fscore, support = score(y_test, y_pred)
print('accuracy = %.2f' % (accuracy*100), '%')
print('Precision:', precision)
print('Recall:', recall)
print('F-Measure:', fscore)
print('Support-score:', support)