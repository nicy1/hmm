import svm
import readfile
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ===============================================================


print ("-- SVM Classifier --")


filename = 'telepathology3.csv' 
reader = readfile.read(filename)
X_train, y_train, X_test, y_test = reader.get_data() 
print(y_train)
clf = svm.SupportVectorMachine(kernel=svm.rbf_kernel, gamma = 1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)
print(y_test)
accuracy = accuracy_score(y_test, y_pred)
print ("Accuracy: %.2f" % (accuracy*100), '%')
