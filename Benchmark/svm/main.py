from sklearn.svm import SVC
import readfile


# ===============================================================


filename = 'telepathology.csv' 
reader = readfile.read(filename)
(x_train,y_train,x_test,y_test) = reader.get_data()

svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train, y_train)

y_pred = svclassifier.predict(x_test)
correct = 0
for t in range(len(y_test)):
  if y_test[t]== y_pred[t]:
    correct += 1
  print('predicted=%s, expected=%s' % (y_pred[t], y_test[t]))

acc = (correct/len(y_test)) * 100
print('accuracy = %.2f' % acc, '%')