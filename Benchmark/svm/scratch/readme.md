Implemention of SVM from scratch with python3
----------------------------------------------

File 'readfile.py'
-------------------
This file selects the features to be used as observations and hidden states. It also splits the dataset in train (x_train: observations, y_train: hidden states) and test data (x_test: observations, y_test: hidden states as targets), 80% and 20% respectively.

File 'svm.py'
--------------
For the implementation of the model. It contains:
- Function "fit()" that allows to train the model
- Function "predict()" for the prediction

File 'main.py'
---------------
It uses all the files cited above, trains the model (SVM) with "fit()" function, and tests the model using "predict" function.

The prediction accuracy of this model is 62.48% (with dataset "telepathology1.csv").
