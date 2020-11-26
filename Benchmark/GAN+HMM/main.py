import readfile
import gan
import utils
import random
import hmm
import time
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import classification_report



util = utils.Utils() 

# Read and normalize dataset according the model
reader = readfile.read('telepathology1.csv')
trainX, trainY, testX, testY = reader.get_data()

# Define GAN parameters
epochs = len(trainX)
latent_dim = 1
feature_num = 1
batch_size = 1


# Build GAN 
gan = gan.GAN(latent_dim, feature_num)
gan.load_data(trainX, trainY)
# Train GAN
start = time.time()
generator = gan.train(epochs, batch_size)
stop = time.time()
print("Training time: %.2f s" % (stop - start))

# Test modello GAN + HMM        
y_pred = []
for i in range(len(testX)):
    temp_X = np.array(testX[i])
    temp_X = np.array(temp_X).reshape(batch_size, latent_dim, feature_num)
    predictions = generator.predict(temp_X)
    y_pred.append(predictions[0][0])

y_pred = util.convert(y_pred)

model = hmm.HmmScaled('init_model1.json')

count = 0
success_prediction = 0
for t, y in enumerate(y_pred):
  if t < len(y_pred)-1:
    next_state = model.predict_next(t, str(y))
    
    if float(next_state) == float(testY[t+1]):
      success_prediction += 1
    count += 1
    #print('time %d: predicted %s - real %s' % (t+1, next_state, testY[t+1]))

accuracy = (success_prediction/count) *100
    
print('PREDICTION ACCURACY: %.2f' % (accuracy), '%')






















