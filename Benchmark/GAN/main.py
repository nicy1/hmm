import readfile
import gan
import utils
import random
import hmm
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import accuracy_score



util = utils.Utils() 
model = hmm.HmmScaled()

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
generator = gan.train(epochs, batch_size)
 

# Test GAN         
y_pred = []
for i in range(len(testX)):
    temp_X = np.array(testX[i])
    temp_X = np.array(temp_X).reshape(batch_size, latent_dim, feature_num)
    predictions = generator.predict(temp_X)
    y_pred.append(predictions[0][0])

y_pred = util.convert(y_pred)
accuracy = accuracy_score(testY, y_pred)
print('accuracy = %.2f' % (accuracy*100), '%')




# Plot the result
pyplot.title('GAN - real and predicted data', fontsize=20)
pyplot.ylim(-3.0, 3.0)
pyplot.plot(testY, "b", label = 'real')
pyplot.plot(y_pred, "r", label = 'predicted')
pyplot.legend()
pyplot.show()






