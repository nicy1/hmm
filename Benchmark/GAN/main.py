import readfile
import gan
import random
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import accuracy_score




# Read and normalize dataset according the model
reader = readfile.read('telepathology1.csv')
x_train, y_train, x_test, y_test = reader.get_data()

# Define GAN parameters
epochs = 10
latent_dim = 40
batch_size = 10
print_output_every_n_steps = 20

# Build GAN 
gan_model = gan.GAN(latent_dim)
gan_model.load_data(x_train, y_train)
# Train GAN
generator = gan_model.train(epochs, batch_size, print_output_every_n_steps)

#----------------------#
# TEST GAN´S GENERATOR #
#----------------------#

# Load (real) data
x_real = x_train[:batch_size]
# Generate (fake) data
noise = np.random.normal(0, 1, (batch_size, latent_dim))
x_generated = generator.predict(noise)
# Convert it into 1D list
x_generated = [x[0][0] for x in x_generated]

print('Generated data: ', x_generated)
print('Real data: ', x_real)

# Plot the result
pyplot.title('GAN - real data and generated data', fontsize=20)
pyplot.ylim(-3.0, 3.0)
pyplot.plot(x_real, "b", label = 'real')
pyplot.plot(x_generated, "r", label = 'generated')
pyplot.legend()
pyplot.show()






