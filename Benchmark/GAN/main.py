import readfile
import gan
import numpy as np
from matplotlib import pyplot


# Read and normalize dataset according the model
reader = readfile.read('telepathology1.csv')
training_file, testing_file = reader.get_data()

latent_dim = 100
gan_model = gan.GAN()
gan_model.load_dataset(training_file)
generator = gan_model.train()

# Create noisy input for generator
batch_size = 100
noise = np.random.normal(0, 1, (batch_size, latent_dim))
# Generate (fake) data
x_generated = generator.predict(noise)
print("Generated data: " , x_generated)

# plot the result
#show_plot(X, 10)
