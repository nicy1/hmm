from keras import layers
from matplotlib import pyplot as plt
from models import Discriminator, Generator
import keras
import pandas as pd
import numpy as np



class GAN():
    def __init__(self, latent_dim=1, feature_num=1):
        self.latent_dim = latent_dim
        self.feature_num = feature_num 
        self.generator = Generator(self.latent_dim, self.feature_num).build_generator()
        self.discriminator = Discriminator(self.latent_dim).build_discriminator()
        discriminator_optimizer = keras.optimizers.RMSprop(lr=8e-4, clipvalue=2.0, decay=1e-8)
        self.discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

        self._build_gan()
        

 
    def _build_gan(self):
        self.discriminator.trainable = False
        gan_input = keras.Input(shape=(self.latent_dim, self.feature_num))
        gan_output = self.discriminator(self.generator(gan_input))
        self.gan = keras.models.Model(gan_input, gan_output)
        gan_optimizer = keras.optimizers.RMSprop(lr=4e-4, clipvalue=2.0, decay=1e-8)
        self.gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
    

    
    def load_data(self, trainx, trainy):
        self.trainX = trainx
        self.trainY = trainy
        print("X_train")
        print(np.array(self.trainX).shape)
        print("Y_train")
        print(np.array(self.trainY).shape)


    
    def _generate_real_samples(self, index, batch_size):
        temp_X = self.trainX[index]
        temp_Y = self.trainY[index]
        # make data in 2D
        temp_X = np.array(temp_X).reshape(self.latent_dim, batch_size)
        temp_Y = np.array(temp_Y).reshape(self.latent_dim, batch_size)
        return temp_X, temp_Y



    def _generate_fake_samples(self, index, batch_size):
        temp_X = self.trainX[index]
        # make data in 3D
        temp_X = np.array(temp_X).reshape(self.latent_dim, batch_size, self.feature_num)
        predictions = self.generator.predict(temp_X)
        return predictions



    # evaluate the discriminator
    def _summarize_performance(self, batch_size):
        # prepare real samples
        X_real, y_real = self._generate_real_samples(batch_size)
        # evaluate discriminator on real examples
        _, acc_real = self.discriminator.evaluate(X_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake = self._generate_fake_samples(batch_size)
        # evaluate discriminator on fake examples
        _, acc_fake = self.discriminator.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance
        print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))



    def train(self, epochs, batch_size=1, print_output_every_n_steps=100):
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Generate fake and real inputs 
            predictions = self._generate_fake_samples(epoch, batch_size)
            temp_X, temp_Y = self._generate_real_samples(epoch, batch_size)
            input_f = np.concatenate([temp_X, predictions], 0)
            input_r = np.concatenate([temp_X, temp_Y], 0)
            
            input = np.concatenate([[input_r],[input_f]])
            labels = np.concatenate([[np.ones((2, 1))], [np.zeros((2, 1))]])

            d_loss = self.discriminator.train_on_batch(input, labels)
            # ---------------------
            #  Train Generator
            # ---------------------
            valid_y = np.ones((batch_size, 1))
            temp_X = temp_X.reshape(self.latent_dim, batch_size, self.feature_num)
            g_loss = self.gan.train_on_batch(temp_Y, valid_y)

            #if epoch % print_output_every_n_steps == 0:
            #   self._summarize_performance(batch_size)
        
        return self.generator








