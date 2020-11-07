from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from models import Discriminator, Generator
import pandas as pd
import sys
import random
import numpy as np



class GAN():
    def __init__(self, latent_dim=100):
        self.data_rows = 1
        self.data_cols = 1
        self.latent_dim = latent_dim
        
        self.data_shape = (self.data_rows, self.data_cols)
        self.discriminator = Discriminator(self.data_shape).build_discriminator()
        self.generator = Generator(self.latent_dim, self.data_shape).build_generator()

        self.gan_model = self._build_gan()



    def _build_gan(self):
        self.discriminator.trainable = False
        # Connect discriminator and generator
        model = Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        # Compile model
        optimizer = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
        return model
    

    
    def load_data(self, x_train, y_train):
        # Reshape into a batch of inputs for the network
        self.X_train = np.array(x_train).reshape(len(x_train), 1)      
        self.Y_train = y_train
        print("X_train")
        print(np.array(self.X_train).shape)
        print("Y_train")
        print(np.array(self.Y_train).shape)

  
    
    def _generate_real_samples(self, batch_size):
        idx = np.random.randint(0, len(self.X_train), batch_size)
        x_real = self.X_train[idx]
        y_real = np.ones((batch_size, 1))
        return x_real, y_real



    def _generate_fake_samples(self, batch_size):
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        x_fake = self.generator.predict(noise)
        y_fake = np.zeros((batch_size, 1))
        return x_fake, y_fake



    def train(self, epochs=100, batch_size=128, print_output_every_n_steps=20):
        for epoch in range(epochs):
            # Get randomly real data
            x_real, y_real = self._generate_real_samples(batch_size)
            # Update discriminator model weights and get the loss for the real data
            d_loss_real = self.discriminator.train_on_batch(x_real, y_real)
            # Generate fake samples
            x_generated, y_fake = self._generate_fake_samples(batch_size)
            # Update discriminator model weights and get the loss for the fake (generated) data
            d_loss_fake = self.discriminator.train_on_batch(x_generated, y_fake)
            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.gan_model.train_on_batch(noise, y_real)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

            if epoch % print_output_every_n_steps == 0:
               print("Generated data: " , [x[0][0] for x in x_generated])
        
        return self.generator








