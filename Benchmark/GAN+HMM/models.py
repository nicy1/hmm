from keras import layers
import keras
import numpy as np



class Generator():
    def __init__(self, latent_dim, feature_num):
        self.latent_dim = latent_dim
        self.feature_num = feature_num


    def build_generator(self):
        generator_input = keras.Input(shape=(self.latent_dim,self.feature_num))
        x = layers.LSTM(75,return_sequences=True)(generator_input)
        x = layers.LSTM(25)(x)
        x = layers.Dense(1)(x)
        x = layers.LeakyReLU()(x)
        generator = keras.models.Model(generator_input, x)
        generator.summary()
        return generator


class Discriminator():
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    
    def build_discriminator(self):
        discriminator_input = layers.Input(shape=(self.latent_dim+1,1))
        y = layers.Dense(72)(discriminator_input)
        y = layers.LeakyReLU(alpha=0.05)(y)
        y = layers.Dense(100)(y)
        y = layers.LeakyReLU(alpha=0.05)(y)
        y = layers.Dense(10)(y)
        y = layers.LeakyReLU(alpha=0.05)(y)
        y = layers.Dense(1,activation='sigmoid')(y)
        discriminator = keras.models.Model(discriminator_input, y)
        discriminator.summary()
        return discriminator


















