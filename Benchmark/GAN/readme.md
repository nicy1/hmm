GAN 
----
Implementation of a Generative adversarial network (GAN) that uses sequential/temporal dataset in python3

File 'readfile.py'
-------------------
This file selects the features to be used as observations and hidden states. It also splits the dataset in train (x_train: observations, y_train: hidden states) and test data (x_test: observations, y_test: hidden states as targets), 80% and 20% respectively.

File 'models.py'
-----------------
It contains two classes for building GAN Genarator and Discriminator models.

File 'gan.py'
--------------
It contains:
   - Function that builds a complete GAN model combining the Generator and Discriminator models.
   - Function for training the Generator, Discriminator and the combined GAN models.

File 'main.py'
---------------
It uses all the files cited above, sets the initial parameters of GAN, trains the model (GAN) with training function, and tests the model using the Generator model.

The prediction accuracy of this model is low, around 69%. We notice that, training the model with x_train data (observations), the model tends to learn how the sequence of observations is composed instead of that of the hidden states, that is why we get a low accuracy. However, training the model with y_train data (hidden states), will produce the best result because the model will learn how the sequence of hidden states is composed. Thus, we will try to fix this problem in the futur work in order to increase the performance of the model.
