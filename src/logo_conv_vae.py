from keras.layers import Lambda, Input, Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, Flatten, MaxPool2D, Reshape, UpSampling2D, Conv2DTranspose, ReLU
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from load_data import load


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


data = load()

image_size = data[0].shape
original_dim = image_size

# x_train = np.reshape(x_train, [-1, original_dim])
# x_test = np.reshape(x_test, [-1, original_dim])
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
data = data.astype('float32') / 255

# network parameters
input_shape = original_dim
batch_size = 128
latent_dim = 2
epochs = 1

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')

# first conv block
enc_conv1 = Conv2D(32, kernel_size=3, padding='same', strides=2, name='enc_conv1')(inputs)
enc_conv1_act = LeakyReLU(alpha=0.2)(enc_conv1)
enc_conv1_drop = Dropout(0.2)(enc_conv1_act)

# second conv block
enc_conv2 = Conv2D(64, kernel_size=3, padding='same', strides=2, name='enc_conv2')(enc_conv1_drop)
enc_conv2_norm = BatchNormalization(momentum=0.8)(enc_conv2)
enc_conv2_act = LeakyReLU(alpha=0.2)(enc_conv2_norm)
enc_conv2_drop = Dropout(0.2)(enc_conv2_act)

# third conv block
enc_conv3 = Conv2D(128, kernel_size=3, padding='same', strides=2, name='enc_conv3')(enc_conv2_drop)
enc_conv3_norm = BatchNormalization(momentum=0.8)(enc_conv3)
enc_conv3_act = LeakyReLU(alpha=0.2)(enc_conv3_norm)
enc_conv3_drop = Dropout(0.2)(enc_conv3_act)

# fourth conv block
enc_conv4 = Conv2D(256, kernel_size=3, padding='same', strides=2, name='enc_conv4')(enc_conv3_drop)
enc_conv4_norm = BatchNormalization(momentum=0.8)(enc_conv4)
enc_conv4_act = LeakyReLU(alpha=0.2)(enc_conv4_norm)
enc_conv4_drop = Dropout(0.2)(enc_conv4_act)

# flatten and condense
flatten = Flatten()(enc_conv4_drop)
# x = Dense(intermediate_dim, activation='relu')(flatten)
z_mean = Dense(latent_dim, name='z_mean')(flatten)
z_log_var = Dense(latent_dim, name='z_log_var')(flatten)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_logo_conv_encoder.png', show_shapes=True)


# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')

# reshape
dec_fc1 = Dense(2 * 2 * 256, activation='relu')(latent_inputs)
dec_reshape = Reshape((2, 2, 256))(dec_fc1)

# deconv block 1
dec_deconv1 = Conv2DTranspose(128, kernel_size=3, padding='same', strides=2, name='dec_deconv1')(dec_reshape)
dec_deconv1_norm = BatchNormalization(momentum=0.8)(dec_deconv1)
dec_deconv1_act = ReLU()(dec_deconv1_norm)

# deconv block 2
dec_deconv2 = Conv2DTranspose(64, kernel_size=3, padding='same', strides=2, name='dec_deconv2')(dec_deconv1_act)
dec_deconv2_norm = BatchNormalization(momentum=0.8)(dec_deconv2)
dec_deconv2_act = ReLU()(dec_deconv2_norm)

# deconv block 3
dec_deconv3 = Conv2DTranspose(32, kernel_size=3, padding='same', strides=2, name='dec_deconv3')(dec_deconv2_act)
dec_deconv3_norm = BatchNormalization(momentum=0.8)(dec_deconv3)
dec_deconv3_act = ReLU()(dec_deconv3_norm)

# deconv block 4
dec_deconv4 = Conv2DTranspose(3, kernel_size=3, padding='same', strides=2, activation='sigmoid', name='dec_deconv4')(dec_deconv3_act)


# outputs = Dense(original_dim, activation='sigmoid')(dec_deconv1)

# instantiate decoder model
decoder = Model(latent_inputs, dec_deconv4, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_logo_conv_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_logo_conv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim[0] * original_dim[1] * original_dim[2]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae,
               to_file='vae_logo_conv.png',
               show_shapes=True)

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(data,
                epochs=epochs,
                batch_size=batch_size)
        vae.save_weights('vae_logo_conv.h5')

    # plot_results(models,
    #              data,
    #              batch_size=batch_size,
    #              model_name="vae_mlp")