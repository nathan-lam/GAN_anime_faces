import tensorflow as tf

from data_prep import train_dataset
from tensorflow.keras import layers

noise_dim = 1000

def make_generator_model():
    model = tf.keras.Sequential()  # layer object
    model.add(layers.Dense(4 * 4 * 256, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())  # does a form of normalizing
    model.add(layers.LeakyReLU())
    L = 1

    model.add(layers.Reshape((4, 4, 256)))  # making the layer 3D
    #print(f"Layer {L} shape: {model.output_shape}")
    assert model.output_shape == (None, 4, 4, 256)  # Note: None is the batch size

    # opposite of a convolution
    L += 1
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    #print(f"Layer {L} shape: {model.output_shape}")
    # assert model.output_shape == (None, 4, 4, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    L += 1
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    #print(f"Layer {L} shape: {model.output_shape}")
    # assert model.output_shape == (None, 8, 8, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    L += 1
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    #print(f"Layer {L} shape: {model.output_shape}")
    # assert model.output_shape == (None, 8, 8, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    L += 1
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    #print(f"Layer {L} shape: {model.output_shape}")
    # assert model.output_shape == (None, 64, 64, 3)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[64, 64, 3]))  # input dimensions of 1 image
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
