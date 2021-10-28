import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

file_path = "C:/Users/nthnt/PycharmProjects/GAN_faces_pycharm/Datasets/np_anime_faces.npy"
data_set = np.load(file_path)

data_normal = (data_set - 127.5) / 127.5
data_normal = data_normal.astype("float32")

BUFFER_SIZE = 60000
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices(data_normal).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


