import imageio
import os

path = "Attempt3/checkpoint_images/"
file_names = os.listdir(path)

images = [imageio.imread(path+file) for file in file_names]
imageio.mimsave('image_epoch.gif', images)
