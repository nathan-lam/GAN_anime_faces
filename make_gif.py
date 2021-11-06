import imageio
import os

#from main import attempt


attempt = 4

path = f"Attempt{attempt}/checkpoint_images/"
file_names = os.listdir(path)

images = [imageio.imread(path+file) for file in file_names]
imageio.mimsave(f'Attempt{attempt}/image_epoch.gif', images)
