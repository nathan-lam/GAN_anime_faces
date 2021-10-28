import numpy as np
import os # get file names from path
import cv2 # image processing
import time # Track runtime
# import matplotlib.pyplot as plt

file_name = "Human_faces"
folder_path = "C:/Users/nthnt/PycharmProjects/GAN_faces_pycharm/Datasets/"

names_list = os.listdir(folder_path+file_name+"/")
n = len(names_list)

# Looking at a single image
data_path = folder_path + file_name + '/'
img = cv2.imread(data_path + names_list[0])
img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
"""
plt.imshow(img_cvt)
plt.axis('off')
plt.show()
"""

# building the dataset

# code to make the dataset in separate sessions
target_path = os.listdir(folder_path)
if f"np_{file_name}.npy" not in target_path:
    data_set = np.copy(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))
else:
    data_set = np.load(folder_path+f"np_{file_name}.npy")

n2 = len(data_set) # number of images already saved
m = len(names_list) - n2 # number of images not saved
print(f"Dataset currenly has {n2} images") # says how many images are left to be saved
print(f"{m} images left out of {n}")

start = time.time()
for i in range(n2+1, m):
    img = cv2.imread(data_path + names_list[i])  # read in image
    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # correct the color values
    data_set = np.concatenate((data_set, img_cvt.reshape(1, img.shape[0], img.shape[1], img.shape[2])), axis=0)
    if i % (n//100) == 0:
        print(f"{i}/{n} = {int(i/n*100)}% - {time.time() - start} seconds - Saved Check point")
        start = time.time()
        np.save(folder_path+f"np_{file_name}.npy", data_set)

