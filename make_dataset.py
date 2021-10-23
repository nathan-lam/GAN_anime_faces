import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt

file_name = "anime_faces"
folder_path = f"C:/Users/nthnt/PycharmProjects/GAN_faces_pycharm/Datasets/{file_name}/"

names_list = os.listdir(folder_path)
n = len(names_list)
print(f"Importing {n} images")

# Looking at a single image
img = cv2.imread(folder_path + names_list[0])
img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
"""
plt.imshow(img_cvt)
plt.axis('off')
plt.show()
"""

# building the dataset
start = time.time()
data_set = np.copy(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))
n = len(names_list)
for i in range(1, n):
    if i % (n//100) == 0:
        print(f"{i}/{n} = {int(i/n*100)}%")
    img = cv2.imread(folder_path + names_list[i])  # read in image
    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # correct the color values
    data_set = np.concatenate((data_set, img_cvt.reshape(1, img.shape[0], img.shape[1], img.shape[2])), axis=0)
print(f"Process took {time.time() - start} seconds")
np.save("C:/Users/nthnt/PycharmProjects/ML_projects/anime_faces.npy", data_set)
