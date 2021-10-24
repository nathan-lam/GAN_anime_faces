import numpy as np
import os # get file names from path
import cv2 # image processing
import time # Track runtime


file_name = "Human_faces"
folder_path = f"C:/Users/nthnt/PycharmProjects/GAN_faces_pycharm/Datasets/"

names_list = os.listdir(folder_path+file_name+"/")
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

target_path = os.listdir(folder_path)
if f"np_{file_name}.npy" not in target_path:
    data_set = np.copy(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))
else:
    data_set = np.load(folder_path+f"np_{file_name}.npy")

n = len(names_list) - len(data_set) + 1
for i in range(1, 521):
    if i % (n//100) == 0:
        print(f"{i}/{n} = {int(i/n*100)}%")
    img = cv2.imread(folder_path + names_list[i])  # read in image
    img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # correct the color values
    data_set = np.concatenate((data_set, img_cvt.reshape(1, img.shape[0], img.shape[1], img.shape[2])), axis=0)
print(f"Process took {time.time() - start} seconds")
np.save(folder_path+f"np_{file_name}.npy", data_set)

