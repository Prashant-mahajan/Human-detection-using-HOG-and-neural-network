import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('crop001030c.bmp')

# Conver the image
R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]
grayImage = R * 299./1000 + G * 587./1000 + B * 114./1000

print('Hello')

plt.imshow(grayImage)
cv2.imwrite('Output.jpg', grayImage)
