import numpy as np
import cv2
import math
from PIL import Image
import glob
# # import matplotlib.pyplot as plt
#
# def __init__():
#     d = [0, 20, 40, 60, 80, 100, 120, 140, 160]
#
# def gaussianSmoothing(image):
#     """
#     Applies 7x7 Gaussian Filter to the image by convolution operation
#     :type image: object
#     """
#     imageArray = np.array(image)
#     gaussianArr = np.array(image)
#     sum = 0
#
#     for i in range(3, image.shape[0] - 3):
#         for j in range(3, image.shape[1] - 3):
#             sum = applyGaussianFilterAtPoint(imageArray, i, j)
#             gaussianArr[i][j] = sum
#
#     return gaussianArr
#
# def histogramOfGradients(image, gradient,gradientAngle):
#
#     height = image.shape[0]
#     width = image.shape[1]
#     histogramArray = np.empty(shape=(height/8, width))
#
#     for i in range(4, height - 4):
#         for j in range(4, width - 4):
#
#             for k in range()
#             if gradientAngle[i][j] >= 170:
#                 gradientAngle[i][j] -= 180
#                 angle = gradientAngle[i][j]
#                 if(angle not in d):
#                     low = floor_key(angle)
#                     high = ceil_key(angle)
#                     bin0 = (high - angle)/(high - low)
#                     bin1 = (angle - low)/(high-low)
#
#
#             low = floor_key()
#
#
# def floor_key(key):
#
#     if key in d:
#         return key
#     return max(k for k in d if k < key)
#
# def ceil_key(key):
#     if key in d:
#         return key
#     return min(k for k in d if k > key)
#
# def normalizedFeatureVector(histogramArray):
#
#
#
# def applyGaussianFilterAtPoint(imageData, row, column):
#     sum = 0
#
#     for i in range(row - 3, row + 4):
#         for j in range(column - 3, column + 4):
#             sum += gaussian_filter[i - row + 3][j - column + 3] * imageData[i][j]
#
#     return sum
#
# def getGradientX(imgArr, height, width):
#     """
#     :param imgArr: NxM image to find the gradient
#     :param height: height of the array
#     :param width: width of the array
#     :return: Array representing the gradient
#     """
#     imageData = np.empty(shape=(height, width))
#     for i in range(3, height - 5):
#         for j in range(3, imgArr[i].size - 5):
#             if liesUnder(imgArr, i, j):
#                 imageData[i + 1][j + 1] = None
#             else:
#                 imageData[i + 1][j + 1] = prewittAtX(imgArr, i, j)
#
#     return abs(imageData)
#
#
# def getGradientY(imgArr, height, width):
#     """
#     Similar to the getGradientX function for Y
#     """
#     imageData = np.empty(shape=(height, width))
#     for i in range(3, height - 5):
#         for j in range(3, imgArr[i].size - 5):
#             if liesUnder(imgArr, i, j):
#                 imageData[i + 1][j + 1] = None
#             else:
#                 imageData[i + 1][j + 1] = prewittAtY(imgArr, i, j)
#
#     return abs(imageData)
#
#
# def getMagnitude(Gx, Gy, height, width):
#     """
#     Computes the gradient magnitude by taking square root of gx-square plus gy-square
#     :param Gx: xGradient of the image array
#     :param Gy: yGradient of the image array
#     :param height:
#     :param width:
#     :return: array representing edge magnitude
#     """
#     gradientData = np.empty(shape=(height, width))
#     for row in range(height):
#         for column in range(width):
#             gradientData[row][column] = ((Gx[row][column] ** 2 + Gy[row][column] ** 2) ** 0.5) / 1.4142
#     return gradientData
#
#
# def getAngle(Gx, Gy, height, width):
#     """
#     Computes the edge angle by taking the tan inverse of yGradient/xGradient
#     :param Gx:
#     :param Gy:
#     :param height:
#     :param width:
#     :return: integer array representing the edge angle
#     """
#     gradientData = np.empty(shape=(height, width))
#     angle = 0
#     for i in range(height):
#         for j in range(width):
#             if Gx[i][j] == 0:
#                 if Gy[i][j] > 0:
#                     angle = 90
#                 else:
#                     angle = -90
#             else:
#                 angle = math.degrees(math.atan(Gy[i][j] / Gx[i][j]))
#             if angle < 0:
#                 angle += 360
#             gradientData[i][j] = angle
#     return gradientData
#
# def liesUnder(imgArr, i, j):
#     return imgArr[i][j] == None or imgArr[i][j + 1] == None or imgArr[i][j - 1] == None or imgArr[i + 1][j] == None or \
#            imgArr[i + 1][j + 1] == None or imgArr[i + 1][j - 1] == None or imgArr[i - 1][j] == None or \
#            imgArr[i - 1][j + 1] == None or imgArr[i - 1][j - 1] == None
#
# def prewittAtX(imageData, row, column):
#     sum = 0
#     horizontal = 0
#     for i in range(0, 3):
#         for j in range(0, 3):
#             horizontal += imageData[row + i, column + j] * prewittX[i, j]
#     return horizontal
#
# def prewittAtY(imageData, row, column):
#     sum = 0
#     vertical = 0
#     for i in range(0, 3):
#         for j in range(0, 3):
#             vertical += imageData[row + i, column + j] * prewittY[i, j]
#     return vertical
#
#
#
# # Load the images
def convertImage(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    grayImage = R * 299./1000 + G * 587./1000 + B * 114./1000
    return grayImage

image_list = []
result = []
for filename in glob.glob('Human/Train_Positive/*.bmp'):
    im = cv2.imread(filename)
    image = convertImage(im)
    cv2.imwrite('Output_filename.jpg', image)
    image_list.append(image)
    result.append(1)

for filename in glob.glob('Human/Train_Negative/*.bmp'):
    im = cv2.imread(filename)
    image = convertImage(im)
    image_list.append(image)
    result.append(0)


print(result)
print(image_list)
#
# # img = cv2.imread('crop001030c.bmp')
#
# prewittX = (1.0 / 3.0) * np.array([[-1, 0, 1],
#                                    [-1, 0, 1],
#                                    [-1, 0, 1]])
#
# prewittY = (1.0 / 3.0) * np.array([[1, 1, 1],
#                                    [0, 0, 0],
#                                    [-1, -1, -1]])
#
# # Conver the image
# R = img[:, :, 0]
# G = img[:, :, 1]
# B = img[:, :, 2]
# grayImage = R * 299./1000 + G * 587./1000 + B * 114./1000
# cv2.imwrite('Output.jpg', grayImage)
#
# height = grayImage.shape[0]
# width = grayImage.shape[1]
#
# # Normalized Gaussian Smoothing
# # gaussianData = gaussianSmoothing(grayImage)
# # print(gaussianData)
# # cv2.imwrite('Outputs/filter_gauss.jpg', gaussianData)
#
# # Normalized Horizontal Gradient
# Gx = getGradientX(grayImage, height, width)
# cv2.imwrite('XGradient.jpg', Gx)
#
# # Normalized Vertical Gradient
# Gy = getGradientY(grayImage, height, width)
# cv2.imwrite('YGradient.jpg', Gy)
#
# # Normalized Edge Magnitude
# gradient = getMagnitude(Gx, Gy, height, width)
# cv2.imwrite('Gradient.jpg', gradient)
#
# # Edge angle
# gradientAngle = getAngle(Gx, Gy, height, width)
