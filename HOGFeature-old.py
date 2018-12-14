import numpy as np
import cv2
import math
import glob
import sys

class MLP:
    def __init__(self, w1, b1, w2, b2, lr):
        self.l1 = FCLayer(w1, b1, lr)
        self.a1 = ReLU()
        self.l2 = FCLayer(w2, b2, lr)
        self.a2 = Sigmoid()

    # Mean square error function
    def MSE(self, prediction, target):
        return np.square(target - prediction).sum()/2

    # Gradient of Mean square - to use in backpropogation
    def MSEGrad(self, prediction, target):
        return - 2.0 * (target - prediction)

    # Shuffle the order so that model doesn't learn based-on input order
    def shuffle(self, X, y):
        idxs = np.arange(y.size)
        np.random.shuffle(idxs)
        return X[idxs], y[idxs]

    def train(self, X, y):
        prevError = sys.maxsize
        while True:
            X, y = self.shuffle(X, y)
            currentError = 0
            for i in range(len(X)):
                xi = np.expand_dims(X[i], axis=0)
                yi = np.expand_dims(y[i], axis=0)

                # Forward pass to train the network
                pred = self.l1.forward(xi)
                pred = self.a1.forward(pred)
                pred = self.l2.forward(pred)
                pred = self.a2.forward(pred)

                # Update loss values
                loss = self.MSE(pred, yi)

                # Backward pass to update weights
                grad = self.MSEGrad(pred, yi)
                grad = self.a2.backward(grad)
                grad = self.l2.backward(grad)
                grad = self.a1.backward(grad)
                self.l1.backward(grad)

                currentError += loss

            error = currentError / 20
            differenceInError = (prevError - error)

            # Stop the model if error is not updating
            if 0 < differenceInError < 0.0005:
                break
            prevError = error


    def predict(self, X):
        pred = self.l1.forward(X)
        pred = self.a1.forward(pred)
        pred = self.l2.forward(pred)
        pred = self.a2.forward(pred)
        # print(pred)
        pred = np.round(pred)
        return np.ravel(pred)

class FCLayer:

    def __init__(self, w, b, lr):
        self.lr = lr
        self.w = w
        self.b = b

    def forward(self, input):
        self.x = input
        activation = input.dot(self.w) + self.b
        return activation

    def backward(self, gradients):
        # Backward pass
        w1 = self.x.transpose().dot(gradients)
        x1 = gradients.dot(self.w.transpose())
        self.w = self.w - self.lr * w1
        self.b = self.b - self.lr * gradients
        return x1

class Sigmoid:

    def __init__(self):
        None

    def forward(self, inputs):
        input = self.getSigmoid(inputs)
        self.input = input
        return input

    def backward(self, gradients):
        gradients = gradients * ((1 - self.input) * self.input)
        return gradients

    def getSigmoid(self, n):
        result = 1.0 / (1.0 + np.exp(-n))
        return result

class ReLU:
    def __init__(self):
        None

    def forward(self, inputs):
        input = self.getReLU(inputs)
        self.input = input
        return input

    # Perfrom backward update which will discard all the inputs except > 0
    def backward(self, gradients):
        gradients = gradients * (self.input)
        gradients = self.getReLU(gradients)

        gradients[gradients > 0] = 1

        return gradients

    # Make all inputs less than 0 as 0
    def getReLU(self, n):
        result = np.maximum(n, 0)
        return result

def histogramOfGradients(image, gradient, gradientAngle):
    """
    This function returns an array containing histogram of gradients
    """
    height = image.shape[0]
    width = image.shape[1]
    histogramArray = []
    result = []
    counter = 0

    # For each block
    for row in range(0, height - 8, 8):
        for col in range(0, width - 8, 8):

            # For each cell in that block
            for i in range(row, row + 16, 8):
                for j in range(col, col + 16, 8):

                    histogramArray.append(getHistogramOfBin(i, j, gradientAngle, gradient))
                    counter += 1
                    if counter % 4 == 0:
                        # Combine 4 cells to form a block of 36 x 1 size
                        blockOfHistogramArray = np.array(histogramArray).reshape(1, 36)

                        # Normalize this block
                        blockNormalized = l2Normalized(blockOfHistogramArray)
                        result.append(blockNormalized)

                        histogramArray = []

    result = np.array(result)
    # Reshape in 1-D
    result = result.reshape(1, result.shape[0] * result.shape[2])

    return result


def getHistogramOfBin(i, j, gradientAngle, gradient):

    # Computes the 9 element 1-D array of bins
    d = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    histogramArray = np.zeros(shape=(1, 9))

    for row in range(i, i + 8):
        for col in range(j, j + 8):
            angle = gradientAngle[row, col]

            if angle >= 170:
                angle -= 180

            if angle in d:
                histogramArray[0, d.index(angle)] += gradient[row, col]
            else:
                if angle > 160:
                    high = 1 - ((180 - angle) / 20)
                    low = 1 - ((angle - 160) / 20)

                    histogramArray[0, 8] += gradient * high
                    histogramArray[0, 0] += gradient * low

                elif angle < 0:
                    high = 1 - ((abs(angle)) / 20)
                    low = 1 - ((160 + angle) / 20)

                    histogramArray[0, 0] += gradient * high             # <---------------
                    histogramArray[0, 8] += gradient * low

                else:
                    high = ceil_key(angle)
                    low = floor_key(angle)

                    weightedHigh, weightedLow = getWeighted(angle, high, low)

                    histogramArray[0, d.index(low)] += gradient[row][col] * weightedLow
                    histogramArray[0, d.index(high)] += gradient[row][col] * weightedHigh

    return histogramArray

def l2Normalized (histogramArray):
    array = np.square(histogramArray)
    dist = math.sqrt(array.sum())

    if dist != 0.0:
        histogramArray = histogramArray / dist
    return histogramArray

def getWeighted(angle, high, low):
    # Get weighted values of the angle to distribute the magnitude accordingly
    weightedHigh = 1 - ((high - angle) / (high-low))                    # <----------------
    weightedLow = 1 - ((angle - low) / (high-low))
    return weightedHigh, weightedLow

def floor_key(key):
    # Finds closet smaller bin to given angle
    d = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    if key in d:
        return key
    return max(k for k in d if k <= key)

def ceil_key(key):
    # Finds closest larger bin to given angle
    d = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    if key in d:
        return key
    return min(k for k in d if k >= key)

def normalizeHOG(gradient):
    # Normalizing HOG
    gradient = np.array(gradient)
    gradient = ((gradient - np.min(gradient))/np.ptp(gradient))
    return gradient

def normalize(gradient):
    # Gradient normalization
    gradient = np.array(gradient)
    gradient = ((gradient - np.min(gradient))/np.ptp(gradient))*255
    return gradient

def getGradientX(imgArr, height, width):
    """

    :param imgArr: NxM image to find the gradient
    :param height: height of the array
    :param width: width of the array
    :return: Array representing the gradient
    """
    imageData = np.empty(shape=(height, width))
    for i in range(3, height - 5):
        for j in range(3, imgArr[i].size - 5):
            if liesInUnderRegion(imgArr, i, j):
                imageData[i + 1][j + 1] = None
            else:
                imageData[i + 1][j + 1] = prewittAtX(imgArr, i, j)

    return abs(imageData)


def getGradientY(imgArr, height, width):
    """
    Similar to the getGradientX function for Y
    """
    imageData = np.empty(shape=(height, width))
    for i in range(3, height - 5):
        for j in range(3, imgArr[i].size - 5):
            if liesInUnderRegion(imgArr, i, j):
                imageData[i + 1][j + 1] = None
            else:
                imageData[i + 1][j + 1] = prewittAtY(imgArr, i, j)

    return abs(imageData)


def getMagnitude(Gx, Gy, height, width):
    """
    Computes the gradient magnitude by taking square root of gx-square plus gy-square
    :param Gx: xGradient of the image array
    :param Gy: yGradient of the image array
    :param height:
    :param width:
    :return: array representing edge magnitude
    """
    gradientData = np.empty(shape=(height, width))
    for row in range(height):
        for column in range(width):
            gradientData[row][column] = ((Gx[row][column] ** 2 + Gy[row][column] ** 2) ** 0.5) / 1.4142
    return gradientData


def getAngle(Gx, Gy, height, width):
    """
    Computes the edge angle by taking the tan inverse of yGradient/xGradient
    :param Gx:
    :param Gy:
    :param height:
    :param width:
    :return: integer array representing the edge angle
    """
    gradientData = np.empty(shape=(height, width))
    angle = 0
    for i in range(height):
        for j in range(width):
            if Gx[i][j] == 0:
                if Gy[i][j] > 0:
                    angle = 90
                else:
                    angle = -90
            else:
                angle = math.degrees(math.atan(Gy[i][j] / Gx[i][j]))
            if angle < 0:
                angle += 360
            gradientData[i][j] = angle
    return gradientData


def liesInUnderRegion(imgArr, i, j):
    return imgArr[i][j] == None or imgArr[i][j + 1] == None or imgArr[i][j - 1] == None or imgArr[i + 1][j] == None or \
           imgArr[i + 1][j + 1] == None or imgArr[i + 1][j - 1] == None or imgArr[i - 1][j] == None or \
           imgArr[i - 1][j + 1] == None or imgArr[i - 1][j - 1] == None


def prewittAtX(imageData, row, column):
    prewittX = (1.0 / 3.0) * np.array([[-1, 0, 1],
                                       [-1, 0, 1],
                                       [-1, 0, 1]])
    horizontal = 0
    for i in range(0, 3):
        for j in range(0, 3):
            horizontal += imageData[row + i, column + j] * prewittX[i, j]
    return horizontal


def prewittAtY(imageData, row, column):
    prewittY = (1.0 / 3.0) * np.array([[1, 1, 1],
                                       [0, 0, 0],
                                       [-1, -1, -1]])
    vertical = 0
    for i in range(0, 3):
        for j in range(0, 3):
            vertical += imageData[row + i, column + j] * prewittY[i, j]
    return vertical

def convertImage(img):
    # Convert image to gray scale using given formula
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    grayImage = R * 299./1000 + G * 587./1000 + B * 114./1000
    return grayImage

def importTrainingImages():
    # Import training images from the folder
    image_list = []
    result = []

    for filename in glob.glob('Human/Train_Positive/*.bmp'):
        im = cv2.imread(filename)
        image = convertImage(im)
        image_list.append(image)
        result.append(1.0)

    for filename in glob.glob('Human/Train_Negative/*.bmp'):
        im = cv2.imread(filename)
        image = convertImage(im)
        image_list.append(image)
        result.append(0.0)

    return image_list, result

def importTestingImages():
    # Import testing images from the folder
    testImageList = []
    for filename in glob.glob('Human/Test_Positive/*.bmp'):
        im = cv2.imread(filename)
        image = convertImage(im)
        testImageList.append(image)

    for filename in glob.glob('Human/Test_Neg/*.bmp'):
        im = cv2.imread(filename)
        image = convertImage(im)
        testImageList.append(image)

    return testImageList

def performHOGOperations(images):
    # Performs HOG operations on imported images
    X_train = []
    count = 0
    for img in images:
        count += 1
        height = img.shape[0]
        width = img.shape[1]

        # Normalized Horizontal Gradient
        Gx = normalize(getGradientX(img, height, width))
        # cv2.imwrite(str(count)  + '_XGradient.jpg', Gx)

        # Normalized Vertical Gradient
        Gy = normalize(getGradientY(img, height, width))
        # cv2.imwrite(str(count) + '_YGradient.jpg', Gy)

        # Normalized Edge Magnitude
        gradient = normalize(getMagnitude(Gx, Gy, height, width))
        # cv2.imwrite(str(count) + '_Gradient.jpg', gradient)

        # Edge angle
        gradientAngle = getAngle(Gx, Gy, height, width)

        X_train.append(normalizeHOG(np.array(histogramOfGradients(img, gradient, gradientAngle))))


    X_train = np.array(X_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2])

    return X_train


def prediction(N, X_train, X_test, result, layerSize):

    # Learning rate
    lr = .0001

    # Input layer
    w1 = np.random.normal(0, .1, size=(N, layerSize))
    b1 = np.random.normal(0, .1, size=(1, layerSize))

    # Hidden Layer
    w2 = np.random.normal(0, .1, size=(layerSize, 1))
    b2 = np.random.normal(0, .1, size=(1, 1))

    mlp = MLP(w1, b1, w2, b2, lr)

    # Train the network
    mlp.train(X_train, np.array(result))
    solution = mlp.predict(X_test)
    print(solution, 'For Layer: ', layerSize)

# Get training images with HOG features
training_images, result = importTrainingImages()
X_train = performHOGOperations(training_images)

# Get testing images with HOG feature
testing_images = importTestingImages()
X_test = performHOGOperations(testing_images)

prediction(X_train.shape[1], X_train, X_test, result, 500)
prediction(X_train.shape[1], X_train, X_test, result, 500)
prediction(X_train.shape[1], X_train, X_test, result, 500)
prediction(X_train.shape[1], X_train, X_test, result, 500)
prediction(X_train.shape[1], X_train, X_test, result, 500)
prediction(X_train.shape[1], X_train, X_test, result, 500)
prediction(X_train.shape[1], X_train, X_test, result, 500)
prediction(X_train.shape[1], X_train, X_test, result, 500)
prediction(X_train.shape[1], X_train, X_test, result, 500)
prediction(X_train.shape[1], X_train, X_test, result, 500)


