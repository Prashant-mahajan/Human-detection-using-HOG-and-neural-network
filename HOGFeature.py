import numpy as np
import cv2
import math
import warnings
import glob
import sys


class MLP:
    def __init__(self, w1, b1, w2, b2, lr):
        # Input layer with Relu Activation
        self.inputLayer = FCLayer(w1, b1, lr)
        self.inputLayerActivation = ReLU()
        # l2,a2 = HiddenLayer with Sigmoid Activation
        self.hiddenLayer = FCLayer(w2, b2, lr)
        self.hiddenLayerActivation = Sigmoid()

    def MSE(self, prediction, target):
        # Mean Squared Error
        return (np.square(target - prediction).sum()) / 2

    def MSEGrad(self, prediction, target):
        # Mean Squared Error Gradient
        return - 2.0 * (target - prediction)

    def shuffle(self, x, y):
        idxs = np.arange(y.size)
        np.random.shuffle(idxs)
        return x[idxs], y[idxs]

    def train(self, X, y):
        prevError = sys.maxsize
        while True:
            # shuffle to increase accuracy and to avoid the Network to simply learn the input order
            X, y = self.shuffle(X, y)
            currentError = 0
            for i in range(len(X)):
                xi = np.expand_dims(X[i], axis=0)
                yi = np.expand_dims(y[i], axis=0)

                # Forward Pass
                prediction = self.inputLayer.forward(xi)
                prediction = self.inputLayerActivation.forward(prediction)
                prediction = self.hiddenLayer.forward(prediction)
                prediction = self.hiddenLayerActivation.forward(prediction)

                # Update Mean Squared Error
                currentError += self.MSE(prediction, yi)

                # Backward Pass, updates weights according to error
                gradients = self.MSEGrad(prediction, yi)
                gradients = self.hiddenLayerActivation.backward(gradients)
                gradients = self.hiddenLayer.backward(gradients)
                gradients = self.inputLayerActivation.backward(gradients)
                self.inputLayer.backward(gradients)

            errorOfEpoch = currentError / 20               # <-------------
            delta = (prevError - errorOfEpoch)
            # Stop condition when Mean Squared Errors do not change much
            if 0 < delta < 0.0001:
                break
            prevError = errorOfEpoch

    def predict(self, X):
        pred = self.inputLayer.forward(X)
        pred = self.inputLayerActivation.forward(pred)
        pred = self.hiddenLayer.forward(pred)
        pred = self.hiddenLayerActivation.forward(pred)
        pred = np.round(pred)
        return np.ravel(pred)


class FCLayer:

    def __init__(self, w, b, lr):
        self.lr = lr
        self.w = w  # Each column represents all the weights going into an output node
        self.b = b

    def forward(self, input):
        self.x = input
        activation = input.dot(self.w) + self.b
        return activation

    def backward(self, gradients):
        # Update weights and return gradient error for next layer.
        w1 = self.x.transpose().dot(gradients)
        x1 = gradients.dot(self.w.transpose())
        self.w = self.w - self.lr * w1
        self.b = self.b - self.lr * gradients
        return x1


class Sigmoid:

    def __init__(self):
        return

    def forward(self, input):
        sigmoid = self.getSigmoidActivation(input)
        self.sigmoid = sigmoid
        return sigmoid

    def backward(self, gradients):
        # Computer Derivative of Sigmoid function i.e.
        # Grad * ((1-Sigmoid())* Sigmoid)

        # We store input after taking sigmoid, so no need to calculate again.
        gradients = gradients * ((1 - self.sigmoid) * self.sigmoid)
        return gradients

    def getSigmoidActivation(self, n):
        result = 1.0 / (1.0 + np.exp(-n))
        return result


class ReLU:
    def __init__(self):
        return

    def forward(self, inputs):
        input = self.getReLUActivation(inputs)
        self.input = input
        return input

    def backward(self, gradients):
        gradients = gradients * self.input
        # Compute Derivative of Relu ActivationFunction i.e.
        # anything less than or equal to 0 is 0 and anything greater than 0 is 1

        # 0 condition is handled by the ReluActivation Function itself
        gradients = self.getReLUActivation(gradients)

        # 1 condition
        gradients[gradients > 0] = 1

        return gradients

    def getReLUActivation(self, n):
        # Convert anything less than 0 to 0
        result = np.maximum(n, 0)
        return result


def initializeMLP(N, hiddenLayerSize):
    # Learning Rate
    lr = .0001

    # w1,b1 - Weight and bias of input layer
    w1 = np.random.normal(0, .1, size=(N, hiddenLayerSize))
    b1 = np.random.normal(0, .1, size=(1, hiddenLayerSize))

    # w2,b2 - Weight and bias of hidden layer
    w2 = np.random.normal(0, .1, size=(hiddenLayerSize, 1))
    b2 = np.random.normal(0, .1, size=(1, 1))

    return MLP(w1, b1, w2, b2, lr)


def trainAndPredictMLP(N, hiddenLayerSize, X_train, result_array, X_test):
    mlp = initializeMLP(N, hiddenLayerSize)
    mlp.train(X_train, np.array(result_array))
    solution = mlp.predict(X_test)
    print(hiddenLayerSize,solution)


def getTestImages():
    test_image_list = []
    # Get all Positive Images
    for filename in glob.glob('Human/Test_Positive/*.bmp'):
        test_image_list.append(getImage(filename))
    # Get all Negative Images
    for filename in glob.glob('Human/Test_Neg/*.bmp'):
        test_image_list.append(getImage(filename))
    return test_image_list


def L2Normalize(histogramArray):
    squaredhistogramArray = np.square(histogramArray)
    distance = math.sqrt(squaredhistogramArray.sum())

    # If distance is 0, that means array cannot be L2 Normalized.
    if distance != 0.0:
        histogramArray = histogramArray / distance
    return histogramArray


def getLowerBin(angle):
    histogramBins = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    return max(bin for bin in histogramBins if bin <= angle)


def getHigherBin(angle):
    histogramBins = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    return min(bin for bin in histogramBins if bin >= angle)


def computeWeightedGradientAndAddToHistogram(angle, edgeMagnitude, histogramArray):
    histogramBins = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    # If angle is greater than 160, we know it contributes to the bins centered at 0 and 160.
    if angle > 160:
        distanceFromHigherBin = 1 - ((180 - angle) / 20)
        distanceFromLowerBin = 1 - ((angle - 160) / 20)

        histogramArray[0, 8] += edgeMagnitude * distanceFromHigherBin
        histogramArray[0, 0] += edgeMagnitude * distanceFromLowerBin

    # If angle is less than 0, we know it contributes to the bins centered at 0 and 160.
    elif angle < 0:
        distanceFromHigherBin = 1 - (abs(angle) / 20)
        distanceFromLowerBin = 1 - ((160 + angle) / 20)

        histogramArray[0, 0] += edgeMagnitude * distanceFromHigherBin
        histogramArray[0, 8] += edgeMagnitude * distanceFromLowerBin

    else:
        higherBin = getHigherBin(angle)
        lowerBin = getLowerBin(angle)

        distanceFromHigherBin = 1 - ((higherBin - angle) / (higherBin - lowerBin))
        distanceFromLowerBin = 1 - ((angle - lowerBin) / (higherBin - lowerBin))

        histogramArray[0, histogramBins.index(higherBin)] += edgeMagnitude * distanceFromHigherBin
        histogramArray[0, histogramBins.index(lowerBin)] += edgeMagnitude * distanceFromLowerBin


def getHistogramOfCell(i, j, gradientAngle, gradient):
    histogramBins = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    histogramArray = np.zeros(shape=(1, 9))
    # Cell Iteration (8x8)
    for k in range(i, i + 8):
        for l in range(j, j + 8):
            angle = gradientAngle[k, l]
            # Subtract Angle to convert it to the Unsigned Representation.
            if angle >= 170:
                angle -= 180

            # If angle is at exactly the centre of the bin
            if angle in histogramBins:
                histogramArray[0, histogramBins.index(angle)] += gradient[k, l]
            else:
                computeWeightedGradientAndAddToHistogram(angle, gradient[k][l], histogramArray)

    return histogramArray


def getHOGFeatureOfImage(height, width, gradient, gradientAngle):
    histogramArray = []
    imageHistogram = []
    counter = 0
    # Block Overlap/ Step Size Iteration
    for row in range(0, height - 8, 8):
        for col in range(0, width - 8, 8):

            # Cell Size Iteration
            for i in range(row, row + 16, 8):
                for j in range(col, col + 16, 8):

                    histogramOfCell = getHistogramOfCell(i, j, gradientAngle, gradient)
                    histogramArray.append(histogramOfCell)
                    counter += 1

                    # Combine cells when block size (4 - 2x2) is reached.
                    if counter % 4 == 0:
                        # Combine 4 [1,9] arrays as 1-D array [1,36]
                        histogramArrayOfBlock = np.array(histogramArray).reshape(1, 36)

                        # Use L2 Normalization for the block
                        normalizedHistogramArray = L2Normalize(histogramArrayOfBlock)
                        imageHistogram.append(normalizedHistogramArray)

                        # Reset Histogram Array as already added to result.
                        histogramArray = []

    imageHistogram = np.array(imageHistogram)
    # Reshape to 1-D array
    imageHistogram = imageHistogram.reshape(1, imageHistogram.shape[0] * imageHistogram.shape[2])
    return imageHistogram


def computeGradientAngle(xEdgeManitude, yEdgeMagnitude):
    angle = 0
    if xEdgeManitude == 0:
        # If Gx = 0 and Gy > 0, tan-1(undefined) = 90
        if yEdgeMagnitude > 0:
            angle = 90

        # If Gx and Gy are 0, angle is 0
        else:
            angle = 0
    else:
        angle = math.degrees(math.atan(yEdgeMagnitude / xEdgeManitude))
        # Convert angle with respect to the positive x-axis to the right
        if angle < 0:
            angle += 360
    return angle


def getGradientAngle(Gx, Gy, height, width):
    gradientAngle = np.empty(shape=(height, width))

    for i in range(height):
        for j in range(width):
            gradientAngle[i][j] = computeGradientAngle(Gx[i][j], Gy[i][j])

    return gradientAngle


def normalizeAndRoundOff(array):
    array = ((array - np.min(array)) / np.ptp(array)) * 255
    return np.round(array)


def getEgdeMagnitude(Gx, Gy):
    edgeMagnitude = np.sqrt((np.square(Gx)) + (np.square(Gy)))
    return edgeMagnitude


def applyPrewittYMask(imageData, row, column):
    prewittY = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [-1, -1, -1]])
    sum = 0
    # Apply Prewitt Mask to corresponsding image matrix centered at [row,column]
    for i in range(row - 1, row + 2):
        for j in range(column - 1, column + 2):
            sum += prewittY[i - row + 1][j - column + 1] * imageData[i][j]

    return sum


def getYGradient(imgArr, height, width):
    imageData = np.zeros(shape=(height, width))
    # Leave the Border pixels as zero as the Prewitt Mask goes out of the border.
    for i in range(1, height - 1):
        for j in range(1, imgArr[i].size - 1):
            imageData[i][j] = applyPrewittYMask(imgArr, i, j)

    return imageData


def applyPrewittXMask(imageData, row, column):
    prewittX = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
    sum = 0

    # Apply Prewitt Mask to corresponsding image matrix centered at [row,column]
    for i in range(row - 1, row + 2):
        for j in range(column - 1, column + 2):
            sum += prewittX[i - row + 1][j - column + 1] * imageData[i][j]

    return sum


def getXGradient(imgArr, height, width):
    imageData = np.zeros(shape=(height, width))
    # Leave the Border pixels as zero as the Prewitt Mask goes out of the border.
    for i in range(1, height - 1):
        for j in range(1, imgArr[i].size - 1):
            imageData[i][j] = applyPrewittXMask(imgArr, i, j)

    return imageData


def getHOGFeatureOfAllImages(image_list):
    X_train = []
    for img in image_list:
        height = img.shape[0]
        width = img.shape[1]

        Gx = getXGradient(img, height, width)
        Gy = getYGradient(img, height, width)

        # Normalize the edge magnitude to be in range [0,255] and round off to Integers
        edgeMagnitude = normalizeAndRoundOff(getEgdeMagnitude(Gx, Gy))

        gradientAngle = getGradientAngle(Gx, Gy, height, width)

        hogFeatureArray = getHOGFeatureOfImage(height, width, edgeMagnitude, gradientAngle)

        X_train.append(np.array(hogFeatureArray))

    X_train = np.array(X_train)
    # Reshape to 2D array
    return X_train.reshape(X_train.shape[0], X_train.shape[2])


def convertColorImageToGrayScaleImage(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    grayImage = R * 299. / 1000 + G * 587. / 1000 + B * 114. / 1000
    return np.round(np.array(grayImage))


def getImage(filename):
    colorImage = cv2.imread(filename)
    image = convertColorImageToGrayScaleImage(colorImage)
    return image


def getTrainingImages():
    image_array = []
    result_array = []

    # Get all Positive Images
    for filename in glob.glob('Human/Train_Positive/*.bmp'):
        image_array.append(getImage(filename))
        result_array.append(1.0)

    # Get all Negative Images
    for filename in glob.glob('Human/Train_Negative/crop001278a.bmp'):
        image_array.append(getImage(filename))
        result_array.append(0.0)

    return image_array, result_array


def humanDetector():
    # Get Train Images and their HOG Features
    image_array, result_array = getTrainingImages()
    X_train = getHOGFeatureOfAllImages(image_array)

    # Get Test Images and their HOG Features
    test_image_list = getTestImages()
    X_test = getHOGFeatureOfAllImages(test_image_list)

    N = X_train.shape[1]

    # Train And Predict a MLP with hidden layer size 250,500 and 1000
    trainAndPredictMLP(N, 500, X_train, result_array, X_test)
    trainAndPredictMLP(N, 500, X_train, result_array, X_test)
    trainAndPredictMLP(N, 500, X_train, result_array, X_test)
    trainAndPredictMLP(N, 500, X_train, result_array, X_test)
    trainAndPredictMLP(N, 500, X_train, result_array, X_test)
    trainAndPredictMLP(N, 500, X_train, result_array, X_test)
    trainAndPredictMLP(N, 500, X_train, result_array, X_test)
    trainAndPredictMLP(N, 500, X_train, result_array, X_test)
    trainAndPredictMLP(N, 500, X_train, result_array, X_test)
    trainAndPredictMLP(N, 500, X_train, result_array, X_test)


humanDetector()
