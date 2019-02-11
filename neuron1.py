from scipy.signal import convolve2d
import numpy as np
import cv2
from matplotlib import pyplot as plt


def show_differences(kernel):
    convolved = convolve2d(image, kernel)
    fig = plt.figure(figsize=(15, 15))
    plt.subplot(121)
    plt.title('Original image')
    plt.axis('off')
    plt.imshow(image, cmap='gray')

    plt.subplot(122)
    plt.title('Convolved image')
    plt.axis('off')
    plt.imshow(convolved, cmap='gray')
    return convolved


image = cv2.imread('cat.png')
# converting the image to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9
output = show_differences(kernel)

kernel = np.ones((8,8), np.float32)/64
dx = show_differences(kernel)

# вертикальный фильтр
kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
dx = show_differences(kernel)

# горизонтальный фильтр:
kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
dy = show_differences(kernel)
