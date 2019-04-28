import cv2
import numpy as np
from matplotlib import pyplot as plt


def calculate_histogram(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    return hist


def normalize_histogram(hist):
    return (hist - hist.min()) / (hist.max() - hist.min())


def calculate_cdf(hist): # cumulative distribution function
    return hist.cumsum()


def normalize_cdf(cdf, hist):
    return cdf / cdf.max() * hist.max()


def plot_histogram_and_cdf(image):
    hist = calculate_histogram(image)
    cdf = calculate_cdf(hist)
    cdf_n = normalize_cdf(cdf, hist)
    plt.plot(cdf_n, color='c')
    plt.hist(image.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()


def equalize_histogram(hist):
    cdf = calculate_cdf(hist)
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    return np.ma.filled(cdf_m, 0).astype('uint8')


def image_after_hist_equ(img):
    hist = calculate_histogram(img)
    temp = equalize_histogram(hist)
    return temp[img]


img = cv2.imread('wiki.jpg', 0)
plot_histogram_and_cdf(img)
img2 = image_after_hist_equ(img)
plot_histogram_and_cdf(img2)
img3 = cv2.imread('wiki2.png', 0)
plot_histogram_and_cdf(img3)
img4 = image_after_hist_equ(img3)
plot_histogram_and_cdf(img4)
cv2.imshow('image', img)
cv2.imshow('image2', img2)
cv2.imshow('image3', img3)
cv2.imshow('image4', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()