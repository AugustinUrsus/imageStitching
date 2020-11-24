import math
import sys
import cv2
import numpy as np


# To execute, do python proj.py <input1> <input2>.
# Will write to out.jpg

# Pipeline: find interest points, then create SIFT descriptor, then matching them, then find stitching line.


# TODO: interest points: change to RGB detector.

def gaussDeriv2D(sigma):
    """ Generate 2d gaussian filter given sigma (sigma > 0) of size 3*sigma
        @Output: 2d gaussian filter as 2d array
    """
    filter_size = 2 * math.ceil(3 * sigma) + 1
    g_x = math.ceil(3 * sigma) - np.ones((filter_size, filter_size)) * np.array(range(filter_size))
    g_y = -g_x.transpose()

    dividend = 2 * math.pi * sigma ** 4

    G_x = -np.multiply(g_x / dividend, np.exp(-(np.square(g_x) + np.square(g_y)) / (2 * sigma * sigma)))
    G_y = -G_x.transpose()

    return [G_x, G_y]

# Can we use just grayscale image to detect the interest points?
def harris(img, si=1, sd=0.7, R=range(16, 23), T=1e6, a=0.05):
    size_i = int(math.ceil(si * 3) * 2 + 1)
    weight = cv2.getGaussianKernel(ksize=size_i, sigma=si)
    weight = weight.dot(weight.T).view('float64')

    gx, gy = gaussDeriv2D(sd)
    dr = cv2.filter2D(img, ddepth=cv2.CV_64F, kernel=gx).view('float64')
    dc = cv2.filter2D(img, ddepth=cv2.CV_64F, kernel=gy).view('float64')

    h = np.stack((dr * dr, dr * dc, dc * dc), axis=-1).view('float64')

    M = cv2.filter2D(h, ddepth=cv2.CV_64F, kernel=weight)

    r_score = (M[:, :, 0] * M[:, :, 2] - M[:, :, 1] * M[:, :, 1]) - a * np.square(M[:, :, 0] + M[:, :, 2])
    return r_score


# TODO: descriptor

def sift(img, x, y):
    pass


# TODO: matching
# Compute shifting vectors
def match_imgs(img1, img2, points1, points2):
    pass


# TODO: stitching
def stitch_imgs(img1, img2, line):
    pass


# Basic entry point with I/O operation.

if __name__ == '__main__':
    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2])
    result = None  # TODO: complete code.
    cv2.imwrite("out.jpg", result)
