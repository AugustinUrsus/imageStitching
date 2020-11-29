import math
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


# To execute, do python proj.py <input1> <input2>.
# Will write to out.jpg

# Pipeline: find interest points, then create SIFT descriptor, then matching them, then find stitching line.

# Basic entry point with I/O operation.

# Using OpenCV and SIFT to find the feature points and their SIFT descriptor
def sift_detect_compute(img):
    sift = cv2.xfeatures2d.SIFT_create()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts, fts = sift.detectAndCompute(img_gray, None)
    pts = np.float32([p.pt for p in pts])
    return pts, fts


# Based on that descriptor, we find the best matches
def matching(pts1, fts1, pts2, fts2, ratio=0.6, thres=4.0):
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(fts1, fts2, 2)  # match 2, then compare the ratio between these two.
    final_matches = []
    for m in raw_matches:
        if len(m) > 1 and m[0].distance < ratio * m[1].distance:  # match is successfully found.
            final_matches.append((m[0].trainIdx, m[0].queryIdx))
    ptsa = np.float32([pts1[m[1]] for m in final_matches])
    ptsb = np.float32([pts2[m[0]] for m in final_matches])
    mat, tmp = cv2.findHomography(ptsa, ptsb, cv2.RANSAC, thres)
    return mat


def stitch(img1, img2, M):
    stitched = cv2.warpPerspective(img1, M, (img1.shape[1] + img2.shape[1], img2.shape[0]))
    stitched[0:img1.shape[0], 0:img2.shape[1]] = img2
    return stitched


if __name__ == '__main__':
    img1 = cv2.imread("img_1.jpg")
    img2 = cv2.imread("img_2.jpg")
    pts1, fts1 = sift_detect_compute(img1)
    pts2, fts2 = sift_detect_compute(img2)
    h_matrix = matching(pts1, fts1, pts2, fts2)
    print(h_matrix)
    result = stitch(img1, img2, h_matrix)  # TODO: complete code.
    cv2.imwrite("out.jpg", result)
