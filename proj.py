import sys
import cv2

# To execute, do python proj.py <input1> <input2>.
# Will write to out.jpg

# Pipeline: find interest points, then create SIFT descriptor, then matching them, then find stitching line.


# TODO: descriptor
"""
    returns a list of points which are considered as features.
"""


def sift(img):
    pass


# TODO: matching

def match_imgs(img1, img2):
    pass


# TODO: stitching
def stitch_imgs(img1, img2, line):
    pass


# Basic entry point with I/O operation.

if __name__ == '__main__':
    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2])
    result = None  ## TODO: complete code.
    cv2.imwrite("out.jpg", result)
