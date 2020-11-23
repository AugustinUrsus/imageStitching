import sys
import cv2

## To execute, do python proj.py <input1> <input2>.
## Will write to out.jpg

## TODO: descriptor
## TODO: matching
## TODO: stitching

## Basic entry point with I/O operation.

if __name__ == '__main__':
    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2])
    result = None ## TODO: complete code.
    cv2.imwrite("out.jpg", result)
