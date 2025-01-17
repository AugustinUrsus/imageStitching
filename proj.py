import math
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


# To execute, do python proj.py <input1> <input2>.
# Will write to out.jpg

# Pipeline: find interest points, then create SIFT descriptor, then matching them, then find stitching line.

# Using OpenCV and SIFT to find the feature points and their SIFT descriptor

# try to implement SIFT without calling CV2 function.
def sift_detect_compute(img):
    sift = cv2.xfeatures2d.SIFT_create()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts, fts = sift.detectAndCompute(img_gray, None)
    pts = np.float32([p.pt for p in pts])
    return pts, fts


# Based on that descriptor, we find the best matches
def matching(fts1, fts2, ratio=0.5):
    matches, distances = descriptor_matching(fts1, fts2)  # match 2, then compare the ratio between these two.
    final_matches = []
    for i in range(matches.shape[0]):
        if distances[i, 0] < ratio * distances[i, 1]:  # match is successfully found.
            final_matches.append((i, matches[i, 0]))
    return final_matches


# Euclidean distance matching, return a list of distance and matching list.
def descriptor_matching(des1, des2):
    target = np.zeros((len(des1), 2))
    distance = np.zeros((len(des1), 2))
    for i in range(len(des1)):
        point = des1[i, :]
        tmp2 = des2 - point
        dist = np.sqrt(np.square(tmp2).sum(axis=1))
        target[i] = np.argsort(dist)[:2]
        distance[i] = np.sort(dist)[:2]

    return target, distance


# To compute the homography matrix.
def compute_homography(x_0, x_1):
    x0_bar = x_0.mean(axis=0)
    x1_bar = x_1.mean(axis=0)
    s1 = np.divide(math.sqrt(2), np.sqrt(np.square(x_0 - x0_bar).sum(axis=1)).sum() / x_0.shape[0])
    T1 = np.array([[s1, 0, -s1 * x0_bar[0]], [0, s1, -s1 * x0_bar[1]], [0, 0, 1]]).astype(np.float)
    s2 = np.divide(math.sqrt(2), np.sqrt(np.square(x_1 - x1_bar).sum(axis=1)).sum() / x_1.shape[0])
    T2 = np.array([[s2, 0, -s2 * x1_bar[0]], [0, s2, -s2 * x1_bar[1]], [0, 0, 1]]).astype(np.float)
    x0 = np.concatenate((x_0, np.ones((x_0.shape[0], 1))), axis=1)
    x1 = np.concatenate((x_1, np.ones((x_1.shape[0], 1))), axis=1)
    x0_t = np.matmul(T1, x0.transpose()).transpose()
    x1_t = np.matmul(T2, x1.transpose()).transpose()
    x_3d = x1_t
    x_2d = x0_t
    A = np.zeros((x_3d.shape[0] * 2, 9))
    A[::2, :] = np.concatenate((x_3d, np.zeros((x_3d.shape[0], 3)), -x_3d * x_2d[:, 0, np.newaxis]), axis=1)
    A[1::2, :] = np.concatenate((np.zeros((x_3d.shape[0], 3)), x_3d, -x_3d * x_2d[:, 1, np.newaxis]), axis=1)
    w, v = np.linalg.eig(np.matmul(A.transpose(), A))
    p = v[:, np.argsort(w)[0]]
    p = p / np.sqrt(np.square(p).sum())
    H_bar = p.reshape(3, 3)
    mat = np.linalg.inv(T1).dot(H_bar).dot(T2)
    return mat


# draw the matching line.
def draw_matching(img1, pts1, img2, pts2):
    (h1, w1) = img1.shape[:2]
    (h2, w2) = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype='uint8')
    canvas[0:h1, 0:w1] = img1
    canvas[0:h2, w1:] = img2

    for i in range(pts1.shape[0]):
        pt1 = tuple((int(pts1[i, 0]), int(pts1[i, 1])))
        pt2 = tuple((int(pts2[i, 0] + w1), int(pts2[i, 1])))
        cv2.line(canvas, pt1, pt2, (0, 0, 255), 1)
    return canvas


# try to implement stitch method without calling cv2 function.
def stitch(img1, img2, M):
    # stitched = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1] ,3))
    # for r in range(img2.shape[0]):
    #     for c in range(img2.shape[1]):
    #         try:
    #             r_proj, c_proj, tmp = np.matmul(M, np.array([r, c, 1]).transpose())
    #             # print(r_proj/tmp, c_proj/tmp)
    #             stitched[int(r_proj / tmp), int(c_proj / tmp] = img2[r, c]
    #         except:
    #             pass
    warped = cv2.warpPerspective(img2, M, (img1.shape[1] + img2.shape[1], max(img1.shape[0], img2.shape[0])))
    cv2.imwrite('warped.jpg', warped)
    stitched = warped.copy()
    stitched[0:img1.shape[0], 0:img1.shape[1]] = img1
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if warped[i, j, 0] > 0 or warped[i, j, 1] > 0 or warped[i, j, 2] > 0:
                weight = 1.0 - (j / img1.shape[1])
                stitched[i, j] = img1[i, j] * weight + warped[i, j] * (1 - weight)
    return stitched


def main():
    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2])
    pts1, fts1 = sift_detect_compute(img1)
    pts2, fts2 = sift_detect_compute(img2)
    matching_pairs = matching(fts1, fts2)
    keypoints1 = np.float32([pts1[int(m[0])] for m in matching_pairs])
    keypoints2 = np.float32([pts2[int(m[1])] for m in matching_pairs])
    h_matrix = compute_homography(keypoints1, keypoints2)
    result = stitch(img1, img2, h_matrix)
    plt.subplot(211)
    correspondence_img = draw_matching(img1, keypoints1, img2, keypoints2)
    cv2.imwrite("matched.jpg", correspondence_img)
    plt.imshow(cv2.cvtColor(correspondence_img, cv2.COLOR_BGR2RGB))
    plt.subplot(212)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite("out.jpg", result)


if __name__ == '__main__':
    main()
