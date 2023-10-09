import collections

import cv2
import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import *


def abc(source1, target1, mask1, y, x):
    height = source1.shape[0]
    width = source1.shape[1]
    c = target1[y:y + height, x:x + width]

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    lap = cv2.filter2D(np.copy(source1), ddepth=cv2.CV_64F, kernel=kernel)
    lap2 = cv2.filter2D(np.copy(c), ddepth=cv2.CV_64F, kernel=kernel)
    b = np.zeros_like(c)
    for i in range(0, height):
        for j in range(0, width):
            if mask1[i, j] == 0:
                b[i, j] = c[i, j]
            else:
                b[i, j] = lap[i, j]
                # if np.abs(lap[i, j]) > np.abs(lap2[i, j]):
                #     b[i, j] = lap[i, j]
                # else:
                #     b[i, j] = lap2[i, j]

    mask2 = np.copy(mask1)
    mask2 = mask2.flatten('C')
    b = b.flatten('C')
    diag0 = []
    diag1 = []
    diag1e = []
    diagi = []
    diagie = []

    for i in range(0, height * width):
        if mask2[i] == 0:
            diag0.append(1)
            diag1.append(0)
            diag1e.append(0)
            diagi.append(0)
            diagie.append(0)
        else:
            diag0.append(-4)
            diag1.append(1)
            diag1e.append(1)
            diagi.append(1)
            diagie.append(1)

    a_list = collections.deque(diag1)
    a_list.rotate(1)
    diag1 = list(a_list)

    a_list = collections.deque(diagi)
    a_list.rotate(width)
    diagi = list(a_list)

    a_list = collections.deque(diag1e)
    a_list.rotate(-1)
    diag1e = list(a_list)

    a_list = collections.deque(diagie)
    a_list.rotate(-width)
    diagie = list(a_list)

    data = np.array([diag0, diag1, diagi, diag1e, diagie])
    print(data.shape)
    offsets = np.array([0, 1, width, -1, -width])
    A = dia_matrix((data, offsets), shape=(height * width, height * width))

    xp = spsolve(A.tocsr(True), b)

    # print(xp.__class__)
    tt = np.reshape(xp, (height, width))
    target1[y:y + height, x:x + width] = tt
    return target1, b
    pass


source = cv2.imread("Resources/olaf-original.jpg").astype("float64")
target = cv2.imread("Resources/beach.jpg").astype("float64")
mask = cv2.imread("Resources/olaf-mask.png").astype("float64")

cv2.imwrite("Result/res05.jpg", source)
cv2.imwrite("Result/res06.jpg", target)

print(source.shape)
print(target.shape)
print(mask.shape)

mask = mask / 255

zxc0, b0 = abc(source[:, :, 0], target[:, :, 0], mask[:, :, 0], y=370, x=660)
zxc1, b1 = abc(source[:, :, 1], target[:, :, 1], mask[:, :, 1], y=370, x=660)
zxc2, b2 = abc(source[:, :, 2], target[:, :, 2], mask[:, :, 2], y=370, x=660)

res = np.zeros((zxc0.shape[0], zxc0.shape[1], 3), zxc0.dtype)
res[:, :, 0] = zxc0
res[:, :, 1] = zxc1
res[:, :, 2] = zxc2
cv2.imwrite("Result/res07.jpg", res)

# # ***************************************************************************************************
# #                                   EXTRA RESULT
# #  EXTRA - RESULT - 1
#
# source = cv2.imread("Resources/Q2/pepper.jpg").astype("float64")
# target = cv2.imread("Resources/Q2/snow.jpg").astype("float64")
# mask = cv2.imread("Resources/Q2/pepper-mask.png").astype("float64")
#
#
# cv2.imwrite("Result/res05a.jpg" , source)
# cv2.imwrite("Result/res06a.jpg" , target)
#
# print(source.shape)
# print(target.shape)
# print(mask.shape)
#
# mask = mask / 255
#
# zxc0, b0 = abc(source[:, :, 0], target[:, :, 0], mask[:, :, 0], y=30, x=40)
# zxc1, b1 = abc(source[:, :, 1], target[:, :, 1], mask[:, :, 1], y=30, x=40)
# zxc2, b2 = abc(source[:, :, 2], target[:, :, 2], mask[:, :, 2], y=30, x=40)
#
# res = np.zeros((zxc0.shape[0], zxc0.shape[1], 3), zxc0.dtype)
# res[:, :, 0] = zxc0
# res[:, :, 1] = zxc1
# res[:, :, 2] = zxc2
# cv2.imwrite("Result/res07a.jpg", res)
#
#
# # ***************************************************************************************************
# #  EXTRA - RESULT - 2
#
# source = cv2.imread("Resources/Q2/boat.jpg").astype("float64")
# target = cv2.imread("Resources/Q2/wave.jpg").astype("float64")
# mask = cv2.imread("Resources/Q2/boat-mask.png").astype("float64")
#
#
# cv2.imwrite("Result/res05b.jpg" , source)
# cv2.imwrite("Result/res06b.jpg" , target)
#
# print(source.shape)
# print(target.shape)
# print(mask.shape)
#
# mask = mask / 255
#
# zxc0, b0 = abc(source[:, :, 0], target[:, :, 0], mask[:, :, 0], y=660, x=1150)
# zxc1, b1 = abc(source[:, :, 1], target[:, :, 1], mask[:, :, 1], y=660, x=1150)
# zxc2, b2 = abc(source[:, :, 2], target[:, :, 2], mask[:, :, 2], y=660, x=1150)
#
# res = np.zeros((zxc0.shape[0], zxc0.shape[1], 3), zxc0.dtype)
# res[:, :, 0] = zxc0
# res[:, :, 1] = zxc1
# res[:, :, 2] = zxc2
# cv2.imwrite("Result/res07b.jpg", res)

