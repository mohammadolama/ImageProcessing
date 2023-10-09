import math
import time

import cv2
import numpy as np


def myResize(image1, scale_percent, mode=cv2.INTER_AREA):
    width = int(image1.shape[1] * scale_percent / 100)
    height = int(image1.shape[0] * scale_percent / 100)
    image1 = cv2.resize(image1, (width, height), interpolation=mode)
    return image1


def mean_shift(points, kernel_bandwidth):
    shift_points = points
    max_min_dist = 1
    MIN_DISTANCE = 0.0000001
    iteration_number = 0
    still_shifting = np.full((points.shape[0], points.shape[1]), True)
    height = points.shape[0]
    width = points.shape[1]
    labels1 = np.arange(height * width, dtype='int32').reshape((height, width))

    while max_min_dist > MIN_DISTANCE:
        max_min_dist = 0

        iteration_number += 1
        print(iteration_number)
        shift_this_turn = np.full((points.shape[0], points.shape[1]), False)
        for i in range(0, shift_points.shape[0]):
            for j in range(0, shift_points.shape[1]):
                if not still_shifting[i, j]:
                    continue
                if shift_this_turn[i, j]:
                    continue
                else:
                    p_new = shift_points[i, j]
                    shift_this_turn, p_new, shift_points, max_min_dist, MIN_DISTANCE, still_shifting, labels1 = shift_point(
                        p_new, points, kernel_bandwidth, shift_this_turn, max_min_dist, MIN_DISTANCE, still_shifting, i,
                        j, labels1)

    shift_points = shift_points.astype("uint8")
    print(still_shifting.shape)
    return labels1, shift_points[:, :, 0:3]
    pass


def shift_point(point, points, kernel_size, shift, max_min_dist, MIN_DISTANCE, still_shifting, i, j, labels1):
    m = 0.1
    dist1 = points - point
    dist2 = (np.sqrt(dist1[:, :, 0] ** 2 + dist1[:, :, 1] ** 2 + dist1[:, :, 2] ** 2)) + m * (
        np.sqrt(dist1[:, :, 3] ** 2 + dist1[:, :, 4] ** 2))
    zxc = np.where(dist2 <= kernel_size)
    new_point0 = (points[:, :, 0])[zxc].mean()
    new_point1 = (points[:, :, 1])[zxc].mean()
    new_point2 = (points[:, :, 2])[zxc].mean()
    new_point = [new_point0, new_point1, new_point2]

    dist = distance(new_point, point)
    if dist > max_min_dist:
        max_min_dist = dist
    if dist < MIN_DISTANCE:
        still_shifting[zxc] = False
        still_shifting[i, j] = False
    labels1[zxc] = labels1[i, j]
    (points[i, j])[0:3] = new_point0
    shift[i, j] = True
    (points[:, :, 0:3])[zxc] = new_point
    shift[zxc] = True
    return shift, new_point, points, max_min_dist, MIN_DISTANCE, still_shifting, labels1
    pass


def distance(a, b):
    total = 0.0
    for dimension in range(0, 3):
        total += (a[dimension] - b[dimension]) ** 2
    return math.sqrt(total)


def create_index_matrix(height, width):
    arr = np.arange(height).reshape((1, height))
    arr = np.transpose(arr)
    arr = np.tile(arr, (1, width))
    arr2 = np.arange(width).reshape((1, width))
    arr2 = np.tile(arr2, (height, 1))
    index1 = np.zeros((height, width, 2), 'int32')
    index1[:, :, 0] = arr
    index1[:, :, 1] = arr2
    return index1
    pass


start = time.time()
image = cv2.imread("Resources/park.jpg")
image = myResize(image, 20)
image = image.astype('float64')
index = create_index_matrix(image.shape[0], image.shape[1])
res = np.zeros((image.shape[0], image.shape[1], 5), 'int32')
res[:, :, 0:3] = image
res[:, :, 3:] = index

labels, shifted = mean_shift(np.copy(res), 25)
shifted2 = np.copy(shifted)


shifted1 = myResize(np.copy(shifted), 500, cv2.INTER_CUBIC)
print("done")
cv2.imwrite("Result/res05.jpg", shifted1)

print(time.time() - start)
