import random
import time

import numpy as np
import cv2
import skimage
from skimage.color import rgb2lab, lab2rgb
from skimage.segmentation import mark_boundaries
from skimage.transform import resize

from skimage.io import imread, imsave


def myResize(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height), cv2.INTER_AREA)
    return image


def fix_connectivity(height, width, labels):
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if labels[i + 1, j] == labels[i - 1, j] and labels[i, j] != labels[i + 1, j]:
                labels[i, j] = labels[i + 1, j]
            elif labels[i, j + 1] == labels[i, j - 1] and labels[i, j] != labels[i, j - 1]:
                labels[i, j] = labels[i, j - 1]
            elif labels[i + 1, j + 1] == labels[i - 1, j - 1] and labels[i, j] != labels[i - 1, j - 1]:
                labels[i, j] = labels[i - 1, j - 1]
            elif labels[i - 1, j + 1] == labels[i + 1, j - 1] and labels[i, j] != labels[i - 1, j + 1]:
                labels[i, j] = labels[i - 1, j + 1]

    return labels
    pass


def enforce_connectivity(height, width, labels, area_treshold):
    new_labels = -1 * np.ones_like(labels)

    new_id = 0
    nearest_label = 0

    for i in range(height):
        for j in range(width):
            pix_curr_label = labels[i, j]

            if new_labels[i, j] == -1:

                pix_cluster = [(i, j)]
                visited_pix = set()

                while len(pix_cluster) != 0:
                    pix = pix_cluster.pop(0)
                    visited_pix.add(pix)

                    neighbor_pix = tuple(np.array(pix) + [0, -1])
                    if (0 <= neighbor_pix[0] < height) and (0 <= neighbor_pix[1] < width):
                        if new_labels[neighbor_pix] == -1 and (labels[neighbor_pix] == pix_curr_label):
                            pix_cluster.append(neighbor_pix)
                            new_labels[neighbor_pix] = new_id

                    neighbor_pix = tuple(np.array(pix) + [0, 1])
                    if (0 <= neighbor_pix[0] < height) and (0 <= neighbor_pix[1] < width):
                        if new_labels[neighbor_pix] == -1 and (labels[neighbor_pix] == pix_curr_label):
                            pix_cluster.append(neighbor_pix)
                            new_labels[neighbor_pix] = new_id

                    neighbor_pix = tuple(np.array(pix) + [1, 0])
                    if (0 <= neighbor_pix[0] < height) and (0 <= neighbor_pix[1] < width):
                        if new_labels[neighbor_pix] == -1 and (labels[neighbor_pix] == pix_curr_label):
                            pix_cluster.append(neighbor_pix)
                            new_labels[neighbor_pix] = new_id

                    neighbor_pix = tuple(np.array(pix) + [-1, 0])
                    if (0 <= neighbor_pix[0] < height) and (0 <= neighbor_pix[1] < width):
                        if new_labels[neighbor_pix] == -1 and (labels[neighbor_pix] == pix_curr_label):
                            pix_cluster.append(neighbor_pix)
                            new_labels[neighbor_pix] = new_id

                area = len(visited_pix)
                if area < area_treshold:
                    nearest_label = get_nearest_label(height, width, (i, j), new_labels, nearest_label,
                                                      pix_curr_label)
                    for pix in visited_pix:
                        new_labels[pix] = nearest_label

                else:
                    new_id += 1

    return new_labels


def get_nearest_label(height, width, pix_pos, new_seg_map, nearest_label, pix_curr_label):
    adj_shift = np.array([[-1, 0], [-2, 0], [-3, 0], [-4, 0], [-5, 0], [0, -1], [0, 1], [1, 0], ])
    for shift in adj_shift:
        neighbor_pix = tuple(np.array(pix_pos) + shift)
        if (0 <= neighbor_pix[0] < height) and (0 <= neighbor_pix[1] < width):
            if new_seg_map[neighbor_pix] >= 0:
                nearest_label = new_seg_map[neighbor_pix]
                break
    return nearest_label


def initialize_clusters(image, number_of_clusters):
    height = image.shape[0]
    width = image.shape[1]
    N = height * width
    k = number_of_clusters
    S = int(np.sqrt(N / k))
    print("alpha is {} ".format(20/S))
    clusters = make_clusters(height, width, S, image)
    return clusters, S, k
    pass


def make_clusters(height, width, S, image):
    clusters = []
    h = int(S / 2)
    w = int(S / 2)
    while h < height:
        while w < width:
            fix_initialize(clusters, h, w, height, width, image)
            w += S
        w = int(S / 2)
        h += S
    return clusters


def fix_initialize(clusters, h, w, height, width, image):
    cluster_gradient = calculate_gradient(h, w, height, width, image)
    hgood = h
    wgood = w
    t = 2
    for dh in range(-t, t + 1):
        for dw in range(-t, t + 1):
            hprime = h + dh
            wprime = w + dw
            new_gradient = calculate_gradient(hprime, wprime, height, width, image)
            if new_gradient < cluster_gradient:
                cluster_gradient = new_gradient
                hgood = hprime
                wgood = wprime

    clusters.append((hgood, wgood))


def calculate_gradient(h, w, height, width, image):
    if w + 1 >= width:
        w = width - 2
    if h + 1 >= height:
        h = height - 2
    if w == 0:
        w = 1
    if h == 0:
        h = 1

    I1 = image[h + 1, w]
    I2 = image[h - 1, w]
    I3 = image[h, w + 1]
    I4 = image[h, w - 1]
    gradient = norm2(I2 - I1) + norm2(I3 - I4)
    return gradient


def norm2(I):
    x = (I[0]) ** 2 + (I[1]) ** 2 + (I[2]) ** 2
    return x
    pass


def cluster_assignment(clusters, labels, image, index, height, width, S, m):
    distances = np.full((height, width), np.inf, dtype='float64')

    for i in range(0, len(clusters)):
        cluster = clusters[i]
        hc = cluster[0]
        wc = cluster[1]

        # find distance tor a neighborhood around "cluster"
        zxc = 1
        a = np.maximum(0, hc - zxc * S)
        b = np.minimum(height - 1, hc + zxc * S + 1)
        c = np.maximum(0, wc - zxc * S)
        d = np.minimum(width - 1, wc + zxc * S + 1)
        temp = image[a:b, c:d]
        indexp = np.copy(index[a:b, c:d])

        Dp = distance_calculator(image, temp, indexp, hc, wc, m, S)
        di = np.full((height, width), np.inf, dtype='float64')
        di[a:b, c:d] = Dp
        labels[di < distances] = i
        distances = np.where(di < distances, di, distances)

    return labels


def update_clusters(indexs, labels, clusters):
    k = len(clusters)
    new_clusters = []
    for i in range(0, k):
        t = np.where(labels == i)
        new_point0 = (indexs[:, :, 0])[t].mean()
        new_point1 = (indexs[:, :, 1])[t].mean()
        new_clusters.append((int(np.floor(new_point0)), int(np.floor(new_point1))))
    return new_clusters


def distance_calculator(image, temp, index, hc, wc, m, S):
    cluster = image[hc, wc]
    temp2 = temp - cluster
    dlab = (temp2[:, :, 0]) ** 2 + (temp2[:, :, 1]) ** 2 + (temp2[:, :, 2]) ** 2
    dlab = np.sqrt(dlab)
    index0 = index[:, :, 0] - hc
    index1 = index[:, :, 1] - wc
    dxy = index0 ** 2 + index1 ** 2

    # print(S)
    dxy = np.sqrt(dxy)
    alpha = m / S
    # alpha = 0.8
    return dlab + (alpha * dxy)
    pass
    pass


def create_index_matrix(height, width):
    arr = np.arange(height).reshape((1, height))
    arr = np.transpose(arr)
    arr = np.tile(arr, (1, width))
    arr2 = np.arange(width).reshape((1, width))
    arr2 = np.tile(arr2, (height, 1))
    index = np.zeros((height, width, 2), 'int32')
    index[:, :, 0] = arr
    index[:, :, 1] = arr2
    return index
    pass


def xyz(m, address):
    start = time.time()
    start1 = time.time()
    img = imread("Resources/slic.jpg")
    img = resize(img, (img.shape[0] // 4, img.shape[1] // 4), anti_aliasing=True)
    # image2 = image2.astype('float64')
    #
    image2 = img.astype('float64')

    image2 = rgb2lab(image2)

    image = np.copy(image2)
    index = create_index_matrix(image.shape[0], image.shape[1])
    labels = np.full((image.shape[0], image.shape[1]), -1)

    print("time for loading and converting image : {} ".format(start - time.time()))

    start = time.time()
    clusters, S, k = initialize_clusters(image, number_of_clusters=m)
    print("time for init centers : {} ".format(start - time.time()))

    for i in range(0, 4):
        start = time.time()
        image = np.copy(image2)
        labels = np.full((image.shape[0], image.shape[1]), -1, dtype='int64')
        labels = cluster_assignment(clusters, labels, image, index, image.shape[0], image.shape[1], S, m=20)
        print("time for 1 iteration : {} ".format(start - time.time()))

        start = time.time()
        clusters = update_clusters(index, labels, clusters)
        print("time for updating clusters : {} ".format(start - time.time()))

    for i in range(0, 1):
        start = time.time()
        labels = fix_connectivity(image.shape[0], image.shape[1], labels)
        print("time for fixing_connectivity : {} ".format(start - time.time()))

        start = time.time()
        labels = enforce_connectivity(image.shape[0], image.shape[1], labels, 2 * S)
        print("time for enforcing : {} ".format(start - time.time()))

        labels = labels.astype('uint8')
        labels2 = myResize(labels, 400)
        labels2 = labels2.astype('uint8')
        img1 = imread("Resources/slic.jpg")
        res = skimage.segmentation.mark_boundaries(np.copy(img1), labels2, color=(1, 0, 0))
        imsave(address, res)
        print("total time : {} ".format(start1 - time.time()))


xyz(64, "Result/res06.jpg")
xyz(256, "Result/res07.jpg")
xyz(1024, "Result/res08.jpg")
xyz(2048, "Result/res09.jpg")
