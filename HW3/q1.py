import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_hough_space(can, image, num_of_theta, max_r, y_max, x_max, thetas):
    hough = np.zeros((2 * max_r, num_of_theta), 'float64')
    for i in range(0, y_max):
        for j in range(0, x_max):
            if can[i, j] != 255:
                continue
            else:
                for k in range(0, num_of_theta):
                    theta = thetas[k]
                    x = j - int(image.shape[1] / 2)
                    y = int(image.shape[0] / 2) - i
                    r = x * np.cos(theta) + y * np.sin(theta)
                    r = r + max_r
                    hough[int(r), k] += 1

    print(20 * "*")
    print(np.max(hough))
    print(np.min(hough))
    return hough
    pass


def line_finder(hough2, num_of_theta, max_r):
    num_of_lines = 27
    neighborx = 15
    neighborxy = 3
    thrsh = 80
    list_of_points = []
    maxVal = 500000

    while len(list_of_points) < num_of_lines and maxVal > thrsh:
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hough2)
        print(maxVal)
        if maxVal < thrsh:
            break
        x = maxLoc[1]
        y = maxLoc[0]
        list_of_points.append(maxLoc)
        for j in range(x - neighborx, x + neighborx + 1):
            for k in range(y - neighborxy, y + neighborxy + 1):
                if 0 <= k < num_of_theta and 0 <= j < 2 * max_r:
                    hough2[j, k] = -1
    return list_of_points, hough2
    pass


def line_grouping(image, list_of_points, max_r, thetas, number):
    line_g1 = []
    line_g2 = []

    lp11 = []
    lp12 = []
    lp21 = []
    lp22 = []

    for k in range(len(list_of_points)):
        p = list_of_points[k]
        i = p[1]
        j = p[0]
        rho = i - max_r
        theta = thetas[j]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = (a * rho)
        y0 = (b * rho)
        if np.tan(theta) > 0:
            thetap = j - 90
        else:
            thetap = 90 + j
        m = np.tan(np.deg2rad(thetap))

        y1 = y0 + 1000
        y2 = y0 - 1000

        if m != 0:
            x1 = ((y1 - y0) / m) + x0
            x2 = ((y2 - y0) / m) + x0
            while -10000 > x1 or x1 > 10000 or -10000 > x2 or x2 > 10000:
                y1 = y1 - 10
                x1 = ((y1 - y0) / m) + x0
                y2 = y2 + 10
                x2 = ((y2 - y0) / m) + x0
        else:
            y1 = y0
            y2 = y0
            x1 = 2000
            x2 = -2000

        x1 = x1 + int(image.shape[1] / 2)
        x2 = x2 + int(image.shape[1] / 2)
        y1 = int(image.shape[0] / 2) - y1
        y2 = int(image.shape[0] / 2) - y2

        if thetap < 90:
            line_g1.append((x0, y0))
            lp11.append((x1, y1))
            lp12.append((x2, y2))
            color = (0, 0, 255)
        else:
            line_g2.append((x0, y0))
            lp21.append((x1, y1))
            lp22.append((x2, y2))
            color = (0, 255, 0)

        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    cv2.imwrite("Result/res0{}-lines.jpg".format(number), image)
    return line_g1, line_g2, lp11, lp12, lp21, lp22
    pass


def final_step(lp11, lp12, lp21, lp22, img, number):
    rp11 = []
    rp12 = []
    rp21 = []
    rp22 = []

    for i in range(0, len(lp11)):
        for j in range(0, len(lp21)):
            px, py = findintersect(lp11[i], lp12[i], lp21[j], lp22[j])
            if is_good(img, px, py):
                (x1, y1) = lp11[i]
                (x2, y2) = lp12[i]
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                (x3, y3) = lp21[j]
                (x4, y4) = lp22[j]
                cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0, 255, 0), 2)
                rp11.append((x1, y1))
                rp12.append((x2, y2))
                rp21.append((x3, y3))
                rp22.append((x4, y4))

    cv2.imwrite("Result/res0{}-chess.jpg".format(number), img)
    number += 2

    for i in range(0, len(rp11)):
        for j in range(0, len(rp21)):
            px, py = findintersect(rp11[i], rp12[i], rp21[j], rp22[j])
            cv2.circle(img, (int(px), int(py)), 4, (255, 0, 255), 4)

    if number == 9:
        cv2.imwrite("Result/res0{}-corners.jpg".format(number), img)
    else:
        cv2.imwrite("Result/res{}-corners.jpg".format(number), img)


def findintersect(a, b, c, d):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    x3 = c[0]
    y3 = c[1]
    x4 = d[0]
    y4 = d[1]

    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    return px, py


def is_good(img, px, py):
    t = 7
    r = 2
    pa = img[int(py) + t, int(px) + r]
    pb = img[int(py) + r, int(px) + t]
    pc = img[int(py) - t, int(px) - r]
    pd = img[int(py) - r, int(px) - t]

    threshold = 15
    if np.abs(np.mean(pa) - np.mean(pc)) < threshold and np.abs(np.mean(pb) - np.mean(pd)) < threshold:
        if (np.mean(pa) + np.mean(pc)) / 2 < 70 and (np.mean(pb) + np.mean(pd)) / 2 > 110:
            return True
        elif (np.mean(pa) + np.mean(pc)) / 2 > 110 and (np.mean(pb) + np.mean(pd)) / 2 < 70:
            return True
        else:
            return False
    else:
        if (np.mean(pb) + np.mean(pc) + np.mean(pd)) / 2 > 110 and (np.mean(pa)) < 70 and similarity(pb, pc,
                                                                                                     pd) < threshold:
            return True
        elif (np.mean(pd) + np.mean(pb) + np.mean(pa)) / 2 > 110 and (np.mean(pc)) < 70 and similarity(pd, pb,
                                                                                                       pa) < threshold:
            return True
        else:
            return False


def similarity(e, r, t):
    k = np.abs(np.mean(e) - np.mean(r)) + np.abs(np.mean(e) - np.mean(t)) + np.abs(np.mean(t) - np.mean(r))
    k = k / 3
    return k


def hough_transform(address, number):
    img = cv2.imread(address)
    image = np.copy(img)
    can = cv2.Canny(image,400 , 400)
    can[can > 0] = 255
    cv2.imwrite("Result/res0{}.jpg".format(number), can)

    img_shape = image.shape
    y_max = img_shape[0]
    x_max = img_shape[1]
    t = 2
    max_r = int((math.hypot(x_max, y_max)) / t)
    num_of_theta = 180
    a = np.arange(0, 180, 1)
    thetas = np.deg2rad(a)

    hough = get_hough_space(can, image, num_of_theta, max_r, y_max, x_max, thetas)
    number += 2

    plt.imshow(hough)
    plt.savefig("Result/res0{}-hough-space.jpg".format(number), dpi=2000)
    plt.show()
    number += 2

    hough2 = np.copy(hough)
    list_of_points, hough2 = line_finder(hough2, num_of_theta, max_r)

    line_g1, line_g2, lp11, lp12, lp21, lp22 = line_grouping(image, list_of_points, max_r, thetas, number)
    number += 2
    final_step(lp11, lp12, lp21, lp22, img, number)


hough_transform("Resources/im01.jpg", 1)
print(40 * "-")
hough_transform("Resources/im02.jpg", 2)
