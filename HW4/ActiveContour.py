import math

import cv2
import imageio
import numpy as np
from matplotlib import pyplot as plt


def calculate_external_energy(v, sobelx1, sobely1):
    y1 = v[0]
    x1 = v[1]
    res = (-1) * (sobelx1[y1, x1] ** 2 + sobely1[y1, x1] ** 2)
    return res


def calculate_internal_energy(vi, vi2, alpha, dbar):
    y1, y2 = vi[0], vi2[0]
    x1, x2 = vi[1], vi2[1]
    res = (x1 - x2) ** 2 + (y1 - y2) ** 2 - 0.99 * dbar
    return alpha * res
    pass


def average_distance(Ve):
    res = 0.0
    for i in range(0, len(Ve) - 1):
        v1, v2 = Ve[i], Ve[i + 1]
        y1, y2 = v1[0], v2[0]
        x1, x2 = v1[1], v2[1]
        res += (x1 - x2) ** 2 + (y1 - y2) ** 2

    v0, vn = Ve[0], Ve[-1]
    y1, y2 = v0[0], vn[0]
    x1, x2 = v0[1], vn[1]
    res += ((x1 - x2) ** 2 + (y1 - y2) ** 2)
    res = res / len(Ve)

    return res


def distance_from_the_center_of_tasbih(vi, lamda, centerx, centery):
    y1 = vi[0]
    x1 = vi[1]
    res = (y1 - centery) ** 2 + (x1 - centerx) ** 2
    res = lamda * res
    return res


def distance_from_the_center_of_image(image1, y_mean, x_mean, beta):
    height = image1.shape[0]
    width = image1.shape[1]
    res = (y_mean - (height / 2)) ** 2 + (x_mean - (width / 2)) ** 2
    res = beta * res
    return res
    pass


def distance_from_the_center_of_contour(vi, y_mean, x_mean, omega):
    y1 = vi[0]
    x1 = vi[1]
    res = (y_mean - y1) ** 2 + (x_mean - x1) ** 2
    res = omega * res
    return res
    pass


def mean_x_y(Ve):
    y_val = [t[0] for t in Ve]
    x_val = [t[1] for t in Ve]
    x_mean = np.mean(x_val)
    y_mean = np.mean(y_val)
    return x_mean, y_mean
    pass


def calculate_energy2(vi, vi2, alpha, beta, gamma, lamda, omega, sobelx, sobely, dbar, image, x_mean, y_mean,centerx , centery):
    # centery = 300
    # centerx = 400
    t = gamma * calculate_external_energy(vi, sobelx, sobely) + calculate_internal_energy(vi, vi2, alpha, dbar) \
        + distance_from_the_center_of_tasbih(vi, lamda, centerx, centery) + distance_from_the_center_of_image(
        image, y_mean, x_mean, beta)

    return t


def make_Array(V):
    list = []
    for i in range(0, len(V)):
        v = V[i]
        y = v[0]
        x = v[1]
        temp_list = [(y - 1, x - 1), (y - 1, x), (y - 1, x + 1), (y, x - 1), (y, x), (y, x + 1), (y + 1, x - 1),
                     (y + 1, x), (y + 1, x + 1)]
        list.append(temp_list)

    return list
    pass


def viterbi(V, alpha, beta, gamma, lamda, omega, sobelx, sobely, image , centerx , centery):
    list = make_Array(V)
    start = list[0]
    min_dist = math.inf
    best_starting = None
    best_end = None
    dbar = average_distance(V)
    best_parent = None
    x_mean, y_mean = mean_x_y(V)
    for i in range(0, len(start)):
        min_dist_for_vertex_i = math.inf
        parent_array = np.zeros((9, len(V)), 'int16')
        start_point = start[i]
        route = np.zeros((9, len(V)), 'float64')
        for j in range(1, len(V)):
            if j == 1:
                for k in range(0, len(start)):
                    temp = (list[j])[k]
                    route[k, j] = calculate_energy2(start_point, temp, alpha, beta, gamma, lamda, omega, sobelx, sobely,
                                                    dbar, image,
                                                    x_mean, y_mean,centerx , centery)
                    parent_array[k, j] = i
            else:
                for k in range(0, len(start)):
                    dist2 = math.inf
                    parent = None
                    for l in range(0, len(start)):
                        vi2 = (list[j])[k]
                        vi = (list[j - 1])[l]
                        dist = route[l, j - 1] + calculate_energy2(vi, vi2, alpha, beta, gamma, lamda, omega, sobelx,
                                                                   sobely, dbar,
                                                                   image, x_mean, y_mean,centerx , centery)
                        if dist < dist2:
                            parent = l
                            dist2 = dist
                    parent_array[k, j] = parent
                    route[k, j] = dist2
                    # print(route)

        parent2 = None
        for l in range(0, len(start)):
            vi2 = start_point
            vi = (list[-1])[l]
            dist = route[l, -1] + calculate_energy2(vi, vi2, alpha, beta, gamma, lamda, omega, sobelx, sobely, dbar,
                                                    image, x_mean,
                                                    y_mean,centerx , centery)
            if dist < min_dist_for_vertex_i:
                parent2 = l
                min_dist_for_vertex_i = dist

        if min_dist_for_vertex_i < min_dist:
            min_dist = min_dist_for_vertex_i
            best_parent = parent_array
            best_starting = start_point
            best_end = parent2

    V = make_changes(V, best_parent, best_end, list)
    return V
    pass


def make_changes(V, parent_array, parent, list):
    new_V = []
    new_V.append(list[-1][parent])
    for i in range(len(V) - 1, 0, -1):
        p = parent_array[parent, i]
        new_V.append(list[i - 1][p])
        parent = p
    new_V.reverse()
    return new_V
    pass


def make_changes2(V):
    new_V = []
    for i in range(0, len(V)):
        v1 = V[i]
        y1 = v1[0]
        x1 = v1[1]
        v2 = V[(i + 1) % len(V)]
        y2 = v2[0]
        x2 = v2[1]
        newx = int((x1 + x2) / 2)
        newy = int((y1 + y2) / 2)
        new_V.append(v1)
        new_V.append((newy, newx))
    return new_V
    pass


def find_min(route):
    minLoc = np.argmin(route)
    minVal = route[minLoc]
    return minVal, minLoc
    pass


def main(V, alpha, beta, gamma, sobelx1, sobely1, image1 , centerx , centery):
    betap = 0
    lamda = 0
    omega = 0
    count = 0
    flag = False
    for i in range(0, 630):
        print(i)
        new_V = viterbi(V, alpha, betap, gamma, lamda, omega, sobelx1, sobely1, image1,centerx , centery)
        print("old v is {} new is {}".format(len(V), len(new_V)))
        print(new_V == V)
        if new_V == V:
            count += 1
            betap = beta
            lamda = 1
        print("count is {}.".format(count))
        if count == 3 or count == 6 or count == 8:
            flag = True
            new_V = make_changes2(new_V)
            gamma = 200
            if count == 6:
                alpha = 0.6
                lamda = 2
            count += 1

        x_val = [t[0] for t in V]
        y_val = [t[1] for t in V]

        plt.imshow(np.copy(image1), cmap='gray')
        plt.plot(y_val, x_val, '-*r', lw=1)
        plt.savefig("Result/q5/a{}.jpg".format(i), dpi=150)
        plt.close()
        V = new_V
        print("*********************************************************************************")
    pass


def gif_maker():
    frames = []
    for i in range(0, 630):
        frames.append(imageio.imread("Result/q5/a{}.jpg".format(i)))
        if i==629:
            imageio.imwrite("Result/res11.jpg" , imageio.imread("Result/q5/a{}.jpg".format(i)))
        print(i)
    imageio.mimsave("Result/contour.mp4", frames, fps=30)



center_x = int(input("please enter centerx of tasbih: (for best result in orginal image enter 400)"))
center_y = int(input("please enter centerx of tasbih: (for best result in orginal image enter 300)"))

image = cv2.imread("Resources/tasbih.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image = cv2.GaussianBlur(gray_image, (7, 7), sigmaX=2, sigmaY=2)
sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0)

sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1)

sobelx = np.abs(sobelx)
sobely = np.abs(sobely)

sobely[sobely < 50] = 0
sobely[sobely >= 50] = 255
sobelx[sobelx < 30] = 0
sobelx[sobelx >= 30] = 255
sobelx = cv2.GaussianBlur(sobelx, (55, 55), sigmaX=9, sigmaY=9)
sobely = cv2.GaussianBlur(sobely, (55, 55), sigmaX=9, sigmaY=9)

s = np.linspace(-2 * np.pi, 0, 50)


y = 400 + 390 * np.sin(s)
x = 550 + 530 * np.cos(s)

y = y.astype('int32')
x = x.astype('int32')
V = list(zip(y, x))
print(V)

main(V, 1.2, 3, 60, sobelx, sobely, image , center_x , center_y)

gif_maker()
