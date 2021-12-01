import cv2
import numpy as np


def range_choose(image):
    a = np.logical_and(image[:, :, 0] >= 133, image[:, :, 0] <= 179)
    b = np.logical_or(image[:, :, 1] > 50, image[:, :, 2] > 200)
    c = np.logical_and(a, b)

    d = np.logical_and(image[:, :, 0] >= 0, image[:, :, 0] <= 10)
    e = np.logical_and(image[:, :, 1] > 150, image[:, :, 2] > 150)
    f = np.logical_and(d, e)

    g = np.logical_and(image[:, :, 0] >= 0, image[:, :, 0] <= 10)
    h = np.logical_and(image[:, :, 1] > 80, image[:, :, 1] < 200)
    k = np.logical_and(image[:, :, 2] > 80, image[:, :, 2] < 200)
    i = np.logical_and(h, g, k)

    l = np.logical_and(image[:, :, 1] > 70, image[:, :, 1] < 130)
    m = np.logical_and(image[:, :, 2] > 170, image[:, :, 2] < 210)
    n = np.logical_and(g, l, m)

    o = np.logical_and(image[:, :, 1] > 135, image[:, :, 1] < 215)
    p = np.logical_and(image[:, :, 2] > 100, image[:, :, 2] < 160)
    q = np.logical_and(g, o, p)

    z = np.logical_or(c, f, i)
    t = np.logical_or(z, n, q)
    return t


image = cv2.imread('resources/Flowers.jpg')
height = image.shape[0]
width = image.shape[1]

image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
t = range_choose(image)
temp = image.copy()             # hsv
temp[:, :, 0][t] = 35
image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

new_image = cv2.blur(image, (5, 5))
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)
new_image[t] = temp[t]
new_image_orginal = new_image.copy()       # hsv

new_image[int(7 * height / 10):, :] = image[int(7 * height / 10):, :]
new_image[int(5 * height / 10): int(7 * height / 10), :int(width / 3)] = image[
                                                                         int(5 * height / 10): int(7 * height / 10),
                                                                         : int(width / 3)]
new_image = new_image.astype('uint8')
new_image_orginal = new_image_orginal.astype('uint8')
new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)
image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
new_image_orginal = cv2.cvtColor(new_image_orginal, cv2.COLOR_HSV2BGR)

cv2.imwrite('Result/res6.jpg', new_image_orginal)
cv2.imwrite('Result/res6_enhanced.jpg', new_image)
print('finish')
