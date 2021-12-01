import cv2
import numpy as np
import time


def opencv_convolve(img, q, w):
    temp = cv2.blur(img, (w, w))
    temp = temp[q:height-q, q:width-q]
    return temp


def loop_convolve(image, q, w):
    new_image3 = np.zeros(image.shape, image.dtype)

    for y in range(q, height - q):
        for x in range(q, width - q):
            temp = [0, 0, 0]
            for k in range(-q, q + 1):
                for l in range(-q, q + 1):
                    temp = temp + image[y + k, x + l]

            temp = temp / (w ** 2)
            new_image3[y, x] = temp

    new_image3 = new_image3[q:height-q, q:width-q]
    return new_image3


def matrix_convolve(image, q, w, height, width):
    temp_image = np.zeros((height - 2 * q, width - 2 * q, 3), image.dtype)

    for i in range(0, w):
        for j in range(0, w):
            temp1 = image[0 + i:height - (w - i - 1), 0 + j: width - (w - j - 1)]
            temp1 = (temp1 / (w ** 2))
            temp_image = temp_image + temp1

    temp_image = temp_image
    return temp_image


image = cv2.imread('resources/Pink.jpg')
image = image.astype('float64')

height = image.shape[0]
width = image.shape[1]

q = 1
w = 2 * q + 1

start = time.time()
cvConvolve = opencv_convolve(image, q, w)
print(time.time() - start)
print("opencv_convolve finished")

start = time.time()
loopConvolve = loop_convolve(image, q, w)
print(time.time() - start)
print("loop_convolve finished")

start = time.time()
matrixConvolve = (matrix_convolve(image, q, w, height, width))
print(time.time() - start)
print("matrix convolve finish")

print(cvConvolve.shape)
print(loopConvolve.shape)
print(matrixConvolve.shape)
#


for i in range(10, 30, 3):
    print(cvConvolve[i, i])
    print(loopConvolve[i, i])
    print(matrixConvolve[i, i])
    print('***************************')
# print(new)


cv2.imwrite('Result/res7.jpg', cvConvolve)
cv2.imwrite('Result/res8.jpg', loopConvolve)
cv2.imwrite('Result/res9.jpg', matrixConvolve)
print('finish')

cvConvolve = cvConvolve.astype('uint8')
loopConvolve = loopConvolve.astype('uint8')
matrixConvolve = matrixConvolve.astype('uint8')

for i in range(10, 30, 3):
    print(cvConvolve[i, i])
    print(loopConvolve[i, i])
    print(matrixConvolve[i, i])
    print('***************************')

cv2.imshow('cv', cvConvolve)
cv2.imshow('matrix', matrixConvolve)
cv2.imshow('loop', loopConvolve)
cv2.waitKey(0)
