import cv2
import numpy as np


image = cv2.imread('resources/Enhance2.JPG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image = image.astype('float64')
height = image.shape[0]
width = image.shape[1]

print(width)
print(height)


alpha = 0.1
temp1 = np.multiply(image[: , : , 2], alpha)
temp2 = np.add(temp1, 1)
temp3 = np.multiply(255, np.log(temp2))
temp4 = np.log(np.add(1, np.multiply(255, alpha)))
temp5 = np.divide(temp3, temp4)

image[:, : ,2] =temp5

new_image1 = image
new_image1 = new_image1.astype('uint8')
new_image1 = cv2.cvtColor(new_image1, cv2.COLOR_HSV2BGR)
cv2.imwrite('Result/res2.JPG', new_image1)
print('finish')
