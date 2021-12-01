import cv2
import numpy as np

image = cv2.imread('resources/Enhance1.JPG')
image = image.astype('float64')
height = image.shape[0]
width = image.shape[1]

print(width)
print(height)

alpha = 0.05
temp1 = np.multiply(image, alpha)
temp2 = np.add(temp1, 1)
temp3 = np.multiply(255, np.log(temp2))
temp4 = np.log(np.add(1, np.multiply(255, alpha)))
new_image1 = np.divide(temp3, temp4)
cv2.imwrite('Result/q1/res1_JustLogFunction.jpg', new_image1)

new_image1 = new_image1.astype('uint8')
new_image1 = cv2.cvtColor(new_image1, cv2.COLOR_BGR2HSV)

new_image1[:, :, 2] = new_image1[:, :, 2] + 15
new_image1 = cv2.cvtColor(new_image1, cv2.COLOR_HSV2BGR)
cv2.imwrite('Result/res1.jpg', new_image1)
print('finish')
