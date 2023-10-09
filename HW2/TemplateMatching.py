import cv2
import numpy as np


def myResize(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def meanfilter(image, template):
    mean_filter = np.ones(np.shape(template)) / np.size(template)
    mean_filter = mean_filter.astype('float64')

    image_mean_array2 = cv2.filter2D(image, -1, mean_filter)
    image_mean_array2 = image_mean_array2.astype('float64')
    return image_mean_array2


def t(image, template):
    s = np.size(template)

    sum_filter = np.ones(np.shape(template))

    stan_tem = template - np.mean(template)
    stan_tem_sum = np.sum(stan_tem)
    stan_tem_power_sum = np.sum(np.square(stan_tem))
    image_mean_array = meanfilter(image, template)
    image_sqaure = np.square(image)
    image_sum_array = cv2.filter2D(image, -1, sum_filter)
    image_square_sum_array = cv2.filter2D(image_sqaure, -1, sum_filter)

    im_tem_convolve = cv2.filter2D(image, -1, stan_tem)

    denom0 = np.sqrt(image_square_sum_array - 2 * image_mean_array * image_sum_array + s * np.square(image_mean_array))
    denom = np.sqrt(stan_tem_power_sum) * denom0

    enum = im_tem_convolve - (image_mean_array * stan_tem_sum)
    # enum = im_tem_convolve

    corr = enum / denom

    return corr

    pass


image = cv2.imread('resources/Greek-ship.jpg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = img.astype('float64')
cv2.imwrite("Result/q2/org_gray.png", img)

# load , crop , scale the patch
patch = cv2.imread('resources/patch.png')
temp = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
temp = temp.astype('float64')
s = 50
k = 50
temp2 = temp[s: temp.shape[0] - s, k: temp.shape[1] - k]
temp = myResize(temp2, 63)


# corr matrix
corr = t(img, temp)

cv2.imwrite('Result/q2/ncc.png', corr * 255)

threshold = 0.45
t = 27
ry = np.zeros(np.shape(img))

# image[corr>threshold] = [0,0,255]
for i in range(0, image.shape[1], t + 1):
    te = corr[:, i:i + t]

    if np.max(te) > threshold:
        v = np.max(te)
        ab = (te == v)
    else:
        ab = np.zeros(np.shape(te))

    ry[:, i:i + t] = ab
    # image[:,i] = [0,255,0]

list_of_points = []
ry = ry.astype('uint8')
found = 1
for i in range(0, image.shape[0]):
    for j in range(0, image.shape[1]):
        if (ry[i, j] > 0):
            image[i, j] = [0, 0, 255]
            list_of_points.append((i, j))
            print('{} found'.format(found))
            found += 1
#
print('end for')
w = int(patch.shape[1] / 2)
h = int(patch.shape[0] / 2)
for pair in list_of_points:
    x = pair[1]
    y = pair[0]
    a = (x - w, y - h)
    b = (x + w, y + h)
    cv2.rectangle(image, a, b, (0, 0, 255), 2)

cv2.imwrite('Result/res15.jpg', image)

# cv2.imwrite('Result/q2/modified_patch.jpg', temp)
