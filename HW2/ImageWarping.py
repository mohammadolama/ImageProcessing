import cv2
import numpy as np


def interpolate(x, y, img):
    a = x - np.floor(x)
    b = y - np.floor(y)

    f = ((1 - a) * (1 - b) * img[int(np.floor(x)), int(np.floor(y))]) + (
            a * (1 - b) * img[int(np.ceil(x)), int(np.floor(y))]) + (
                b * (1 - a) * img[int(np.floor(x)), int(np.ceil(y))]) + (
                a * b * img[int(np.ceil(x)), int(np.ceil(y))])
    return f


def projective(p, image, address):
    tool = np.sqrt(((p[0][0] - p[1][0]) ** 2) + (p[0][1] - p[1][1]) ** 2)
    arz = np.sqrt(((p[1][0] - p[2][0]) ** 2) + ((p[1][1] - p[2][1]) ** 2))

    tool = int(np.ceil(tool))
    arz = int(np.ceil(arz))

    projection = [[1, 1], [1 + tool, 1], [1 + tool, 1 + arz], [1, 1 + arz]]

    print(tool)
    print(arz)
    poi = np.array(p)
    pro = np.array(projection)
    h, status = cv2.findHomography(pro, poi)
    tf_img_warp2 = np.zeros((tool, arz, 3), 'float64')

    for i in range(0, tool):
        for j in range(0, arz):
            temp = [[i], [j], [1]]
            temp2 = np.matmul(h, temp)
            temp2 = temp2 / temp2[2]
            if temp2[0] < image.shape[1]-1 and temp2.shape[0] > 0 and 0 < temp2[1] < image.shape[0]-1:
                tf_img_warp2[i, j] = interpolate(temp2[1], temp2[0], image)

    cv2.imwrite(address, tf_img_warp2)

    print(h)
    print(20*"*")
    pass


img = cv2.imread('resources/books.jpg')
img = img.astype('float64')
p1 = [[667, 208], [382, 103], [314, 295], [600, 403]]
p2 = [[364, 744], [413, 466], [204, 424], [150, 711]]
p3 = [[816, 968], [614, 656], [410, 784], [605, 1105]]
projective(p1, img, "Result/res16.png")
projective(p2, img, "Result/res17.png")
projective(p3, img, "Result/res18.png")
