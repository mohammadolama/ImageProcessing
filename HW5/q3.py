import cv2
import numpy as np


def create_stacks(image):
    Gause = []
    Laplace = []
    sigma = 2
    kernel = (45, 45)
    x1 = np.copy(image)
    x2 = np.copy(image)
    for i in range(0, 5):
        x2 = cv2.GaussianBlur(x1, kernel, sigmaX=sigma, sigmaY=sigma)
        Gause.append(x2)
        x3 = x1 - x2
        Laplace.append(x3)
        sigma = sigma * 2
        x1 = x2
    print(kernel, sigma)
    for i in range(4, -1, -1):
        x1 = x1 + Laplace[i]
    return Gause, Laplace
    pass


def blend(pyg, pyl, peg, pel, mask):
    last = 4
    x1 = pyg[last]
    x2 = peg[last]

    kernel = (45, 45)
    sigma = 32
    mask2 = cv2.GaussianBlur(np.copy(mask), kernel, sigmaX=sigma, sigmaY=sigma)
    x3 = x2 * mask2 + x1 * (1 - mask2)
    for i in range(3, -1, -1):
        sigma = sigma / 2
        x1 = pyl[i]
        x2 = pel[i]
        mask2 = cv2.GaussianBlur(np.copy(mask), kernel, sigmaX=sigma, sigmaY=sigma)
        x4 = x2 * mask2 + x1 * (1 - mask2)

        x3 = x3 + x4
        pass

    return x3
    pass


def zxc(image1, image2, mask, address):
    xp1 = zxc2(image1[:, :, 0], image2[:, :, 0], mask[:, :, 0])
    xp2 = zxc2(image1[:, :, 1], image2[:, :, 1], mask[:, :, 1])
    xp3 = zxc2(image1[:, :, 2], image2[:, :, 2], mask[:, :, 2])
    res = np.zeros((xp1.shape[0], xp1.shape[1], 3), dtype='float64')
    res[:, :, 0] = xp1
    res[:, :, 1] = xp2
    res[:, :, 2] = xp3
    cv2.imwrite("{}".format(address), res)

    pass


def zxc2(py, pe, pem):
    pyg, pyl = create_stacks(py)
    peg, pel = create_stacks(pe)
    xp = blend(pyg, pyl, peg, pel, pem)
    return xp
    pass


py = cv2.imread("Resources/banana.jpg").astype('float64')
pe = cv2.imread("Resources/Kiwi.jpg").astype('float64')
pem = cv2.imread("Resources/mask.jpg").astype('float64')
# pem[pem>0] = 255.0
cv2.imwrite("Result/res08.jpg", py)
cv2.imwrite("Result/res09.jpg", pe)
pem = pem / 255
zxc(py, pe, pem, "Result/res10.jpg")

# # ***************************************************************************************************
# #                                   EXTRA RESULTS
# #  EXTRA - RESULT - 1
#
# py = cv2.imread("Resources/Q3/12.jpg").astype('float64')
# pe = cv2.imread("Resources/Q3/11.jpg").astype('float64')
# pem = np.zeros_like(py)
# pem[: , int(py.shape[1]/2)+10:] = 255.0
# # pem = cv2.imread("Resources/Q3/pemask.jpeg").astype('float64')
# # pem[pem>0] = 255.0
# cv2.imwrite("Result/res08b.jpg" , py)
# cv2.imwrite("Result/res09b.jpg" , pe)
# pem = pem / 255
# zxc(py , pe , pem , "Result/res10b.jpg")
#
# #  *************************************************************
#
# #  EXTRA - RESULT - 2
#
# py = cv2.imread("Resources/Q3/sword.jpg").astype('float64')
# pe = cv2.imread("Resources/Q3/flame.jpg").astype('float64')
# pem = cv2.imread("Resources/Q3/swordmask.jpg").astype('float64')
# # pem[pem>0] = 255.0
# cv2.imwrite("Result/res08d.jpg" , py)
# cv2.imwrite("Result/res09d.jpg" , pe)
# pem = pem / 255
# zxc(py , pe , pem , "Result/res10d.jpg")
#
