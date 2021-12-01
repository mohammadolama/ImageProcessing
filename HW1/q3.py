import math
import os
import time
import cv2
import numpy as np

def givelimit(image, channel):
    lowerband = 4000
    upperbound = 60000
    percent = 10
    m = channel.mean(axis=1)
    m = m.astype('uint16')
    # m = np.round(m)
    m[m < lowerband] = 0
    m[m > upperbound] = 0
    i0 = 0
    i00=0
    while (i0 < image.shape[0] / percent):
        if (m[i0] == 0):
            i00 =i0
        i0+=1

    j0 = channel.shape[0] - 1
    j00 = channel.shape[0] - 1
    while (image.shape[0] - j0 < image.shape[0] / percent):
        if (m[j0] == 0):
            j00 = j0
        j0-=1


    m = channel.mean(axis=0)
    m = np.round(m)
    m = m.astype('uint16')
    m[m < lowerband] = 0
    m[m > upperbound] = 0

    l0 = 0
    l00 = 0
    while (l0 < image.shape[1] / percent):
        if (m[l0] == 0):
            l00 = l0
        l0 += 1

    k0 = channel.shape[1] - 1
    k00 = channel.shape[1] - 1
    while (image.shape[1] - k0 < image.shape[1] / percent):
        if (m[k0] == 0):
            k00 =k0
        k0-=1

    return i00, j00, l00, k00


def edgeDetector(image):
    kernel = np.array([[-1], [1]])
    temp1 = cv2.filter2D(image.astype(float), -1, kernel) + (2 ** 16)
    return temp1


def myResize(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def VsamerangerDetector(image, ymin, ymax, j):
    if (ymin < 0 and ymax < 0):
        sf = -j
        sl = image.shape[0] + (ymin - j)
    elif (ymin < 0):
        sf = ymax - j
        sl = image.shape[0] + (ymin - j)
    else:

        sf = ymax - j
        sl = image.shape[0] - j
    return sf, sl


def HsamerangerDetector(image, xmin, xmax, i):
    if (xmin < 0 and xmax < 0):
        sf = -i
        sl = image.shape[1] + (xmin - i)
    elif (xmin < 0):
        sf = xmax - i
        sl = image.shape[1] + (xmin - i)
    else:
        sf = xmax - i
        sl = image.shape[1] - i
    return sf, sl


def VbaseRange(image, ymin, ymax):
    if (ymin < 0 and ymax < 0):
        bf = 0
        bl = image.shape[0] + ymin
    elif (ymin < 0):
        bf = ymax
        bl = image.shape[0] + ymin
    else:
        bf = ymax
        bl = image.shape[0]
    return bf, bl


def HbaseRange(image, xmin, xmax):
    if (xmin < 0 and xmax < 0):
        bf = 0
        bl = image.shape[1] + xmin
    elif (xmin < 0):
        bf = xmax
        bl = image.shape[1] + xmin
    else:
        bf = xmax
        bl = image.shape[1]
    return bf, bl


def matcher(xmin, xmax, ymin, ymax, channel, blue):
    leastvalue = math.inf
    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            ysf, ysl = VsamerangerDetector(channel, ymin, ymax, j)
            xsf, xsl = HsamerangerDetector(channel, xmin, xmax, i)
            ybf, ybl = VbaseRange(blue, ymin, ymax)
            xbf, xbl = HbaseRange(blue, xmin, xmax)

            temp1 = channel[ysf: ysl, xsf: xsl]
            temp2 = blue[ybf: ybl, xbf: xbl]
            temp11 = edgeDetector(temp1)
            temp22 = edgeDetector(temp2)
            temp3 = temp22 - temp11
            temp3 = np.abs(temp3)
            t = np.sum(temp3)
            if t < leastvalue:
                leastvalue = t
                jchanel = j
                ichanel = i

    return jchanel, ichanel


#  *********************************************************************************************************************
image_name = 'master-pnp-prok-00400-00458a'
address = 'resources/{}.tif'.format(image_name)
image = cv2.imread(address, cv2.IMREAD_UNCHANGED)



dirName = 'Result/q3/{}'.format(image_name)
if not os.path.exists(dirName):
    os.makedirs(dirName)

start = time.time()

gxmin = -15
gxmax = +15
gymin = -15
gymax = +15
rxmin = -15
rxmax = +15
rymin = -15
rymax = +15
for i in range(4, -1, -1):
    percent = int(np.floor((100 / (2 ** i))))
    image1 = myResize(image, percent)
    x = image1.shape[0] // 3
    height = x
    width = image1.shape[1]
    blue = image1[0:x, :]
    green = image1[x:2 * x, :]
    red = image1[2 * x: 3 * x, :]
    j_for_g_channel, i_for_g_channel = matcher(gxmin, gxmax, gymin, gymax, green, blue)
    j_for_R_channel, i_for_r_channel = matcher(rxmin, rxmax, rymin, rymax, red, blue)

    res = np.zeros((blue.shape[0], blue.shape[1], 3), image1.dtype)

    res[:, :, 0] = blue

    res[np.maximum(0, j_for_g_channel): np.minimum(green.shape[0], green.shape[0] + j_for_g_channel),
    np.maximum(0, i_for_g_channel): np.minimum(green.shape[1], green.shape[1] + i_for_g_channel), 1] = \
        green[np.maximum(0, -j_for_g_channel): np.minimum(green.shape[0], green.shape[0] - j_for_g_channel),
        np.maximum(0, -i_for_g_channel): np.minimum(green.shape[1], green.shape[1] - i_for_g_channel)]

    res[np.maximum(0, j_for_R_channel): np.minimum(red.shape[0], red.shape[0] + j_for_R_channel),
    np.maximum(0, i_for_r_channel): np.minimum(red.shape[1], red.shape[1] + i_for_r_channel), 2] = \
        red[np.maximum(0, -j_for_R_channel): np.minimum(red.shape[0], red.shape[0] - j_for_R_channel),
        np.maximum(0, -i_for_r_channel): np.minimum(red.shape[1], red.shape[1] - i_for_r_channel)]

    res8 = (res / 256).astype('uint8')
    cv2.imwrite('{}/bestLayer{}.jpg'.format(dirName, i), res8)

    blue = res[:, :, 0]
    green = res[:, :, 1]
    red = res[:, :, 2]

    i0, j0, l0, k0 = givelimit(res, blue)
    i1, j1, l1, k1 = givelimit(res, green)
    i2, j2, l2, k2 = givelimit(res, red)

    ii = np.maximum(np.maximum(i0, i1), i2)
    jj = np.minimum(np.minimum(j0, j1), j2)
    ll = np.maximum(np.maximum(l0, l1), l2)
    kk = np.minimum(np.minimum(k0, k1), k2)
    temp2 = res[ii:jj, ll:kk]
    cv2.imwrite('{}/layer{}finialCrop.jpg'.format(dirName , i), temp2 / 256)

    gxmin = 2 * i_for_g_channel - 5
    gxmax = 2 * i_for_g_channel + 5
    gymin = 2 * j_for_g_channel - 5
    gymax = 2 * j_for_g_channel + 5
    rxmin = 2 * i_for_r_channel - 5
    rxmax = 2 * i_for_r_channel + 5
    rymin = 2 * j_for_R_channel - 5
    rymax = 2 * j_for_R_channel + 5
    print('*******************************************************************')
    print('Layer {}'.format(i))
    print('best for R chanel : i ={}  and  j={}'.format(i_for_r_channel, j_for_R_channel))
    print('best for G chanel : i ={}  and  j={}'.format(i_for_g_channel, j_for_g_channel))

print(time.time() - start)

if image_name=='master-pnp-prok-01800-01886a':
    cv2.imwrite('Result/res03-Amir.jpg', temp2 / 256)
elif image_name=='master-pnp-prok-01800-01833a':
    cv2.imwrite('Result/res04-Mosque.jpg', temp2 / 256)
elif image_name=='master-pnp-prok-00400-00458a':
    cv2.imwrite('Result/res05-Train.jpg', temp2 / 256)
