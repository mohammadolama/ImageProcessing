import cv2
import numpy as np
from matplotlib import pyplot as plt


def getCDF(a):
    histogram, bins = np.histogram(a, range(257))
    cdf = np.cumsum(histogram)
    cdf = cdf / a.size
    return cdf, histogram


def transform(a, q):
    c = q[a]
    return c


def match(a, b):
    shapeA = a.shape
    shapeB = b.shape
    # a = a.flatten()
    # b = b.flatten()
    CDFA, histogramA = getCDF(a)
    CDFB, histogramB = getCDF(b)
    q = np.zeros(CDFA.shape, CDFA.dtype)
    i = 0
    j = 0
    while i < 256:
        if (CDFB[j] >= CDFA[i]):
            q[i] = j
            i += 1
        else:
            print('*')
            if (j < 255):
                j += 1

    c = transform(a, q)

    # a = np.reshape(a, (-1, shapeA[1]))
    # b = np.reshape(b, (-1, shapeB[1]))
    # c = np.reshape(c, (-1, shapeA[1]))
    print(a.shape)
    print('*********************')
    return c, histogramA, histogramB


image1 = cv2.imread('resources/Dark.jpg')
image2 = cv2.imread('resources/Pink.jpg')



c0, histo01, histo02 = match(image1[:, :, 0], image2[:, :, 0])
c1, histo11, histo12 = match(image1[:, :, 1], image2[:, :, 1])
c2, histo21, histo22 = match(image1[:, :, 2], image2[:, :, 2])

temp = np.zeros(image1.shape, image1.dtype)
temp[:, :, 0] = c0
temp[:, :, 1] = c1
temp[:, :, 2] = c2

cv2.imwrite('Result/res11.jpg', temp)
cv2.imwrite('Result/q6/res11.jpg', temp)



cdf0a, histogram0a = getCDF(image1[:, :, 0])
cdf1a, histogram1a = getCDF(image1[:, :, 1])
cdf2a, histogram2a = getCDF(image1[:, :, 2])
cdf0b, histogram0b = getCDF(image2[:, :, 0])
cdf1b, histogram1b = getCDF(image2[:, :, 1])
cdf2b, histogram2b = getCDF(image2[:, :, 2])






cdf0, histo0 = getCDF(c0)
cdf1, histo1 = getCDF(c1)
cdf2, histo2 = getCDF(c2)


# plt.subplot(3,3,1)
plt.yticks(np.arange(0, 1.1, 0.2))
plt.xticks(np.arange(0, 256, 50))
plt.title('CDF for Blue channels')
plt.plot(cdf0a, color='r')
plt.plot(cdf0b, color='g')
plt.plot(cdf0, color='b')
plt.legend(['Dark.png' , 'Pink.png' , 'final result'])
plt.savefig('Result/q6/CDFforBlueChannels.png')
plt.show()

# plt.subplot(3,3,2)
plt.yticks(np.arange(0, 1.1, 0.2))
plt.xticks(np.arange(0, 256, 50))
plt.title('CDF for Green channels')
plt.plot(cdf1a, color='r')
plt.plot(cdf1b, color='g')
plt.plot(cdf1, color='b')
plt.legend(['Dark.png' , 'Pink.png' , 'final result'])

plt.savefig('Result/q6/CDFforGreenChannels.png')
plt.show()

# plt.subplot(3,3,3)
plt.yticks(np.arange(0, 1.1, 0.2))
plt.xticks(np.arange(0, 256, 50))
plt.title('CDF for Red channels')
plt.plot(cdf2a, color='r')
plt.plot(cdf2b, color='g')
plt.plot(cdf2, color='b')
plt.legend(['Dark.png' , 'Pink.png' , 'final result'])

plt.savefig('Result/q6/CDFforRedChannels.png')
plt.show()


#  *********************************************************************
plt.xticks(np.arange(0, 256, 50))
plt.title('Histogram for Blue channel of Dark.png')
plt.plot(histogram0a, color='b')
plt.savefig('Result/q6/Dark_Blue_histo.png')
plt.show()

plt.xticks(np.arange(0, 256, 50))
plt.title('Histogram for Green channel of Dark.png')
plt.plot(histogram1a, color='g')
plt.savefig('Result/q6/Dark_Green_histo.png')
plt.show()

plt.xticks(np.arange(0, 256, 50))
plt.title('Histogram for Red channel of Dark.png')
plt.plot(histogram2a, color='r')
plt.savefig('Result/q6/Dark_Red_histo.png')
plt.show()
#  ******************************************************************************
plt.xticks(np.arange(0, 256, 50))
plt.title('Histogram for Blue channel of Pink.png')
plt.plot(histogram0b, color='b')
plt.savefig('Result/q6/Pink_Blue_histo.png')
plt.show()

plt.xticks(np.arange(0, 256, 50))
plt.title('Histogram for Green channel of Pink.png')
plt.plot(histogram1b, color='g')
plt.savefig('Result/q6/Pink_Green_histo.png')
plt.show()

plt.xticks(np.arange(0, 256, 50))
plt.title('Histogram for Red channel of Pink.png')
plt.plot(histogram2b, color='r')
plt.savefig('Result/q6/Pink_Red_histo.png')
plt.show()
# ********************************************************************************

plt.xticks(np.arange(0, 256, 50))
plt.title('Histogram for Blue channel of Resust image')
plt.plot(histo0 , 'b')
plt.savefig('Result/q6/Resust_Blue_histo.png')
plt.show()

plt.xticks(np.arange(0, 256, 50))
plt.title('Histogram for Green channel of Resust image')
plt.plot(histo1 , 'g')
plt.savefig('Result/q6/Resust_Green_histo.png')
plt.show()

plt.xticks(np.arange(0, 256, 50))
plt.title('Histogram for Red channel of Resust image')
plt.plot(histo2 , 'r')
plt.savefig('Result/q6/Resust_Red_histo.png')
plt.show()

#  ********************************************************************************

plt.xticks(np.arange(0, 256, 50))
plt.title('Cumulative Histogram for Blue channels')
plt.plot(histogram0a, color='b')
plt.plot(histogram0b, color='g')
plt.plot(histo0 , 'r')
plt.legend(['Dark.png' , 'Pink.png' , 'final result'])
plt.savefig('Result/q6/Cumulative_Blue_histo.png')
plt.show()

plt.xticks(np.arange(0, 256, 50))
plt.title('Cumulative Histogram for Green channels')
plt.plot(histogram1a, color='b')
plt.plot(histogram1b, color='g')
plt.plot(histo1 , 'r')
plt.legend(['Dark.png' , 'Pink.png' , 'final result'])
plt.savefig('Result/q6/Cumulative_Green_histo.png')
plt.show()

plt.xticks(np.arange(0, 256, 50))
plt.title(' Cumulative Histogram for Red channels')
plt.plot(histogram2a, color='b')
plt.plot(histogram2b, color='g')
plt.plot(histo2 , 'r')
plt.legend(['Dark.png' , 'Pink.png' , 'final result'])
plt.savefig('Result/q6/Cumulative_Red_histo.png')
plt.show()
#  *******************************************************************************
plt.xticks(np.arange(0, 256, 50))
plt.title(' Histogram of Dark.png ')
plt.plot(histogram0a , 'b')
plt.plot(histogram1a , 'g')
plt.plot(histogram2a, 'r')
plt.legend(['Blue' , 'Green' , 'Red'])
plt.savefig('Result/q6/Dark_histogram.jpg')
plt.show()


plt.xticks(np.arange(0, 256, 50))
plt.title(' Histogram of Pink.png ')
plt.plot(histogram0b , 'b')
plt.plot(histogram1b , 'g')
plt.plot(histogram2b, 'r')
plt.legend(['Blue' , 'Green' , 'Red'])
plt.savefig('Result/q6/Pink_histogram.jpg')
plt.show()


plt.xticks(np.arange(0, 256, 50))
plt.title(' Histogram of Result image ')
plt.plot(histo0 , 'b')
plt.plot(histo1 , 'g')
plt.plot(histo2 , 'r')
plt.legend(['Blue' , 'Green' , 'Red'])
plt.savefig('Result/q6/Result_histogram.jpg')
plt.savefig('Result/res10.jpg')
plt.show()
