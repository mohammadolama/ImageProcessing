import numpy as np
import cv2
from matplotlib import pyplot as plt


def myResize(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def create_mask(image, D0, mode):
    h = image.shape[0]
    w = image.shape[1]
    H = np.zeros((h, w), 'float64')
    for u in range(0, image.shape[0]):
        for v in range(0, image.shape[1]):
            T = - ((u - h / 2) ** 2 + (v - w / 2) ** 2) / (2 * (D0 ** 2))
            if mode == 'high':
                H[u, v] = 1 - np.exp(T)
            else:
                H[u, v] = np.exp(T)
            # H[u, v] = -4 * (math.pi ** 2) * ((u - h / 2) ** 2 + (v - w / 2) ** 2)
    return H


def create_layer(far_image, near_image):
    i = 33
    j = 12
    high_mask = create_mask(near_image, i, mode='high')
    low_mask = create_mask(far_image, j, mode='low')

    a0, a1, ffsh0, nfsh0, l0, h0, lr0, hr0 = subdef(far_image[:, :, 0], near_image[:, :, 0], i, j)
    b0, b1, ffsh1, nfsh1, l1, h1, lr1, hr1 = subdef(far_image[:, :, 1], near_image[:, :, 1], i, j)
    c0, c1, ffsh2, nfsh2, l2, h2, lr2, hr2 = subdef(far_image[:, :, 2], near_image[:, :, 2], i, j)

    plott(nfsh0, nfsh1, nfsh2, 'res23-dft-near', 5, 'plasma', 'Fourier transform magnitude of near image')
    plott(ffsh0, ffsh1, ffsh2, 'res24-dft-far', 5, 'plasma', 'Fourier transform magnitude of Far image')
    cv2.imwrite("Result/res25-highpass-{}.jpg".format(i), high_mask * 255)
    cv2.imwrite("Result/res26-lowpass-{}.jpg".format(j), low_mask * 255)
    plott(l0 + 1, l1 + 1, l2 + 1, 'res28-lowpassed', 5, 'plasma', 'Fourier transform magnitude of Far image with mask')
    plott((h0 + 1), (h1 + 1), (h2 + 1), 'res27-highpassed', 5, 'plasma', 'Fourier transform magnitude of Near image with mask')
    plott(a1, b1, c1, 'res29-hybrid', 5, 'plasma', 'Fourier transform magnitude of hybrid image')

    res = merge(a0, b0, c0)
    cv2.imwrite('Result/res30-hybrid-near.jpg', res)
    resized = myResize(res, 5)
    cv2.imwrite('Result/res31-hybrid-far.jpg', resized)
    concate(res, i, j)


def plott(f_mag0, f_mag1, f_mag2, name, s, mode, title):
    fig, axs = plt.subplots(1, 3)
    fig.suptitle(title)
    axs[0].imshow(s * np.log(f_mag0), cmap=mode)
    axs[0].set_title('B channel')
    axs[1].imshow(s * np.log(f_mag1), cmap=mode)
    axs[1].set_title('G channel')
    axs[2].imshow(s * np.log(f_mag2), cmap=mode)
    axs[2].set_title('R channel')
    plt.savefig('Result/{}.jpg'.format(name), dpi=500)


def subdef(far_image, near_image, i, j):
    high_mask = create_mask(near_image, i, mode='high')
    low_mask = create_mask(far_image, j, mode='low')

    far_fft = np.fft.fft2(far_image)
    far_fft_shift = np.fft.fftshift(far_fft)
    l1 = np.multiply(far_fft_shift, low_mask)

    near_fft = np.fft.fft2(near_image)
    near_fft_shift = np.fft.fftshift(near_fft)
    h1 = np.multiply(near_fft_shift, high_mask)

    res = l1 + h1
    f1 = np.fft.ifftshift(res)
    f2 = np.fft.ifft2(f1)
    f3 = np.real(f2)
    f4 = np.abs(res)

    return f3, f4, np.abs(far_fft_shift), np.abs(near_fft_shift), np.abs(l1), np.abs(h1), np.real(
        np.fft.ifft2(np.fft.ifftshift(l1))), np.real(np.fft.ifft2(np.fft.ifftshift(h1)))


def concate(image, i, j):
    print(image.shape)
    print(image.shape[0])
    print(image.shape[1])
    w1 = image.shape[1]
    h1 = image.shape[0]
    width = int((3 / 2) * image.shape[1])
    height = image.shape[0]

    res = np.zeros((height, width, 3), 'float64')
    res[0:h1, 0:w1] = image

    resize1 = myResize(image, 50)
    w2 = resize1.shape[1]
    h2 = resize1.shape[0]
    res[0:h2, w1:w1 + w2] = resize1

    resize2 = myResize(image, 25)
    w3 = resize2.shape[1]
    h3 = resize2.shape[0]
    res[h2:h2 + h3, w1:w1 + w3] = resize2

    resize3 = myResize(image, 12)
    w4 = resize3.shape[1]
    h4 = resize3.shape[0]
    res[h2 + h3:h2 + h3 + h4, w1:w1 + w4] = resize3

    resize4 = myResize(image, 6)
    w5 = resize4.shape[1]
    h5 = resize4.shape[0]
    res[h2 + h3 + h4:h2 + h3 + h4 + h5, w1:w1 + w5] = resize4
    cv2.imwrite('Result/q6/res32-hybrid-concate.png', res)


def merge(a, b, c):
    res = np.zeros((a.shape[0], a.shape[1], 3), 'float64')
    res[:, :, 0] = a
    res[:, :, 1] = b
    res[:, :, 2] = c
    return res


far = cv2.imread("resources/panda.jpg")
far = far.astype('float64')

near = cv2.imread("resources/shrek.jpg")
near = near.astype('float64')

cv2.imwrite("Result/res19-near.jpg", near)
cv2.imwrite("Result/res21-near.jpg", near)
cv2.imwrite("Result/res20-far.jpg", far)
cv2.imwrite("Result/res22-far.jpg", far)
create_layer(far, near)
