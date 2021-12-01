import math

import cv2
import numpy as np
import scipy.ndimage.filters
from matplotlib import pyplot as plt


def partA(k):
    def part_a_convolver():
        a = np.zeros((101, 101), 'float64')
        a[50, 50] = 255
        f = cv2.GaussianBlur(a, (9, 9), sigmaX=2, sigmaY=2)

        alpha = 0.5
        temp1 = np.multiply(f, alpha)
        temp2 = np.add(temp1, 1)
        temp3 = np.multiply(255, np.log(temp2))
        temp4 = np.log(np.add(1, np.multiply(255, alpha)))
        temp5 = np.divide(temp3, temp4)
        return temp5

    image = cv2.imread('resources/flowers.blur.png')
    image = image.astype('float64')
    bluerd_image = cv2.GaussianBlur(image, (9, 9), sigmaX=2, sigmaY=2)
    unsharp_mask = (image - bluerd_image)

    result = image + k * unsharp_mask

    gaussfilter = part_a_convolver()
    gaussfilter = cv2.convertScaleAbs(gaussfilter)
    unsharp_mask = unsharp_mask + np.abs(np.min(unsharp_mask))

    cv2.imwrite('Result/res1.jpg', gaussfilter)
    cv2.imwrite('Result/res2.jpg', bluerd_image)
    cv2.imwrite('Result/res3.jpg', cv2.convertScaleAbs(unsharp_mask))
    cv2.imwrite('Result/res4.jpg', result)


def partB(k):
    def part_b_convolver():
        a = np.zeros((101, 101), 'float64')
        for i in range(0, 3):
            for j in range(0, 3):
                a[49 + i, 49 + j] = 255
        b = scipy.ndimage.filters.gaussian_laplace(a, 1)

        return b

    image = cv2.imread('resources/flowers.blur.png')
    image = image.astype('float64')

    unsharp_mask0 = scipy.ndimage.filters.gaussian_laplace(image[:, :, 0], 1)
    unsharp_mask1 = scipy.ndimage.filters.gaussian_laplace(image[:, :, 1], 1)
    unsharp_mask2 = scipy.ndimage.filters.gaussian_laplace(image[:, :, 1], 1)

    unsharp_mask = np.zeros(image.shape, 'float64')
    unsharp_mask[:, :, 0] = unsharp_mask0
    unsharp_mask[:, :, 1] = unsharp_mask1
    unsharp_mask[:, :, 2] = unsharp_mask2
    result = image
    result = result - (k * unsharp_mask)
    result = cv2.convertScaleAbs(result)

    unsharp_mask = unsharp_mask + np.abs(np.min(unsharp_mask))
    log_filter = part_b_convolver()
    log_filter = log_filter + np.abs(np.min(log_filter))
    cv2.imwrite('Result/res5.jpg', log_filter)
    cv2.imwrite('Result/res6.jpg', unsharp_mask)
    cv2.imwrite('Result/res7.jpg', result)


def partC(k, radius):
    def fourier_c(img, k):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        fourier_image_magnitude = np.abs(fshift)

        mask = create_mask(img, radius, mode='high')
        temp = np.multiply(fshift, mask)
        temp = np.add(k * temp, fshift)

        fourier_result_magnitude = np.abs(temp)

        temp = np.fft.ifftshift(temp)
        temp = np.fft.ifft2(temp)
        temp = np.real(temp)
        return temp, fourier_result_magnitude, fourier_image_magnitude

    # create lowpass or highpass gaussian mask
    def create_mask(image, d0, mode):
        h = image.shape[0]
        w = image.shape[1]
        H = np.zeros((h, w), 'float64')
        for u in range(0, image.shape[0]):
            for v in range(0, image.shape[1]):
                T = - ((u - h / 2) ** 2 + (v - w / 2) ** 2) / (2 * (d0 ** 2))
                if mode == 'high':
                    H[u, v] = 1 - np.exp(T)
                else:
                    H[u, v] = np.exp(T)
        return H

    img = cv2.imread('resources/flowers.blur.png')
    img = img.astype('float64')
    temp0, magnitude0, img_mag0 = fourier_c(img[:, :, 0], k)
    temp1, magnitude1, img_mag1 = fourier_c(img[:, :, 1], k)
    temp2, magnitude2, img_mag2 = fourier_c(img[:, :, 2], k)

    # save original image magnitude

    fig, axs = plt.subplots(1, 3)
    fig.suptitle('magnitude of original image')
    axs[0].imshow(10 * np.log(img_mag0))
    axs[0].set_title('B channel')
    axs[1].imshow(10 * np.log(img_mag1))
    axs[1].set_title('G channel')
    axs[2].imshow(10 * np.log(img_mag2))
    axs[2].set_title('R channel')
    plt.savefig('Result/res8.jpg', dpi=1000)

    # save highpass mask
    mask = create_mask(img, radius, mode='high')
    mask = mask * 255
    cv2.imwrite('Result/res9.jpg', mask)

    # save magnitude of (1+kH).F
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('magnitude of (1+kH).F')
    axs[0].imshow(10 * np.log(magnitude0))
    axs[0].set_title('B channel')
    axs[1].imshow(10 * np.log(magnitude1))
    axs[1].set_title('G channel')
    axs[2].imshow(10 * np.log(magnitude2))
    axs[2].set_title('R channel')
    plt.savefig('Result/res10.jpg', dpi=1000)

    # save final result
    res = np.zeros(img.shape, img.dtype)
    res[:, :, 0] = temp0
    res[:, :, 1] = temp1
    res[:, :, 2] = temp2
    res[res < 0] = 0
    res[res > 255] = 255
    cv2.imwrite('Result/res11.jpg', res)


def part_d(k):
    def fourier_d(image):
        hp_mask = create_mask(image)
        f = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f)
        temp = np.multiply(hp_mask, f_shift)
        temp_magnitude = np.abs(temp)
        temp = np.fft.ifftshift(temp)
        temp = np.fft.ifft2(temp)
        fourier_inverse = np.real(temp)
        fourier_inverse_mag = np.copy(np.real(temp))
        return fourier_inverse, temp_magnitude, fourier_inverse_mag

    # this function create a laplacian mask
    def create_mask(image):
        h = image.shape[0]
        w = image.shape[1]
        H = np.zeros((h, w), img.dtype)
        for u in range(0, img.shape[0]):
            for v in range(0, img.shape[1]):
                H[u, v] = -4 * (math.pi ** 2) * ((u - h / 2) ** 2 + (v - w / 2) ** 2)

        return H

    def normalize(x):
        x += np.abs(np.min(x))
        x /= np.max(x)
        return x

    img = cv2.imread('resources/flowers.blur.png')
    img = img.astype('float64')
    img = img / 255
    fi0, f_mag0, fim0 = fourier_d(img[:, :, 0])
    fi1, f_mag1, fim1 = fourier_d(img[:, :, 1])
    fi2, f_mag2, fim2 = fourier_d(img[:, :, 2])

    #  save figure of 'magnitude of 4.pi^2.(u^2 + v^2)F'
    f_mag0 += 1
    f_mag1 += 1
    f_mag2 += 1
    s = 20
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('magnitude of 4.pi^2.(u^2 + v^2)F')
    axs[0].imshow(s * np.log(f_mag0))
    axs[0].set_title('B channel')
    axs[1].imshow(s * np.log(f_mag1))
    axs[1].set_title('G channel')
    axs[2].imshow(s * np.log(f_mag2))
    axs[2].set_title('R channel')
    plt.savefig('Result/res12.jpg', dpi=500)

    # save figure of 'magnitude of F^-1 {4.pi^2.(u^2 + v^2)F }'
    fim0 = normalize(fim0)
    fim1 = normalize(fim1)
    fim2 = normalize(fim2)
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('magnitude of F^-1 {4.pi^2.(u^2 + v^2)F }')
    axs[0].imshow(fim0 * 255)
    axs[0].set_title('B channel')
    axs[1].imshow(fim1 * 255)
    axs[1].set_title('G channel')
    axs[2].imshow(fim2 * 255)
    axs[2].set_title('R channel')
    plt.savefig('Result/res13.jpg', dpi=1000)

    # save result image
    fi0 = (fi0 / np.max(np.abs(fi0)))
    fi1 = (fi1 / np.max(np.abs(fi1)))
    fi2 = (fi2 / np.max(np.abs(fi2)))
    tempp = np.zeros((fi0.shape[0], fi0.shape[1], 3), 'float64')
    tempp[:, :, 0] = fi0
    tempp[:, :, 1] = fi1
    tempp[:, :, 2] = fi2
    res = img + (k * tempp)
    res[res<0] = 0
    res[res>1]=1
    cv2.imwrite('Result/res14.jpg', res * 255)


# partA(k=2)
# partB(k=3)
# partC(k=7, radius=130)
part_d(k=-2)
