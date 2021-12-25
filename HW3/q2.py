import os
import random
import time

import imageio
import cv2
import numpy as np
from matplotlib import pyplot as plt


def myResize(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def best_match(tex, over, mask):
    corr = cv2.matchTemplate(tex, over, cv2.TM_CCORR_NORMED, mask=mask)
    a = []
    num = 15
    for i in range(0, num):
        minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(corr)
        a.append(maxLoc1)
        corr[maxLoc1[1], maxLoc1[0]] = -100

    top_left = a.pop(random.randint(0, num - 1))
    best_x = top_left[0]
    best_y = top_left[1]
    return best_x, best_y
    pass


# this function will fill the first row of the result
def first_row_filler(image, patch, bs, os, image_gif, second_save, name, gif_enabled, step_by_step):
    global step

    stepx = bs
    stepy = 0

    if gif_enabled:
        image_gif.append(myResize(cv2.cvtColor(((np.copy(image)).astype('uint8')), cv2.COLOR_BGR2RGB), percentage))

    if step_by_step:
        Addres = second_save + "/pic{}.jpg".format(step)
        cv2.imwrite(Addres, np.copy(image))

    while stepx < image.shape[1]:
        stepx = stepx - os
        bsx = min(bs, image.shape[1] - stepx)
        bsy = bs
        overlap = np.copy(image[stepy:stepy + bsy, stepx:stepx + bsx])
        mask = np.ones(overlap.shape, 'float32')
        mask[:, os:] = 0

        best_x, best_y = best_match(np.copy(patch), overlap, mask)
        row_filler_helper(image, patch, best_y, best_x, stepy, stepx, bsx, bsy, os)
        stepx = stepx + bs
        step += 1
        print(step)

        if step_by_step:
            Addres = second_save + "/pic{}.jpg".format(step)
            cv2.imwrite(Addres, np.copy(image))

        if gif_enabled:
            image_gif.append(myResize(cv2.cvtColor(((np.copy(image)).astype('uint8')), cv2.COLOR_BGR2RGB), percentage))

    first_col_filler(image, patch, bs, os, image_gif, second_save, name, gif_enabled, step_by_step)


# this function is a helping function to first_row_filler
def row_filler_helper(image, patch, by, bx, stepy, stepx, bsx, bsy, os):
    a = image[stepy:stepy + bsy, stepx:stepx + os]
    b = np.copy(patch[by: by + bsy, bx:bx + os])
    path = row_min_path(a, b)
    t = np.copy(patch[by: by + bsy, bx: bx + bsx])
    path2, path2r = path_filler2(path, 'hor', bsx, bsy, os)
    maperr = merger(np.copy(path2), np.copy(path2), np.copy(path2))
    maper = merger(np.copy(path2r), np.copy(path2r), np.copy(path2r))

    al = 5
    bl = 1
    g = np.zeros((maper.shape[0], maper.shape[1] + 2 * al, 3), maper.dtype)
    g[:, 2 * al:] = maperr
    g = cv2.GaussianBlur(g, (al, al), sigmaX=bl, sigmaY=bl)
    g_prime = 1 - g
    maperr2 = g[:, 2 * al:]
    maper2 = g_prime[:, 2 * al:]

    t[:, :os] = np.multiply(t[:, :os], maperr2)
    image[stepy:stepy + bsy, stepx: stepx + os] = np.multiply(image[stepy:stepy + bsy, stepx: stepx + os], maper2)
    image[stepy:stepy + bsy, stepx: stepx + os] = image[stepy:stepy + bsy, stepx: stepx + os] + t[:, :os]
    image[stepy:stepy + bsy, stepx + os: stepx + bsx] = t[:, os:]


# find minimum path in a vertical overlap array. used for two blocks which are horizontally neighbor
# this function will calculate an error array.
def row_min_path(a, b):
    t = ((a[:, :, 0] - b[:, :, 0]) ** 2) + ((a[:, :, 1] - b[:, :, 1]) ** 2) + ((a[:, :, 2] - b[:, :, 2]) ** 2)

    e = np.zeros_like(t)
    for i in range(0, t.shape[0]):
        for j in range(0, t.shape[1]):
            if i == 0:
                e[i, j] = t[i, j]
            else:
                if j == 0:
                    e[i, j] = t[i, j] + np.minimum(e[i - 1, j], e[i - 1, j + 1])

                elif j == t.shape[1] - 1:
                    e[i, j] = t[i, j] + np.minimum(e[i - 1, j], e[i - 1, j - 1])
                else:
                    e[i, j] = t[i, j] + min3(e[i - 1, j], e[i - 1, j - 1], e[i - 1, j + 1])
    return row_trace_back(e)


# find minimum path using error array input
def row_trace_back(e):
    path = np.zeros((e.shape[0], e.shape[1]), 'float32')
    argmini = (np.argmin(e, axis=1))
    l_index = argmini[-1]
    path[-1, l_index] = 1
    for i in range(e.shape[0] - 2, -1, -1):
        if l_index == 0:
            if e[i, l_index] > e[i, l_index + 1]:
                l_index = l_index + 1
            path[i, l_index] = 1
        elif l_index == e.shape[1] - 1:
            if e[i, l_index] > e[i, l_index - 1]:
                l_index = l_index - 1
            path[i, l_index] = 1
        else:
            if min3(e[i, l_index], e[i, l_index + 1], e[i, l_index - 1]) == e[i, l_index + 1]:
                l_index = l_index + 1
            elif min3(e[i, l_index], e[i, l_index + 1], e[i, l_index - 1]) == e[i, l_index - 1]:
                l_index = l_index - 1
            path[i, l_index] = 1

    return path


# this function will fill the first column of the result
def first_col_filler(image, patch, bs, os, image_gif, second_save, name, gif_enabled, step_by_step):
    stepy = bs
    stepx = 0
    global step
    while stepy < image.shape[0]:
        stepy = stepy - os
        bsx = bs
        bsy = min(bs, image.shape[0] - stepy)
        overlap = np.copy(image[stepy:stepy + bsy, stepx:stepx + bsx])
        mask = np.ones(overlap.shape, 'float32')
        mask[os:, :] = 0
        best_x, best_y = best_match(np.copy(patch), overlap, mask)
        col_filler_helper(image, patch, best_y, best_x, stepx, stepy, bsx, bsy, os)
        stepy = stepy + bs
        step += 1
        print(step)

        if step_by_step:
            Addres = second_save + "/pic{}.jpg".format(step)
            cv2.imwrite(Addres, np.copy(image))

        if gif_enabled:
            image_gif.append(myResize(cv2.cvtColor(((np.copy(image)).astype('uint8')), cv2.COLOR_BGR2RGB), percentage))

    rest_filler(image, patch, bs, os, image_gif, second_save, name, gif_enabled, step_by_step)


# this function is a helping function to first_col_filler
def col_filler_helper(image, patch, by, bx, stepx, stepy, bsx, bsy, os):
    a = image[stepy:stepy + os, stepx: stepx + bsx]
    b = patch[by: by + os, bx:bx + bsx]
    path = col_min_path(a, b)
    t = np.copy(patch[by: by + bsy, bx: bx + bsx])

    path2, path2r = path_filler2(path, 'ver', bsx, bsy, os)

    maperr = merger(np.copy(path2), np.copy(path2), np.copy(path2))
    maper = merger(np.copy(path2r), np.copy(path2r), np.copy(path2r))

    al = 5
    bl = 1
    g = np.zeros((maper.shape[0], maper.shape[1] + 2 * al, 3), maper.dtype)
    g[:, 2 * al:] = maperr

    g = cv2.GaussianBlur(g, (al, al), sigmaX=bl, sigmaY=bl)
    g_prime = 1 - g

    maperr2 = g[:, 2 * al:]
    maper2 = g_prime[:, 2 * al:]

    t[:os, :] = np.multiply(t[:os, :], maperr2)
    image[stepy:stepy + os, stepx: stepx + bsx] = np.multiply(image[stepy:stepy + os, stepx: stepx + bsx], maper2)
    image[stepy:stepy + os, stepx: stepx + bsx] = image[stepy:stepy + os, stepx: stepx + bsx] + t[:os, :]
    image[stepy + os:stepy + bsy, stepx: stepx + bsx] = t[os:, :]


# find minimum path in a horizontal overlap array. used for two blocks which are vertically neighbor
# this function will calculate an error array.
def col_min_path(a, b):
    t = ((a[:, :, 0] - b[:, :, 0]) ** 2) + ((a[:, :, 1] - b[:, :, 1]) ** 2) + ((a[:, :, 2] - b[:, :, 2]) ** 2)
    e = np.zeros_like(t)
    for j in range(0, t.shape[1]):
        for i in range(0, t.shape[0]):
            if j == 0:
                e[i, j] = t[i, j]
            else:
                if i == 0:
                    e[i, j] = t[i, j] + np.minimum(e[i, j - 1], e[i + 1, j - 1])
                elif i == t.shape[0] - 1:
                    e[i, j] = t[i, j] + np.minimum(e[i, j - 1], e[i - 1, j - 1])
                else:
                    e[i, j] = t[i, j] + min3(e[i - 1, j - 1], e[i, j - 1], e[i + 1, j - 1])
    return col_trace_back(e)


# find minimum path using error array input
def col_trace_back(e):
    path = np.zeros((e.shape[0], e.shape[1]), 'float32')
    argmini = (np.argmin(e, axis=0))
    l_index = argmini[-1]
    path[l_index, -1] = 1
    for j in range(e.shape[1] - 2, -1, -1):
        if l_index == 0:
            if e[l_index + 1, j] < e[l_index, j]:
                l_index += 1
            path[l_index, j] = 1
        elif l_index == e.shape[0] - 1:
            if e[l_index - 1, j] < e[l_index, j]:
                l_index -= 1
            path[l_index, j] = 1
        else:
            if min3(e[l_index, j], e[l_index + 1, j], e[l_index - 1, j]) == e[l_index + 1, j]:
                l_index += 1
            elif min3(e[l_index, j], e[l_index + 1, j], e[l_index - 1, j]) == e[l_index - 1, j]:
                l_index -= 1
            path[l_index, j] = 1
    return path


# this function will fill rest of the result which has L shape overlap with prevous bloks
def rest_filler(image, patch, bs, os, image_gif, second_save, name, gif_enabled, step_by_step):
    stepy = bs
    global step
    while stepy < image.shape[0]:
        stepx = bs
        stepy = stepy - os
        while stepx < image.shape[1]:
            stepx = stepx - os
            bsx = min(bs, image.shape[1] - stepx)
            bsy = min(bs, image.shape[0] - stepy)
            overlap = np.copy(image[stepy:stepy + bsy, stepx:stepx + bsx])
            mask = np.ones(overlap.shape, 'float32')
            mask[os:, os:] = 0
            best_x, best_y = best_match(np.copy(patch), overlap, mask)
            rest_filler_helper(image, patch, best_y, best_x, stepx, stepy, bsx, bsy, os)
            stepx = stepx + bs
            step += 1
            print(step)

            if step_by_step:
                Addres = second_save + "/pic{}.jpg".format(step)
                cv2.imwrite(Addres, np.copy(image))

            if gif_enabled:
                image_gif.append(
                    myResize(cv2.cvtColor(((np.copy(image)).astype('uint8')), cv2.COLOR_BGR2RGB), percentage))

        stepy = stepy + bs

    saver(image, patch, image_gif, second_save, name, gif_enabled)
    pass


# this function is a helping function to rest_filler
def rest_filler_helper(image, patch, by, bx, stepx, stepy, bsx, bsy, os):
    a1 = image[stepy:stepy + os, stepx: stepx + bsx]
    b1 = patch[by: by + os, bx:bx + bsx]
    path1 = col_min_path(a1, b1)

    a2 = image[stepy:stepy + bsy, stepx:stepx + os]
    b2 = patch[by: by + bsy, bx:bx + os]
    path2 = row_min_path(a2, b2)

    t = np.copy(patch[by: by + bsy, bx: bx + bsx])
    maper, maperr = maper_filler(path1, path2, bsx, bsy, os)

    al = 5
    bl = 1
    g = np.zeros((maper.shape[0] + 2 * al, maper.shape[1] + 2 * al, 3), maper.dtype)
    g[2 * al:, 2 * al:] = maperr
    g = cv2.GaussianBlur(g, (al, al), sigmaX=bl, sigmaY=bl)
    g_prime = 1 - g

    maperr2 = g[2 * al:, 2 * al:]
    maper2 = g_prime[2 * al:, 2 * al:]

    t = np.multiply(t, maperr2)
    image[stepy:stepy + bsy, stepx: stepx + bsx] = np.multiply(image[stepy:stepy + bsy, stepx: stepx + bsx], maper2)
    image[stepy:stepy + bsy, stepx: stepx + bsx] = image[stepy:stepy + bsy, stepx: stepx + bsx] + t

    return t


# this function will create a masks to be applied to picture and found block
def maper_filler(path1, path2, bsx, bsy, os):
    pf1, pf1r = path_filler(path1, 'ver', bsx, bsy, os=os)
    pf2, pf2r = path_filler(path2, 'hor', bsx, bsy, os=os)

    maper = np.zeros((bsy, bsx), path1.dtype)
    maper[os:, os:] = 3
    maper[:os, :] = pf1r
    maper[:, :os] = maper[:, :os] + pf2r

    maper[maper == 0] = -1
    maper[maper == -2] = -1
    maper[maper == 2] = 1
    maper[maper == 1] = 0
    maper[maper == -1] = 1
    maper[maper == 3] = 0

    return merger(np.copy(maper), np.copy(maper), np.copy(maper)), map_reverse(
        merger(np.copy(maper), np.copy(maper), np.copy(maper)))


# this function will fill a path array to be a matrix of -1's and +1's
def path_filler(path, mode, bsx, bsy, os):
    if mode == 'hor':
        for h in range(0, bsy):
            flag = False
            for w in range(os - 1, -1, -1):
                if not flag:
                    if path[h, w] == 1:
                        flag = True
                    path[h, w] = 1
    elif mode == 'ver':
        for w in range(0, bsx):
            flag = False
            for h in range(os - 1, -1, -1):
                if not flag:
                    if path[h, w] == 1:
                        flag = True
                    path[h, w] = 1

    return np.copy(path), reverser(np.copy(path))


def path_filler2(path, mode, bsx, bsy, os):
    if mode == 'hor':
        for h in range(0, bsy):
            flag = False
            for w in range(os - 1, -1, -1):
                if not flag:
                    if path[h, w] == 1:
                        flag = True
                    path[h, w] = 1
    elif mode == 'ver':
        for w in range(0, bsx):
            flag = False
            for h in range(os - 1, -1, -1):
                if not flag:
                    if path[h, w] == 1:
                        flag = True
                    path[h, w] = 1

    return np.copy(path), map_reverse(np.copy(path))


# this function is used for saving the results
def saver(result, patch, gif, second_save, name, gif_enabled):
    print("saving final image")
    result = result.astype('uint8')
    patch = patch.astype('uint8')
    ad = second_save + "/result.jpg"
    cv2.imwrite(ad, result)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Result of {}".format(name))
    axs[0].imshow(patch)
    axs[0].set_title("original texture")
    axs[1].imshow(result)
    axs[1].set_title("Result")
    Ad = 'Result/{}.jpg'.format(name)
    plt.savefig(Ad, dpi=500)

    if gif_enabled:
        print("saving gif...")
        Addres = second_save + '/{}.gif'.format(name)
        imageio.mimsave(Addres, gif, fps=30)
        print("gif saved")
    pass


def merger(a, b, c):
    res = np.zeros((a.shape[0], a.shape[1], 3), a.dtype)
    res[:, :, 0] = a
    res[:, :, 1] = b
    res[:, :, 2] = c
    return res


# convert a binary(0 , 1) array to a -1 , +1 array
def reverser(a):
    a[a == 0] = -1
    return a


# convert a binary(0 , 1) array to another (1 , 0) array
def map_reverse(a):
    a[a > 0] = 2
    a[a == 0] = 1
    a[a == 2] = 0
    return a


# minimum of 3 element
def min3(a, b, c):
    return np.minimum(a, np.minimum(b, c))


def run(load_address, name, second_save_address, gif_enabled, step_by_step):
    # if (gif_enabled or step_by_step):
    if not os.path.exists(second_save_address):
        os.makedirs(second_save_address)

    global step
    step = 0
    texture = cv2.imread(load_address)
    texture = texture.astype('float32')
    height = texture.shape[1]
    width = texture.shape[0]
    batch_size = 185
    overlap_size = 40
    image_gif = []
    res = np.zeros((2500, 2500, 3), 'float32')
    r1 = random.randint(0, width - batch_size)
    r2 = random.randint(0, height - batch_size)
    temp = texture[r1:r1 + batch_size, r2:r2 + batch_size]
    res[0:batch_size, 0:batch_size] = temp
    first_row_filler(res, texture, batch_size, overlap_size, image_gif, second_save_address, name, gif_enabled,
                     step_by_step)
    print('image{} finished'.format(load_address))
    print(30 * "*")


# if you want to create a gif of this procedure, set the "gifenabled" to be "True"
# if you want to save every step of the procedure, set the "stepbystep" to be "True"
step = 0
percentage = 10
gifenabled = False
stepbystep = False
start = time.time()
run("Resources/texture6.jpg", 'res11', 'Result/q2/tex6', gif_enabled=gifenabled, step_by_step=stepbystep)
run("Resources/texture11.jpeg", 'res12', 'Result/q2/tex11', gif_enabled=gifenabled, step_by_step=stepbystep)
run("Resources/texture20.jpg", 'res13', 'Result/q2/tex13', gif_enabled=gifenabled, step_by_step=stepbystep)
run("Resources/texture21.jpg", 'res14', 'Result/q2/tex15', gif_enabled=gifenabled, step_by_step=stepbystep)
print("total time is:", time.time() - start)
