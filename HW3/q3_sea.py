import random

import numpy as np
import cv2


def merger(a, b, c):
    result = np.zeros((a.shape[0], a.shape[1], 3), a.dtype)
    result[:, :, 0] = a
    result[:, :, 1] = b
    result[:, :, 2] = c
    return result


def reverser(a):  # take a binary(0,1) matrix in convert it to a (-1 , 1) matrix
    a[a == 0] = -1
    return a


def reverser2(a):
    b = 1 - a
    return b


def map_reverse(a):  # take a binary matrix and and reverse it (e.g 0's to 1 & 1's to 0)
    a[a > 0] = 2
    a[a == 0] = 1
    a[a == 2] = 0
    return a


def min3(a, b, c):
    return np.minimum(a, np.minimum(b, c))


def best_match(tex, over, mask, xi, xf, yi, yf):
    Area1 = np.copy(tex[:, 0:xi - 1])
    Area2 = np.copy(tex[0:yi - 1, :])
    Area3 = np.copy(tex[:, xf + 51:])
    Area4 = np.copy(tex[yf + 51:, :])
    corr1 = cv2.matchTemplate(Area1, over, cv2.TM_CCORR_NORMED, mask=mask)
    corr2 = cv2.matchTemplate(Area2, over, cv2.TM_CCORR_NORMED, mask=mask)
    corr3 = cv2.matchTemplate(Area3, over, cv2.TM_CCORR_NORMED, mask=mask)
    corr4 = cv2.matchTemplate(Area4, over, cv2.TM_CCORR_NORMED, mask=mask)

    a = []
    b = []
    c = []
    d = []
    num = 1
    for i in range(0, num):
        minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(corr1)
        minVal2, maxVal2, minLoc2, maxLoc2 = cv2.minMaxLoc(corr2)
        minVal3, maxVal3, minLoc3, maxLoc3 = cv2.minMaxLoc(corr3)
        minVal4, maxVal4, minLoc4, maxLoc4 = cv2.minMaxLoc(corr4)
        a.append(maxLoc1)
        b.append(maxLoc2)
        c.append(maxLoc3)
        d.append(maxLoc4)
        corr1[maxLoc1[1], maxLoc1[0]] = -100
        corr2[maxLoc2[1], maxLoc2[0]] = -100
        corr3[maxLoc3[1], maxLoc3[0]] = -100
        corr4[maxLoc4[1], maxLoc4[0]] = -100

    maxVal1 = a.pop(random.randint(0, num - 1))
    maxVal2 = b.pop(random.randint(0, num - 1))
    maxVal3 = c.pop(random.randint(0, num - 1))
    maxVal4 = d.pop(random.randint(0, num - 1))
    e = [maxVal1, maxVal2, maxVal3, maxVal4]

    maxmax = e.pop(random.randint(0, 3))

    if maxmax == maxVal1:
        top_left = maxVal1
        best_x = top_left[0]
        best_y = top_left[1]
        return best_x, best_y
    elif maxmax == maxVal2:
        top_left = maxVal2
        best_x = top_left[0]
        best_y = top_left[1]
        return best_x, best_y
    elif maxmax == maxVal3:
        top_left = maxVal3
        best_x = top_left[0] + xf
        best_y = top_left[1]
        return best_x, best_y
    else:
        top_left = maxVal4
        best_x = top_left[0]
        best_y = top_left[1] + yf
        return best_x, best_y


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


def filler(image, patch, bs, os, name, xi, xf, yi, yf, height, width):
    num_patch = 0
    stepy = yi
    while stepy < yf:
        stepx = xi
        stepy = stepy - os
        while stepx < xf:
            stepx = stepx - os
            bsx = min(bs, xf - stepx)
            bsy = min(bs, yf - stepy)
            overlap = np.copy(image[stepy:stepy + bsy, stepx:stepx + bsx])
            mask = np.ones(overlap.shape, 'float32')
            mask[os:, os:] = 0
            best_x, best_y = best_match(np.copy(patch), overlap, mask, xi, xf, yi, yf)
            filler_helper(image, patch, best_y, best_x, stepx, stepy, bsx, bsy, os)
            stepx = stepx + bs
            num_patch += 1
            print(num_patch)
        stepy = stepy + bs
    return image


def filler_helper(image, patch, by, bx, stepx, stepy, bsx, bsy, os):
    a1 = image[stepy:stepy + os, stepx: stepx + bsx]
    b1 = patch[by: by + os, bx:bx + bsx]
    path1 = col_min_path(a1, b1)
    a2 = image[stepy:stepy + bsy, stepx:stepx + os]
    b2 = patch[by: by + bsy, bx:bx + os]
    path2 = row_min_path(a2, b2)
    t = np.copy(patch[by: by + bsy, bx: bx + bsx])
    maper, maperr = main_maper_filler(path1, path2, bsx, bsy, os)

    al = 5
    bl = 1
    mp = np.zeros((maper.shape[0] + 2 * al, maper.shape[1] + 2 * al, 3), maper.dtype)
    mp[2 * al:, 2 * al:] = maperr
    mp = cv2.GaussianBlur(mp, (al, al), sigmaX=bl, sigmaY=bl)
    mp2 = reverser2(mp)

    maperr2 = mp[2 * al:, 2 * al:]
    maper2 = mp2[2 * al:, 2 * al:]
    t = np.multiply(t, maperr2)
    image[stepy:stepy + bsy, stepx: stepx + bsx] = np.multiply(image[stepy:stepy + bsy, stepx: stepx + bsx], maper2)
    image[stepy:stepy + bsy, stepx: stepx + bsx] = image[stepy:stepy + bsy, stepx: stepx + bsx] + t
    return t


def main_maper_filler(path1, path2, bsx, bsy, os):
    pf1, pf1r = main_path_filler(path1, 'ver', bsx, bsy, os=os)
    pf2, pf2r = main_path_filler(path2, 'hor', bsx, bsy, os=os)
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


def main_path_filler(path, mode, bsx, bsy, os):
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


def last_row_fixer(image, base_image, bst, xi, xf, yi, yf):
    osy = 30
    stepy = yf - osy
    stepx = xi
    osx = 25
    bs = 100
    while stepx < xf + 1:
        stepx = stepx - osx
        bsx = min(bs, xf - stepx)
        bsy = 2 * osy
        overlap = np.copy(image[stepy:stepy + bsy, stepx:stepx + bsx])
        mask = np.ones(overlap.shape, 'float32')
        best_x, best_y = best_match(np.copy(base_image), overlap, mask, xi, xf, yi, yf)
        last_row_fixer_helper(image, base_image, best_y, best_x, stepy, stepx, bsx, bsy, osx, osy)
        stepx = stepx + bs
    return image


def last_row_fixer_helper(image, base_image, by, bx, stepy, stepx, bsx, bsy, osx, osy):
    temp = np.copy(base_image[by: by + bsy, bx:bx + bsx])
    a1 = image[stepy:stepy + osy, stepx: stepx + bsx]
    b1 = temp[: osy, :]
    path1 = col_min_path(a1, b1)

    a3 = image[stepy + osy: stepy + osy + osy, stepx: stepx + bsx]
    b3 = temp[osy:, :]
    path3 = col_min_path(a3, b3)

    a2 = image[stepy:stepy + bsy, stepx:stepx + osx]
    b2 = temp[:, :osx]
    path2 = row_min_path(a2, b2)

    t = np.copy(base_image[by: by + bsy, bx: bx + bsx])
    maper, maperr = maper_filler3(path1, path2, path3, bsx, bsy, osx, osy)

    al = 5
    bl = 0
    mp = np.zeros((maper.shape[0] + 2 * al, maper.shape[1] + al, 3), maper.dtype)
    mp[al:-al, al:] = maperr

    mp = cv2.GaussianBlur(mp, (al, al), sigmaX=bl, sigmaY=bl)
    mp2 = reverser2(mp)
    maperr = mp[al:-al, al:]
    maper = mp2[al:-al, al:]

    t = np.multiply(t, maperr)
    image[stepy:stepy + bsy, stepx: stepx + bsx] = np.multiply(image[stepy:stepy + bsy, stepx: stepx + bsx], maper)

    image[stepy:stepy + bsy, stepx: stepx + bsx] = image[stepy:stepy + bsy, stepx: stepx + bsx] + t
    pass


def maper_filler3(path1, path2, path3, bsx, bsy, osx, osy):
    pf1, pf1r = path_filler_fixer_helper(path1, 'col', bsx, bsy, osx, osy)
    pf2, pf2r = path_filler_fixer_helper(path2, 'row', bsx, bsy, osx, osy)
    pf3, pf3r = path_filler_fixer_helper(path3, 'col', bsx, bsy, osx, osy)

    pf3r[pf3r == 1] = 2
    pf3r[pf3r == -1] = 1
    pf3r[pf3r == 2] = -1

    maper = np.zeros((bsy, bsx), path1.dtype)
    maper[:, :] = 0
    maper[:osy, :] = pf1r
    maper[:, :osx] = maper[:, :osx] + pf2r
    maper[osy:, :] = maper[osy:, :] + pf3r

    maper[maper == 0] = -1
    maper[maper == -2] = -1
    maper[maper == 2] = 1
    maper[maper == 1] = 0
    maper[maper == -1] = 1
    maper[maper == 3] = 0

    return merger(np.copy(maper), np.copy(maper), np.copy(maper)), map_reverse(
        merger(np.copy(maper), np.copy(maper), np.copy(maper)))


def path_filler_fixer_helper(path, mode, bsx, bsy, osx, osy):
    if mode == 'row':
        for h in range(0, bsy):
            flag = False
            for w in range(osx - 1, -1, -1):
                if not flag:
                    if path[h, w] == 1:
                        flag = True
                    path[h, w] = 1
    elif mode == 'col':
        for w in range(0, bsx):
            flag = False
            for h in range(osy - 1, -1, -1):
                if not flag:
                    if path[h, w] == 1:
                        flag = True
                    path[h, w] = 1

    return np.copy(path), reverser(np.copy(path))


def last_col_fixer(image, base_image, bst, xi, xf, yi, yf):
    osx = 30
    stepy = yi
    stepx = xf - osx
    osy = 25
    bs = 100
    while stepy < yf + 1:
        stepy = stepy - osy
        bsx = 2 * osx
        bsy = min(bs, yf - stepy)
        overlap = np.copy(image[stepy:stepy + bsy, stepx:stepx + bsx])
        mask = np.ones(overlap.shape, 'float32')
        best_x, best_y = best_match(np.copy(base_image), overlap, mask, xi, xf, yi, yf)
        last_col_fixer_helper(image, base_image, best_y, best_x, stepy, stepx, bsx, bsy, osx, osy)
        stepy = stepy + bs
    return image


def last_col_fixer_helper(image, base_image, by, bx, stepy, stepx, bsx, bsy, osx, osy):
    temp = np.copy(base_image[by: by + bsy, bx:bx + bsx])
    a1 = image[stepy:stepy + osy, stepx: stepx + bsx]
    b1 = temp[: osy, :]
    path1 = col_min_path(a1, b1)

    a2 = image[stepy:stepy + bsy, stepx:stepx + osx]
    b2 = temp[:, :osx]
    path2 = row_min_path(a2, b2)

    a3 = image[stepy: stepy + bsy, stepx + osx: stepx + osx + osx]
    b3 = temp[:, osx:]
    path3 = row_min_path(a3, b3)

    t = np.copy(base_image[by: by + bsy, bx: bx + bsx])
    maper, maperr = maper_filler4(path1, path2, path3, bsx, bsy, osx, osy)

    al = 5
    bl = 0
    mp = np.zeros((maper.shape[0] + al, maper.shape[1] + 2 * al, 3), maper.dtype)
    mp[al:, al:-al] = maperr

    mp = cv2.GaussianBlur(mp, (al, al), sigmaX=bl, sigmaY=bl)
    mp2 = reverser2(mp)
    maperr = mp[al:, al:-al]
    maper = mp2[al:, al:-al]

    t = np.multiply(t, maperr)
    image[stepy:stepy + bsy, stepx: stepx + bsx] = np.multiply(image[stepy:stepy + bsy, stepx: stepx + bsx], maper)

    image[stepy:stepy + bsy, stepx: stepx + bsx] = image[stepy:stepy + bsy, stepx: stepx + bsx] + t
    pass


def maper_filler4(path1, path2, path3, bsx, bsy, osx, osy):
    pf1, pf1r = path_filler_fixer_helper(path1, 'col', bsx, bsy, osx, osy)
    pf2, pf2r = path_filler_fixer_helper(path2, 'row', bsx, bsy, osx, osy)
    pf3, pf3r = path_filler_fixer_helper(path3, 'row', bsx, bsy, osx, osy)

    pf3r[pf3r == 1] = 2
    pf3r[pf3r == -1] = 1
    pf3r[pf3r == 2] = -1

    maper = np.zeros((bsy, bsx), path1.dtype)
    maper[:, :] = 0
    maper[:osy, :] = pf1r
    maper[:, :osx] = maper[:, :osx] + pf2r
    maper[:, osx:] = maper[:, osx:] + pf3r

    maper[maper == 0] = -1
    maper[maper == -2] = -1
    maper[maper == 2] = 1
    maper[maper == 1] = 0
    maper[maper == -1] = 1
    maper[maper == 3] = 0

    return merger(np.copy(maper), np.copy(maper), np.copy(maper)), map_reverse(
        merger(np.copy(maper), np.copy(maper), np.copy(maper)))


def last_block(image, base_image, bst, xi, xf, yi, yf):
    stepy = yf - 50
    stepx = xf - 50
    osy = 50
    osx = 50
    bs = 100
    bsx = bs
    bsy = bs
    overlap = np.copy(image[stepy:stepy + bsy, stepx:stepx + bsx])
    mask = np.ones(overlap.shape, 'float32')
    best_x, best_y = best_match(np.copy(base_image), overlap, mask, xi, xf, yi, yf)
    last_block_fixer(image, base_image, best_y, best_x, stepy, stepx, bsx, bsy, osx, osy)
    return image


def last_block_fixer(image, base_image, by, bx, stepy, stepx, bsx, bsy, osx, osy):
    temp = np.copy(base_image[by: by + bsy, bx:bx + bsx])
    a1 = image[stepy:stepy + osy, stepx: stepx + bsx]
    b1 = temp[: osy, :]
    path1 = col_min_path(a1, b1)

    a2 = image[stepy:stepy + bsy, stepx:stepx + osx]
    b2 = temp[:, :osx]
    path2 = row_min_path(a2, b2)

    a3 = image[stepy: stepy + bsy, stepx + osx: stepx + osx + osx]
    b3 = temp[:, osx:]
    path3 = row_min_path(a3, b3)

    a4 = image[stepy + osy: stepy + osy + osy, stepx: stepx + bsx]
    b4 = temp[osy:, :]
    path4 = col_min_path(a4, b4)

    t = np.copy(base_image[by: by + bsy, bx: bx + bsx])
    maper, maperr = maper_filler5(path1, path2, path3, path4, bsx, bsy, osx, osy)

    al = 5
    bl = 0
    g = np.zeros((maper.shape[0] + 2 * al, maper.shape[1] + 2 * al, 3), maper.dtype)
    g[al:-al, al:-al] = maperr

    g = cv2.GaussianBlur(g, (al, al), sigmaX=bl, sigmaY=bl)
    g_prime = 1 - g
    maperr = g[al:-al, al:-al]
    maper = g_prime[al:-al, al:-al]

    t = np.multiply(t, maperr)
    image[stepy:stepy + bsy, stepx: stepx + bsx] = np.multiply(image[stepy:stepy + bsy, stepx: stepx + bsx], maper)

    image[stepy:stepy + bsy, stepx: stepx + bsx] = image[stepy:stepy + bsy, stepx: stepx + bsx] + t
    pass


def maper_filler5(path1, path2, path3, path4, bsx, bsy, osx, osy):
    pf1, pf1r = path_filler_fixer_helper(path1, 'col', bsx, bsy, osx, osy)
    pf2, pf2r = path_filler_fixer_helper(path2, 'row', bsx, bsy, osx, osy)
    pf3, pf3r = path_filler_fixer_helper(path3, 'row', bsx, bsy, osx, osy)
    pf4, pf4r = path_filler_fixer_helper(path4, 'col', bsx, bsy, osx, osy)

    pf3r[pf3r == 1] = 2
    pf3r[pf3r == -1] = 1
    pf3r[pf3r == 2] = -1

    pf4r[pf4r == 1] = 2
    pf4r[pf4r == -1] = 1
    pf4r[pf4r == 2] = -1

    maper = np.zeros((bsy, bsx), path1.dtype)
    maper[:, :] = 0
    maper[:osy, :] = pf1r
    maper[:, :osx] = maper[:, :osx] + pf2r
    maper[osy:, :] = maper[osy:, :] + pf4r
    maper[:, osx:] = maper[:, osx:] + pf3r

    maper[maper == 0] = -1
    maper[maper == -2] = -1
    maper[maper == 2] = 1
    maper[maper == 1] = 0
    maper[maper == -1] = 1
    maper[maper == 3] = 0

    return merger(np.copy(maper), np.copy(maper), np.copy(maper)), map_reverse(
        merger(np.copy(maper), np.copy(maper), np.copy(maper)))



def run():
    texture = cv2.imread('Resources/im04.png')
    texture = texture.astype('float32')

    batch_size = 70
    overlap_size = 23
    height = 525
    width = 300
    xstart = 720
    ystart = 670
    xend = xstart + width
    yend = ystart + height
    res = np.copy(texture)

    res2 = filler(res, texture, batch_size, overlap_size, 'Result/HV3.png', xstart, xend, ystart, yend, height, width)
    res3 = last_row_fixer(res2, texture, batch_size, xstart, xend, ystart, yend)
    res4 = last_col_fixer(res3, texture, batch_size, xstart, xend, ystart, yend)
    res4 = last_block(res4, texture, batch_size, xstart, xend, ystart, yend)
    cv2.imwrite('Result/res16.jpg', res4)

    print('image{} finished'.format(1))
