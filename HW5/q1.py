import time

import imageio
import cv2
import numpy as np
from scipy.spatial import Delaunay


def warpTriangle(image1, image2, points1, points2):
    r1 = cv2.boundingRect(points1)
    r2 = cv2.boundingRect(points2)

    t1points = []
    t2points = []

    t1points.append(((points1[0][0] - r1[0]), (points1[0][1] - r1[1])))
    t2points.append(((points2[0][0] - r2[0]), (points2[0][1] - r2[1])))
    t1points.append(((points1[1][0] - r1[0]), (points1[1][1] - r1[1])))
    t2points.append(((points2[1][0] - r2[0]), (points2[1][1] - r2[1])))
    t1points.append(((points1[2][0] - r1[0]), (points1[2][1] - r1[1])))
    t2points.append(((points2[2][0] - r2[0]), (points2[2][1] - r2[1])))

    image1Cropped = image1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    m = cv2.getAffineTransform(np.float32(t1points), np.float32(t2points))

    warped_image = cv2.warpAffine(image1Cropped, m, (r2[2], r2[3]), None, flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REFLECT_101)

    x = image2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    mask = np.zeros(x.shape, dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2points), (1.0, 1.0, 1.0), 16, 0)

    image2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = image2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
            (1.0, 1.0, 1.0) - mask) + warped_image * mask


start = time.time()
bradpitt = cv2.imread("Resources/im1.png")
bradpitt = bradpitt.astype('float64')
tomcruise = cv2.imread("Resources/im2.png")
tomcruise = tomcruise.astype('float64')

new_brad = np.zeros((bradpitt.shape[0] + 2, bradpitt.shape[1] + 2, 3), bradpitt.dtype)
new_brad[1:bradpitt.shape[0] + 1, 1: bradpitt.shape[1] + 1] = bradpitt
new_tom = np.zeros((tomcruise.shape[0] + 2, tomcruise.shape[1] + 2, 3), tomcruise.dtype)
new_tom[1:tomcruise.shape[0] + 1, 1: tomcruise.shape[1] + 1] = tomcruise

cv2.imwrite("Result/res01.jpg", new_brad)
cv2.imwrite("Result/res02.jpg", new_tom)

tomcruise = new_tom
bradpitt = new_brad
with open("Resources/points1.txt", 'r') as f:
    brad_points = f.readlines()
with open("Resources/points2.txt", 'r') as f:
    tom_points = f.readlines()

brad_x = []
brad_y = []
brad_T = []
for i in range(0, 126):
    string = brad_points[i].split()
    xp = int(string[0])
    yp = int(string[1])
    brad_T.append([xp + 1, yp + 1])
    brad_x.append(xp + 1)
    brad_y.append(yp + 1)
shape = tuple(zip(brad_x, brad_y))
b_points = np.array(brad_T)

tom_x = []
tom_y = []
tom_T = []
for i in range(0, 126):
    string = tom_points[i].split()
    xp = int(string[0])
    yp = int(string[1])
    tom_T.append([xp + 1, yp + 1])
    tom_x.append(xp + 1)
    tom_y.append(yp + 1)
shape = tuple(zip(tom_x, tom_y))
t_points = np.array(tom_T)

b_triangles = Delaunay(b_points)
print(b_triangles.simplices.shape)
print("****************************")
print(t_points[0])
number = 45
rate = 1 / number
listofimages = []
for i in range(0, number + 1):
    alpha = i * rate
    positions = (1 - alpha) * b_points + alpha * t_points
    positions = positions.astype('int32')
    brad_output = np.zeros(bradpitt.shape, dtype=bradpitt.dtype)
    tom_output = np.zeros(tomcruise.shape, dtype=tomcruise.dtype)

    for j in range(0, b_triangles.simplices.shape[0]):
        triplet_brad = b_triangles.simplices[j]
        triplet_tom = triplet_brad

        A = np.array([b_points[triplet_brad[0]], b_points[triplet_brad[1]], b_points[triplet_brad[2]]], "int32")
        B = np.array([positions[triplet_brad[0]], positions[triplet_brad[1]], positions[triplet_brad[2]]], "int32")
        C = np.array([t_points[triplet_tom[0]], t_points[triplet_tom[1]], t_points[triplet_tom[2]]], "int32")
        D = np.array([positions[triplet_tom[0]], positions[triplet_tom[1]], positions[triplet_tom[2]]], "int32")

        warpTriangle(bradpitt, brad_output, A, B)
        warpTriangle(tomcruise, tom_output, C, D)
        pass
    output = (1 - alpha) * brad_output + alpha * tom_output
    listofimages.append(output)

cv2.imwrite("Result/res03.jpg", listofimages[14])
cv2.imwrite("Result/res04.jpg", listofimages[29])

new_list = []
for i in range(0, 45):
    new_list.append(cv2.cvtColor((listofimages[i]).astype('uint8'), cv2.COLOR_BGR2RGB))
for i in range(44, -1, -1):
    new_list.append(cv2.cvtColor((listofimages[i]).astype('uint8'), cv2.COLOR_BGR2RGB))

imageio.mimsave("Result/morph.mp4", new_list, fps=30)
print(b_triangles.simplices.shape)
print(b_triangles.simplices[0, 1])
print(b_triangles.simplices[0][1])

print(time.time() - start)
