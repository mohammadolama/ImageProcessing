import matplotlib.pyplot as plt
import numpy as np


def run_k_means(X, K, max_iteration, centers):
    index = np.zeros((len(X)), 'uint8')
    for i in range(0, max_iteration):
        index = find_closest_center(X, K, centers)

        centers = compute_new_centers(X, index, K)
        pass
    return X, index, centers
    pass


def random_center(X, K):
    np.random.shuffle(X)
    centers = X[0:K, :]
    return centers
    pass


def compute_new_centers(X, idx, K):
    m = X.shape[0]
    n = X.shape[1]
    centers = np.zeros((K, n))
    count = np.zeros(K)
    for i in range(0, m):
        r = idx[i]
        count[r] += 1
        centers[r, :] += X[i, :]

    for i in range(0, K):
        centers[i, :] = centers[i, :] / (count[i])
    return centers


def find_closest_center(X, K, centers):
    m = X.shape[0]
    n = X.shape[1]
    idx = np.zeros(m, 'uint8')
    for i in range(0, m):
        dis = 1000000000
        u = 600000

        for j in range(0, K):
            x = X[i, :]
            c = centers[j, :]

            v_norm = 0
            for t in range(0, n):
                v_norm += (x[t] - c[t]) ** 2

            if v_norm < dis:
                u = j
                dis = v_norm

        idx[i] = u
        pass

    return idx

    pass


def plot2d(Y, index, centers, num_lines, name):
    figure1 = plt.figure()
    for i in range(0, num_lines):
        if index is None or index[i] == 0:
            plt.scatter(Y[i, 0], Y[i, 1], color='r')
            pass
        else:
            plt.scatter(Y[i, 0], Y[i, 1], color='b')
            pass

    if centers is not None:
        for i in range(0, centers.shape[0]):
            plt.scatter(centers[i, 0], centers[i, 1], color='g')
    plt.savefig(name)


def plot3d(Y, index, centers, num_lines, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0, num_lines):
        if index is None or index[i] == 0:
            ax.plot3D(Y[i, 0], Y[i, 1], Y[i, 2], 'ro')

        else:
            ax.plot3D(Y[i, 0], Y[i, 1], Y[i, 2], 'bo')

    for i in range(0, centers.shape[0]):
        ax.plot3D(centers[i, 0], centers[i, 1], centers[i, 2], 'gx')
    plt.savefig(name, dpi=1000)
    plt.show()

    pass


def run(k, max_iter):
    with open("Resources/Points.txt", 'r') as f:
        contents = f.readlines()

    num_lines = int(contents[0])

    x = []
    y = []
    z = []
    X = np.zeros((num_lines, 3))
    j = 0
    for i in range(0, int(num_lines)):
        string = contents[i + 1].split()
        xp = float(string[0])
        yp = float(string[1])
        zp = 10 * np.sqrt(xp ** 2 + yp ** 2) ** 2
        x.append(xp)
        y.append(yp)
        z.append(zp)
        X[i] = [xp, yp, zp]

        j += 1
    print(j)

    plot2d(X, None, None, num_lines, "Result/res01.jpg")

    Y, index, centers = run_k_means(X[:, 0:2], K=k, max_iteration=max_iter, centers=random_center(np.copy(X), K=k))
    plot2d(Y, index, centers, num_lines, "Result/res02.jpg")

    Y, index, centers = run_k_means(X[:, 0:2], K=k, max_iteration=max_iter, centers=random_center(np.copy(X), K=k))
    plot2d(Y, index, centers, num_lines, "Result/res03.jpg")

    Y, index, centers = run_k_means(X[:, 0:3], K=k, max_iteration=max_iter, centers=random_center(np.copy(X), K=k))

    print(Y)
    plot3d(Y, index, centers, num_lines, "Result/res04.jpg")


run(k=2, max_iter=10)
