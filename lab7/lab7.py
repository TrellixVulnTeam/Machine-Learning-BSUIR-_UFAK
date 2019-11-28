from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import imageio
from mpl_toolkits.mplot3d import Axes3D


# 3
def compute_covariance_matrix(x):
    m = len(x)
    return (1 / m) * np.dot(x.T, x)


# 6
def project_data(k, u, x):
    u = u[:, :k]
    z = np.dot(x, u)
    return z


# 7
def recover_data(z, u, k):
    return z.dot(u[:, :k].T)


def normalize_data(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    mean = np.mean(data)
    data -= mean
    return data


if __name__ == "__main__":
    # 1
    mat = loadmat("ex7data1")
    x = mat['X']

    # 2
    plt.scatter(x.T[0], x.T[1])
    plt.show()

    # 4
    x_norm = normalize_data(x)
    sigma = compute_covariance_matrix(x_norm)
    u, s, v = np.linalg.svd(sigma)
    u1 = [u[0][0], u[1][0]]
    u2 = [u[0][1], u[1][1]]

    print("u1 = " + str(u1))
    print("u2 = " + str(u2))

    # 5
    plt.scatter(x_norm.T[0], x_norm.T[1])

    plt.plot([0, u1[0]], [0, u1[1]], color='red')
    plt.plot([0, u2[0]], [0, u2[1]], color='red')

    plt.show()

    # 8
    k = 1
    z = project_data(k, u, x_norm)
    x_recovered = recover_data(z, u, k)

    plt.figure(figsize=(8, 8))
    plt.plot([0, u[0][0]], [0, u[1][0]], color='green')
    plt.scatter(x_norm.T[0], x_norm.T[1], s=20, label='Original Data Points')
    plt.plot(x_recovered.T[0], x_recovered.T[1], 'ro', mfc='none', mec='r', ms=3, label='PCA Reduced Data Points')
    plt.xlabel('x1', fontsize=14)
    plt.ylabel('x2', fontsize=14)
    plt.legend(loc=4)
    for (x, y), (x_rec, y_rec) in zip(x_norm, x_recovered):
        plt.plot([x, x_rec], [y, y_rec], 'k--', lw=0.5, color='grey')

    plt.xlim(-0.7, 0.7)
    plt.ylim(-0.6, 0.8)

    plt.show()

    # 9
    mat = loadmat('ex7faces.mat')
    x = mat['X']

    # 10
    fig = plt.figure(figsize=(16, 16))
    columns = 10
    rows = 10
    for i in range(1, columns * rows + 1):
        img = np.array(random.choice(x))
        img = img.reshape(32, 32).T
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, interpolation='nearest', cmap='gray')
    plt.show()

    # 11
    mean = np.mean(x)
    x -= mean
    sigma = compute_covariance_matrix(x)
    u, s, v = np.linalg.svd(sigma)

    def visualize_components(cols, rows):
        fig = plt.figure(figsize=(cols, rows))
        for i in range(1, cols * rows + 1):
            img = u[:, i: i + 1].flatten()
            img = img.reshape(32, 32).T
            fig.add_subplot(rows, cols, i)
            plt.imshow(img, interpolation='nearest', cmap='gray')
        plt.show()

    # 12
    visualize_components(6, 6)

    # 13
    visualize_components(10, 10)

    # 16
    A = imageio.imread("bird_16.png")
    X = A.reshape(-1, 3).astype(np.float64)

    # 17
    sel = np.random.choice(X.shape[0], size=1000)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    idx = np.loadtxt('output.txt')
    ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], cmap='rainbow', c=idx[sel], s=10)
    plt.show()

    # 18
    mean = np.mean(X, axis=0)
    X -= mean
    sigma = compute_covariance_matrix(X)
    u, s, v = np.linalg.svd(sigma)

    z = project_data(2, u, X)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    ax.scatter(z[sel, 0], z[sel, 1], cmap='rainbow', c=idx[sel], s=32)
    plt.show()
