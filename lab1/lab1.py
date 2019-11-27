from sklearn.preprocessing import MinMaxScaler
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})


def plot_data(x, y):
    plt.scatter(x, y)


def load_data(filename, normalize=False):
    data = pd.read_csv(filename, header=None)
    if normalize:
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(data)
        data = pd.DataFrame(scaled_values)

    cols = len(data.columns)
    X = np.array(data[0])
    for i in range(1, cols - 1):
        x = np.array(data[i], dtype=float)
        X = np.column_stack((X, x))
    Y = np.array(data[cols - 1], dtype=float)
    return X, Y


class LinearRegressionNonVectorized:
    def __init__(self, x, y, learning_rate=0.01):
        self.n = len(y)
        self.X = np.array(x)
        self.Y = np.array(y)
        self.w0 = 0
        self.w1 = 0
        self.w2 = 0
        if len(self.X.shape) != 2 or self.X.shape[1] == 1:
            self.X = np.column_stack((self.X, np.zeros(self.n)))
        self.learning_rate = learning_rate
        self.learning_history = []

    def y(self, w0, w1, x):
        return w0 + w1 * x

    def loss(self, w0, w1, w2):
        prediction = w0 + w1 * self.X.T[0] + w2 * self.X.T[1]
        error = self.Y - prediction
        return np.sum(np.dot(error, error.T)) / (2*self.n)

    def calculate_descent_direction(self):
        w0_dir = (2/self.n) * np.sum(self.X.T[1] * self.w2 + self.X.T[0] * self.w1 + self.w0 - self.Y)
        w1_dir = (2/self.n) * np.sum((self.X.T[1] * self.w2 + self.X.T[0] * self.w1 + self.w0 - self.Y) * self.X.T[0])
        w2_dir = (2/self.n) * np.sum((self.X.T[1] * self.w2 + self.X.T[0] * self.w1 + self.w0 - self.Y) * self.X.T[1])
        return w0_dir, w1_dir, w2_dir

    def train(self, silent=False, number_of_iterations=1000):
        self.learning_history = []
        for iteration in range(number_of_iterations):
            w0_dir, w1_dir, w2_dir = self.calculate_descent_direction()
            self.w0 = self.w0 - self.learning_rate * w0_dir
            self.w1 = self.w1 - self.learning_rate * w1_dir
            self.w2 = self.w2 - self.learning_rate * w2_dir
            new_err = self.loss(self.w0, self.w1, self.w2)
            self.learning_history.append(new_err)

        if not silent:
            print("Loss function: " + str(self.loss(self.w0, self.w1, self.w2)))

    def plot_learning_curve(self, show=True):
        plt.plot(self.learning_history)
        if show:
            plt.show()

    def plot3d(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        w1 = np.arange(-10, 10, 0.1)
        w2 = np.arange(-1, 4, 0.1)
        n = len(w1)
        m = len(w2)
        w1, w2 = np.meshgrid(w1, w2)
        E = np.array(np.zeros((m, n), dtype=float))
        for i in range(m):
            for j in range(n):
                E[i, j] = self.loss(w1[i, j], w2[i, j], 0)

        ax.plot_surface(w1, w2, E)
        plt.show()

        fig = plt.figure()
        ax = fig.gca()
        contour = ax.contour(w1, w2, E)
        plt.plot(self.w0, self.w1, 'b+')
        ax.clabel(contour, fontsize=9, inline=1)

    def plot2d(self):
        min_x = np.min(self.X)
        max_x = np.max(self.X)
        plt.plot([min_x, max_x], [self.y(self.w0, self.w1, min_x), self.y(self.w0, self.w1, max_x)], color='red')


class LinearRegression:
    def __init__(self, x, y, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.learning_history = []
        self.n = len(y)
        e = np.array([1.0] * self.n)
        self.X = np.column_stack((e, x))
        self.Y = y
        self.model_parameters = np.array([0] * len(self.X[0]))
        self.mse_values = []

    def predict(self, x):
        return np.dot(self.model_parameters, [1, x])

    def loss(self):
        error = self.Y - (np.dot(self.X, self.model_parameters))
        return np.sum(np.dot(error, error.T)) / (2*self.n)

    def calculate_descent_direction(self):
        dw = []
        E = np.dot(self.X, self.model_parameters.T) - self.Y
        dw0 = np.dot(E, self.X.T[0]) / self.n
        dw.append(dw0)
        for i in range(1, len(self.model_parameters)):
            dwi = np.dot(E, self.X.T[i])
            dw.append(dwi / self.n)
        dw = np.array(dw)
        return dw

    def train(self, iteration_count=1000):
        for it in range(iteration_count):
            dw = self.calculate_descent_direction()
            self.model_parameters = self.model_parameters - self.learning_rate * dw
            new_err = self.loss()
            self.learning_history.append(new_err)

    def calculate_analytics(self):
        self.model_parameters = np.dot(np.linalg.inv(np.dot(self.X.T, self.X)), np.dot(self.X.T, self.Y))

    def calculate_analytic_solution(self):
        dw = np.dot(np.dot(self.X.T, self.X)**-1, np.dot(self.X.T, self.Y))
        return dw

    def plot_learning(self):
        x = list(range(0, len(self.learning_history)))
        plt.plot(x, self.learning_history)


if __name__ == "__main__":
    # 1
    x, y = load_data("ex1data1.txt")

    # 2
    plot_data(x, y)
    plt.show()

    # 3
    regression1 = LinearRegressionNonVectorized(x, y)
    regression1.train()

    # 4
    regression1.plot2d()
    plot_data(x, y)
    plt.show()

    # 5
    regression1.plot3d()
    plt.show()

    # 6
    x_norm, y_norm = load_data("ex1data2.txt", normalize=True)

    # 7
    regression2 = LinearRegressionNonVectorized(x_norm, y_norm, learning_rate=1e-1)
    regression2.train()

    x, y = load_data("ex1data2.txt", normalize=False)
    regression21 = LinearRegressionNonVectorized(x, y, learning_rate=1e-8)
    regression21.train()

    plt.plot(regression2.learning_history)
    plt.show()

    plt.plot(regression21.learning_history)
    plt.show()

    # 8 9
    bars = []
    start_time = time.time()
    for _ in range(25):
        regression2 = LinearRegressionNonVectorized(x_norm, y_norm)
        regression2.train(True)
    elapsed = time.time() - start_time
    bars.append(elapsed)

    start_time = time.time()
    for _ in range(25):
        regression3 = LinearRegression(x_norm, y_norm)
        regression3.train(True)
    elapsed = time.time() - start_time
    bars.append(elapsed)

    plt.bar([0, 1], bars)
    plt.xticks([0, 1], ('Non vectorized', 'Vectorized'))
    plt.show()

    # 10
    for rate in np.arange(0.001, 0.01, 0.001):
        regression3 = LinearRegression(x_norm, y_norm, rate)
        regression3.train()
        plt.plot(regression3.learning_history, label=str(rate))

    plt.legend()
    plt.show()

    # 11
    regression = LinearRegression(x_norm, y_norm)
    regression.calculate_analytics()
    print(regression.model_parameters)
    print(regression.loss())

    regression = LinearRegression(x_norm, y_norm, learning_rate=0.1)
    regression.train()
    print(regression.model_parameters)
    print(regression.loss())
