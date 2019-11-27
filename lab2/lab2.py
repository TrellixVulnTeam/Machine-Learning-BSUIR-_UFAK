import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import scipy.io
import random
import sys
from matplotlib.pyplot import cm


def read_csv(filename):
    csv_data = pd.read_csv(filename, header=None)
    cols = len(csv_data.columns)
    X = np.array([np.array(csv_data[col]) for col in range(cols - 1)]).T
    y = np.array(csv_data[cols - 1], dtype=float)
    return X, y


def plot_test_results(x, y):
    accepted_idx = y == 1
    plt.plot(x[accepted_idx].T[0], x[accepted_idx].T[1], 'go', label="admitted")
    plt.plot(x[~accepted_idx].T[0], x[~accepted_idx].T[1], 'rx', label="not admitted")
    plt.xlabel('Exam 1')
    plt.ylabel('Exam 2')
    plt.legend(loc=3)


class LogisticRegression:
    def __init__(self, X, y, learning_rate=0.001, reg_coeff=0.0, polynom=1):
        self.learning_rate = learning_rate
        self.learning_history = []
        row_count = len(X)
        col_count = len(X[0])
        self.X = np.array(X)
        self.min_x = np.min(self.X)
        self.max_x = np.max(self.X)
        singular_column = np.array([1.0] * row_count)
        singular_column.shape = (row_count, 1)
        self.X = np.hstack((singular_column, self.X))
        self.Y = np.array(y)

        if col_count > 2:
            features_count = col_count + 1
        else:
            features_count = 0
            for i in range(1, polynom + 2):
                features_count += i
        self.model_parameters = np.array([0] * features_count)
        powers = []
        for i in range(0, polynom + 1):
            for j in range(0, polynom + 1):
                if i + j <= polynom:
                    powers.append((i, j))
        self.powers = sorted(powers, key=lambda p: p[0] + p[1])
        if polynom > 1:
            self.X = np.array([self.compute_polynom(x[1:]) for x in self.X])
        self.polynom = polynom
        self.reg_coeff = reg_coeff

    def compute_polynom(self, x_vector):
        p_len = len(self.powers)
        x_vector = np.repeat(np.array([x_vector]), p_len, axis=0)
        return np.prod(x_vector**self.powers, axis=1)

    @staticmethod
    def sigmoid(v):
        return 1.0 / (1 + np.exp(-v))

    def cost(self, theta):
        m = self.X.shape[0]
        h = LogisticRegression.sigmoid(np.matmul(self.X, theta))
        cost = np.matmul(-self.Y.T, np.log(h)) - np.matmul((1 - self.Y.T), np.log(1 - h))
        cost += (1/2) * self.reg_coeff * theta.dot(theta)
        cost /= m
        self.learning_history.append(cost)
        return cost

    def cost_gradient(self, theta):
        m = self.X.shape[0]
        h = self.sigmoid(np.dot(self.X, theta))
        grad = np.dot(self.X.T, (h - self.Y)) / m
        grad[1:] = grad[1:] + (self.reg_coeff / m) * theta[1:]
        return grad

    def train(self, mode='gradient', n_iterations=100000):
        self.learning_history = []
        self.model_parameters.fill(0)
        if mode == 'gradient':
            for _ in range(n_iterations):
                grad = self.cost_gradient(self.model_parameters)
                self.model_parameters = self.model_parameters - self.learning_rate * grad
                self.cost(self.model_parameters)
        elif mode == 'nedler':
            res = minimize(self.cost, self.model_parameters, method='nelder-mead',
                           options={'xtol': 1e-4, 'disp': True})
            self.model_parameters = np.array(res.x)
        elif mode == 'bro':
            res = minimize(self.cost, self.model_parameters, method='BFGS', jac=self.cost_gradient,
                           options={'disp': True})
            self.model_parameters = np.array(res.x)

    def predict(self, x):
        if self.polynom > 1:
            t = self.model_parameters.dot(self.compute_polynom(x))
        else:
            t = self.model_parameters.dot(np.insert(x, 0, 1))
        pred = 1 / (1 + np.exp(-t))
        return pred

    def surface(self, x):
        return (-self.model_parameters[0] - self.model_parameters[1] * x) / self.model_parameters[2]

    def poly_surface(self):
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
        n = len(u)
        m = len(v)
        xx, yy = np.meshgrid(u, v)
        for i in range(m):
            for j in range(n):
                z[i, j] = self.model_parameters.dot(self.compute_polynom([xx[i, j], yy[i, j]]))

        plt.contour(u, v, z, 0)

    def plot_surface(self):
        x = [self.min_x, self.max_x]
        y = [self.surface(x[0]), self.surface(x[1])]
        plt.plot(x, y, color='green')


if __name__ == "__main__":
    # 1
    X, y = read_csv("ex2data1.txt")

    # 2
    plot_test_results(X, y)

    # 3 4 5
    logistic = LogisticRegression(X, y)
    logistic.train('nedler')

    # 6
    logistic.plot_surface()

    plt.show()

    # 7
    X, y = read_csv("ex2data2.txt")

    # 8
    plot_test_results(X, y)

    plt.show()

    # 9 10
    logistic = LogisticRegression(X, y, polynom=6, reg_coeff=0.0001, learning_rate=15)
    logistic.train('gradient', n_iterations=1000)
    print("Gradient: " + str(logistic.cost(logistic.model_parameters)))
    plt.plot(logistic.learning_history, label=str("Gradient"))

    # 11
    logistic.train('nedler')
    print("Nedler-mean: " + str(logistic.cost(logistic.model_parameters)))
    plt.plot(logistic.learning_history, label=str("Nedler"))

    logistic.train('bro')
    print("Bro: " + str(logistic.cost(logistic.model_parameters)))
    plt.plot(logistic.learning_history, label=str("Broyden-Fletcher"))

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Cost function')
    plt.show()

    # 13
    plot_test_results(X, y)
    logistic.poly_surface()
    plt.show()

    # 14
    bars = []
    reg_coeffs = [0.001, 0.01, 1, 10]
    for reg_coeff in reg_coeffs:
        logistic.reg_coeff = reg_coeff
        logistic.train('bro')
        logistic.poly_surface()
        plot_test_results(X, y)
        bars.append(logistic.cost(logistic.model_parameters))
        plt.show()

    plt.bar(range(len(reg_coeffs)), bars)
    plt.ylabel('Cost function')
    plt.xlabel('Regularization coefficient')
    plt.xticks(range(len(reg_coeffs)), [str(c) for c in reg_coeffs])
    plt.show()

    # 15
    mat = scipy.io.loadmat('ex2data3.mat')
    X = mat['X']
    y = np.array([label[0] for label in mat['y']])

    # 16
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(1, columns * rows + 1):
        img = np.array(random.choice(X))
        img = img.reshape(20, 20)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, interpolation='nearest', cmap='Greys')
    plt.show()

    # 17 18 19
    classes = list(range(1, 11))
    predictors = []
    for cls in classes:
        mask = y == cls
        sample_y = np.zeros(len(y))
        sample_y[mask] = 1
        predictor = LogisticRegression(X, sample_y, learning_rate=10)
        predictor.train('gradient', n_iterations=1500)
        predictors.append(predictor)
        print("Class " + str(cls) + " trained.")

    # 20
    def predict_class(predictors, image):
        predictions = [p.predict(image) for p in predictors]
        index_of_max = np.argmax(predictions)
        return index_of_max + 1

    # 21
    correct = 0
    incorrect = 0
    for x, label in zip(X, y):
        predicted_class = predict_class(predictors, x)
        if predicted_class == label:
            correct += 1
        else:
            incorrect += 1

    correct_rate = correct / len(y)
    print("Correct predictions: " + str(correct_rate * 100) + "%")

