import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import sys


class LinearRegression:
    def __init__(self, x, y, learning_rate=0.01, reg_coeff=0.0, p=1):
        self.learning_rate = learning_rate
        self.learning_history = []
        self.n = len(x)
        singular_column = np.array([1.0] * self.n)
        singular_column.shape = (self.n, 1)
        poly = PolynomialFeatures(degree=p)
        self.X = poly.fit_transform(x)
        self.Y = y
        self.features = len(self.X[0])
        self.model_parameters = np.random.rand(self.features)
        self.reg_coeff = reg_coeff
        self.p = p

    def predict(self, x):
        poly = PolynomialFeatures(degree=self.p)
        n = len(x)
        x.shape = (n, 1)
        x = poly.fit_transform(x)
        return np.dot(self.model_parameters, x.T)

    def mse(self):
        error = np.dot(self.X, self.model_parameters.T) - self.Y
        reg = self.reg_coeff * self.model_parameters.dot(self.model_parameters)
        return (np.dot(error, error.T) + reg) / (2 * self.n)

    def calculate_descent_direction(self):
        E = np.dot(self.X, self.model_parameters.T) - self.Y
        dw0 = np.dot(E, self.X.T[0]) / self.n
        dw = [dw0]
        for i in range(1, self.features):
            dwi = np.dot(E, self.X.T[i])
            dw.append((dwi + self.reg_coeff * self.model_parameters[i]) / self.n)
        dw = np.array(dw)
        return dw

    def calculate_analytics(self):
        return np.dot(np.dot(self.X.T, self.X)**-1, np.dot(self.X.T, self.Y))

    def train(self, n_iterations=1000):
        for _ in range(n_iterations):
            dw = self.calculate_descent_direction()
            self.model_parameters = self.model_parameters - self.learning_rate * dw
            new_err = self.mse()
            self.learning_history.append(new_err)

    def plot_learning_curve(self):
        plt.plot(reg.learning_history, label=str(len(self.X)))
        plt.ylabel("Cost function")
        plt.xlabel("Iterations")

    def plot2d(self):
        min_x = -0.5
        max_x = 0.5
        x_values = np.arange(min_x, max_x, 0.01)
        plt.plot(x_values, self.predict(x_values), label='reg_coeff=' + str(self.reg_coeff))


def find_error(model: LinearRegression, x, y):
    y_predicted = model.predict(x)
    E = y_predicted - y
    error = np.dot(E.T, E)
    return error


if __name__ == "__main__":
    # 1
    mat = scipy.io.loadmat('ex3data1.mat')
    X = mat['X'] / np.linalg.norm(mat['X'])
    Y = mat['y'].T[0]
    Y = Y / np.linalg.norm(Y)
    X_val = mat['Xval'] / np.linalg.norm(mat['Xval'])
    Y_val = mat['yval'].T[0]
    Y_val = Y_val / np.linalg.norm(Y_val)
    X_test = mat['Xtest'] / np.linalg.norm(mat['Xtest'])
    Y_test = mat['ytest'].T[0]
    Y_test = Y_test / np.linalg.norm(Y_test)

    # 2
    plt.scatter(X_val, Y_val)
    plt.ylabel("Amount of poured water")
    plt.xlabel("Change in water level")
    plt.show()

    # 3 4
    reg = LinearRegression(X, Y, learning_rate=1)

    # 5
    reg.train(n_iterations=60)
    reg.plot2d()
    plt.scatter(X, Y)
    plt.ylabel("Amount of poured water")
    plt.xlabel("Change in water level")
    plt.show()

    # 6
    reg.plot_learning_curve()
    reg = LinearRegression(X_val, Y_val, learning_rate=1)
    reg.train(n_iterations=60)
    reg.plot_learning_curve()
    plt.legend()
    plt.show()

    # 7 8 9
    reg = LinearRegression(X, Y, p=8, learning_rate=1, reg_coeff=0)
    reg.train(n_iterations=100000)

    # 10
    reg.plot2d()
    plt.scatter(X, Y)
    plt.show()

    # 11
    plt.scatter(X, Y)
    reg.learning_rate = 0.1
    for reg_coeff in [1, 100]:
        reg.reg_coeff = reg_coeff
        reg.train(n_iterations=1000)
        reg.plot2d()

    plt.legend()
    plt.show()

    reg.learning_rate = 1
    plt.scatter(X_val, Y_val)
    bars = []
    reg_coeffs = np.arange(0.0001, 0.0008, 0.0001)
    for reg_coeff in reg_coeffs:
        reg.reg_coeff = reg_coeff
        reg.train(n_iterations=10000)
        reg.plot2d()
        err = find_error(reg, X_val, Y_val)
        bars.append(err)
    plt.legend()
    plt.show()

    plt.bar(range(len(reg_coeffs)), bars)
    plt.ylabel('Error for validation set')
    plt.xlabel('Regularization coefficient')
    plt.xticks(range(len(reg_coeffs)), [str(round(c, 5)) for c in reg_coeffs])

    plt.show()

    best_reg_coeff = reg_coeffs[np.argmin(bars)]
    reg.reg_coeff = best_reg_coeff
    print("Best regularization coeff is " + str(best_reg_coeff))

    err = find_error(reg, X_test, Y_test)
    print("Error in test set is " + str(err))

    reg.plot2d()
    plt.scatter(X_test, Y_test)
    plt.show()
