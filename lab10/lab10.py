from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


def predict(predictors, coefficients, x):
    prediction = np.zeros(len(x))
    for predictor, coeff in zip(predictors, coefficients):
        predictions = predictor.predict(x) * coeff
        prediction += predictions
    return prediction


# 7
def cost(predictors, coefficients, x, y):
    prediction = predict(predictors, coefficients, x)
    return mean_squared_error(y, prediction)


# 4
def train_predictors(coefficients, x_train, y_train, m=50, d=5):
    yi = y_train
    predictors = []
    for i in range(m):
        tree = DecisionTreeRegressor(max_depth=d, random_state=42)
        predictors.append(tree)
        tree.fit(x_train, yi)
        prediction = coefficients[i] * tree.predict(x_train)
        e = yi - prediction
        yi = e
    return predictors


if __name__ == "__main__":
    # 1
    boston = load_boston()
    x = boston['data']
    y = boston['target']

    # 2
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 3
    m = 50
    coefficients = [0.9] * m

    # 5
    predictors = train_predictors(coefficients, x_train, y_train)
    test_cost = cost(predictors, coefficients, x_test, y_test)
    print("Cost of test set: " + str(test_cost))

    # 8
    coefficients = [0.9 / (1 + i) for i in range(m)]
    predictors = train_predictors(coefficients, x_train, y_train, m)
    test_cost = cost(predictors, coefficients, x_test, y_test)
    print("Cost of test set: " + str(test_cost))

    # 9
    bars = []
    iteration_values = [60, 80, 100, 120, 150]
    for m in iteration_values:
        coefficients = [0.9 / (1 + i) for i in range(m)]
        predictors = train_predictors(coefficients, x_train, y_train, m)
        test_cost = cost(predictors, coefficients, x_test, y_test)
        print("For m = " + str(m))
        print("Cost of test set: " + str(test_cost))
        bars.append(test_cost)

    plt.ylabel("Cost function")
    plt.xlabel("Number of iterations")
    plt.bar(range(len(iteration_values)), bars)
    plt.xticks(range(len(iteration_values)), [str(m) for m in iteration_values])
    plt.show()

    bars = []
    depth_values = [5, 6, 7, 8, 9]
    for d in depth_values:
        predictors = train_predictors(coefficients, x_train, y_train, d=d)
        test_cost = cost(predictors, coefficients, x_test, y_test)
        bars.append(test_cost)
        print("For d = " + str(d))
        print("Cost of test set: " + str(test_cost))

    plt.bar(range(len(depth_values)), bars)
    plt.ylabel("Cost function")
    plt.xlabel("Depth")
    plt.xticks(range(len(depth_values)), [str(d) for d in depth_values])
    plt.show()

    # 10
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    error = mean_squared_error(y_test, lin_reg.predict(x_test))
    print("Cost of lin reg algorithm: " + str(error))
