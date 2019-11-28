import scipy.io
import numpy as np
import random
import scipy.optimize as opt
import matplotlib.pyplot as plt


def sigmoid(v):
    return 1.0 / (1 + np.exp(-v))


# 8
def sigmoid_derivative(z):
    return np.multiply(sigmoid(z), 1-sigmoid(z))


class SigmoidLayer:
    def __init__(self, is_final_layer=False):
        self.output = []
        self.is_final_layer = is_final_layer

    def activate(self, input_vector, weight):
        result = np.dot(weight, input_vector)
        self.output = sigmoid(result)
        if not self.is_final_layer:
            self.output = np.insert(self.output, 0, 1.0)
        return self.output


class NeuralNetwork:
    def __init__(self, layer_weights):
        self.layers = []
        self.layer_weights = layer_weights
        for _ in range(len(layer_weights)):
            self.layers.append(SigmoidLayer())
        self.layers[-1].is_final_layer = True
        self.layers_num = len(self.layers)
        self.input_layer_size = self.layer_weights[0].shape[1] - 1
        self.hidden_layer_size = self.layer_weights[1].shape[1] - 1
        self.output_layer_size = self.layer_weights[1].shape[0]
        self.sizes = [self.input_layer_size, self.hidden_layer_size, self.output_layer_size]

    @staticmethod
    def initialize_weights(sizes):
        eps = 0.0001
        weights = []
        for i in range(len(sizes) - 1):
            in_size = sizes[i]
            out_size = sizes[i + 1]
            weight_matrix = np.random.rand(out_size, in_size + 1) * (2 * eps) - eps
            weights.append(weight_matrix)
        return weights

    @staticmethod
    def new(sizes):
        weights = NeuralNetwork.initialize_weights(sizes)
        return NeuralNetwork(weights)

    def propagate_forward(self, input_vector, weights=None):
        if weights == None:
            weights = self.layer_weights
        input_vector = np.insert(input_vector, 0, 1.0)
        for layer, weight in zip(self.layers, weights):
            output = layer.activate(input_vector, weight)
            input_vector = output
        return input_vector

    def train(self, x, y, reg_coeff=0.0):
        initial_weights = np.append(self.layer_weights[0].flatten(), self.layer_weights[1].flatten())
        optimal_weights = opt.fmin_cg(
            maxiter=30,
            f=self.cost_func,
            x0=initial_weights,
            fprime=self.back_propagate_func,
            args=(x, y, reg_coeff))
        self.layer_weights = self.reshape_weights(optimal_weights)

    def reshape_weights(self, flatten_weights):
        weights1 = np.reshape(flatten_weights[:self.hidden_layer_size * (self.input_layer_size + 1)],
                              (self.hidden_layer_size, self.input_layer_size + 1), 'F')
        weights2 = np.reshape(flatten_weights[self.hidden_layer_size * (self.input_layer_size + 1):],
                              (self.output_layer_size, self.hidden_layer_size + 1), 'F')
        return [weights1, weights2]

    def back_propagate_func(self, flatten_weights, x, y, reg_coeff):
        weights1, weights2 = self.reshape_weights(flatten_weights)
        delta1, delta2 = self.back_propagate(weights1, weights2, x, y, reg_coeff)
        return np.hstack((delta1.ravel(order='F'), delta2.ravel(order='F')))

    #  10
    def back_propagate(self, initial_theta1, initial_theta2, x, y, reg_coeff=0):
        delta1, delta2 = self.compute_gradient(x, y, [initial_theta1, initial_theta2])
        m = len(y)
        delta1[:, 1:] = delta1[:, 1:] + initial_theta1[:, 1:] * reg_coeff / m
        delta2[:, 1:] = delta2[:, 1:] + initial_theta2[:, 1:] * reg_coeff / m

        return delta1, delta2

    def compute_gradient(self, x, y, layer_weights=None):
        if layer_weights == None:
            layer_weights = self.layer_weights
        delta1 = np.zeros(layer_weights[0].shape)
        delta2 = np.zeros(layer_weights[1].shape)
        m = len(y)
        for x_i, y_i in zip(x, y):
            ones = np.ones(1)
            a1 = np.hstack((ones, x_i))
            z2 = np.dot(a1, layer_weights[0].T)
            a2 = np.hstack((ones, sigmoid(z2)))
            z3 = np.dot(a2, layer_weights[1].T)
            a3 = sigmoid(z3)
            d3 = a3 - y_i
            d2 = np.multiply(np.dot(layer_weights[1].T, d3.T), np.multiply(a2, 1 - a2))
            delta1 = delta1 + np.outer(d2[1:], a1)
            delta2 = delta2 + np.outer(d3, a2)
        delta1 /= m
        delta2 /= m
        return delta1, delta2

    # 11
    def check_gradient(self, gradient, x, y, n_checks=5):
        eps = 1e-4
        layer_num = 1
        layer = self.layer_weights[layer_num]
        rows = len(layer)
        cols = len(layer[0])

        for _ in range(n_checks):
            i = random.choice(range(rows))
            j = random.choice(range(cols))

            minus_layer = np.copy(layer)
            minus_layer[i][j] -= eps

            plus_layer = np.copy(layer)
            plus_layer[i][j] += eps

            plus_weights = list(self.layer_weights)
            plus_weights[layer_num] = plus_layer

            minus_weights = list(self.layer_weights)
            minus_weights[layer_num] = minus_layer

            plus_network = NeuralNetwork(minus_weights)
            minus_network = NeuralNetwork(plus_weights)

            approximation = (minus_network.cost(x, y, 0) - plus_network.cost(x, y, 0)) / (2 * eps)
            diff = approximation - gradient[layer_num][i][j]
            print("Diff is " + str(diff))

    def cost_func(self, flatten_weights, x, y, reg_coeff):
        weights1, weights2 = self.reshape_weights(flatten_weights)
        return self.cost(x, y, reg_coeff, [weights1, weights2])

    # 6 7
    def cost(self, x, y, reg_coeff, weights=None):
        if weights == None:
            weights = self.layer_weights
        total_sum = 0
        m = len(x)
        for x_i, y_i in zip(x, y):
            result = self.propagate_forward(x_i, weights)
            temp_sum = 0
            for output, y_i_k in zip(result, y_i):
                temp_sum += (y_i_k * np.log(output) + (1 - y_i_k) * np.log(1 - output))
            total_sum += temp_sum
        reg = (reg_coeff / 2) * np.sum([np.sum(theta**2) for theta in weights])
        return (- total_sum - reg) / m

    def visualize_hidden(self):
        layer = self.layer_weights[0][:, 1:]
        m, n = layer.shape
        width = int(np.round(np.sqrt(n)))
        rows = int(np.floor(np.sqrt(m)))
        cols = int(np.ceil(m / rows))
        fig, ax_array = plt.subplots(rows, cols, figsize=(10, 10))
        fig.subplots_adjust(wspace=0.025, hspace=0.025)
        ax_array = [ax_array] if m == 1 else ax_array.ravel()
        for i, ax in enumerate(ax_array):
            ax.imshow(layer[i].reshape(width, width, order='F'), cmap='Greys', extent=[0, 1, 0, 1])
            ax.axis('off')


def convert_to_one_hot(y_vector):
    result = []
    for y in y_vector:
        one_hot = np.zeros(10)
        one_hot[y - 1] = 1
        result.append(one_hot)
    return np.array(result)


def check_model(model: NeuralNetwork, x, y):
    correct = 0
    incorrect = 0
    for input_vector, label in zip(x, y):
        prediction = model.propagate_forward(input_vector)
        predicted_class = np.argmax(prediction) + 1
        if predicted_class == label:
            correct += 1
        else:
            incorrect += 1
    percent = correct / (correct + incorrect) * 100
    print("Correct predictions: " + str(correct))
    print("Incorrect predictions: " + str(incorrect))
    print("Percent of correct predictions is " + str(percent))
    return percent


if __name__ == "__main__":
    # 1
    mat = scipy.io.loadmat('ex4data1.mat')
    X = mat['X']
    Y = mat['y'].T[0]

    # 2
    trained_weights = scipy.io.loadmat("ex4weights.mat")
    theta1 = trained_weights["Theta1"]
    theta2 = trained_weights["Theta2"]

    # 3
    network = NeuralNetwork([theta1, theta2])

    # 4
    check_model(network, X, Y)

    # 5
    y_one_hot = convert_to_one_hot(Y)

    # 13
    grad = network.compute_gradient(X, y_one_hot)
    network.check_gradient(grad, X, y_one_hot)

    # 14
    network = NeuralNetwork.new([400, 25, 10])
    network.train(X, y_one_hot)

    # 15
    check_model(network, X, Y)

    # 16
    network.visualize_hidden()
    plt.show()

    # 17
    bars = []
    reg_coeffs = [0.0001, 0.001, 0.002, 0.005, 0.1, 0.2, 0.5]
    # we will use the same initial weights for each iteration
    initial_weights = network.initialize_weights(network.sizes)
    for reg_coeff in reg_coeffs:
        print("For reg coeff = " + str(reg_coeff))
        network.layer_weights = initial_weights
        network.train(X, y_one_hot, reg_coeff)
        percent_of_correct = check_model(network, X, Y)
        bars.append(percent_of_correct)
        network.visualize_hidden()
        plt.show()
        print("#==============================#")

    best_reg_coeff = reg_coeffs[np.argmax(bars)]
    print("Best regularization coeff is " + str(best_reg_coeff))

    plt.bar(range(len(reg_coeffs)), bars)
    plt.ylabel('Percent of correct classifications')
    plt.xlabel('Regularization coefficient')
    plt.xticks(range(len(reg_coeffs)), [str(round(c, 5)) for c in reg_coeffs])

    plt.show()