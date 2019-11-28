from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from sklearn.metrics import f1_score


def plot_distribution(figure, values, mean, sigma, n_bins=45):
    figure.hist(values, bins=n_bins, density=True, alpha=0.6, color='g')
    x_min = np.min(values)
    x_max = np.max(values)
    x = np.linspace(x_min, x_max, 100)
    figure.plot(x, st.norm.pdf(x, mean, sigma))


class AnomalyDetector:
    def __init__(self, threshold=0.005):
        self.threshold = threshold
        self.mu = []
        self.sigma = []

    def fit(self, data):
        for values in data.T:
            mu, var = self.analyze_distribution(values)
            self.mu.append(mu)
            self.sigma.append(var)

    def is_anomaly(self, example):
        p = 1
        for x, mu, s in zip(example, self.mu, self.sigma):
            p *= st.norm.pdf(x, mu, s)
        return p < self.threshold

    @staticmethod
    def analyze_distribution(values):
        mean, variance = st.norm.fit(values)
        sigma = np.sqrt(variance)
        return mean, sigma


def find_optimal_threshold(detector: AnomalyDetector, x_val, y_val, thresholds):
    best_score = 0.0
    best_threshold = 1
    for threshold in thresholds:
        detector.threshold = threshold
        y_predicted = []
        for point in x_val:
            is_anomaly = detector.is_anomaly(point)
            if is_anomaly:
                y_predicted.append(1)
            else:
                y_predicted.append(0)
        f1 = f1_score(y_val, y_predicted)
        print("Threshold is " + str(threshold) + " gives f1 score as " + str(f1))

        if f1 > best_score:
            best_score = f1
            best_threshold = threshold
    return best_threshold, best_score


if __name__ == "__main__":
    # 1
    mat = loadmat("ex8data1")
    x = mat['X']
    x_val, y_val = mat['Xval'], mat['yval'].flatten()

    # 2
    x1 = x.T[0]
    x2 = x.T[1]
    plt.scatter(x1, x2)
    plt.show()

    fig, axs = plt.subplots(1, 2, tight_layout=True)

    detector = AnomalyDetector()
    detector.fit(x)

    # 3 4
    for values, figure, mu, sigma in zip(x.T, axs, detector.mu, detector.sigma):
        print("mu = " + str(mu) + ". sigma = " + str(sigma))
        plot_distribution(figure, values, mu, sigma)

    plt.show()

    # 5
    multi = st.multivariate_normal(detector.mu, np.diag(detector.sigma))

    x1_min = np.min(x1)
    x1_max = np.max(x2)
    x2_min = np.min(x2)
    x2_max = np.max(x2)

    xx, yy = np.mgrid[x1_min:x1_max:.01, x2_min:x2_max:.01]
    pos = np.dstack((xx, yy))
    plt.contour(xx, yy, multi.pdf(pos), colors='red')

    plt.scatter(x1, x2)
    plt.show()

    # 6
    threshold, score = find_optimal_threshold(detector, x_val, y_val, np.arange(0.00001, 0.001, 0.00001))
    print("Best threshold is " + str(threshold) + ". It gives f1 score as " + str(score))
    detector.threshold = threshold

    # 7
    non_anomalies = []
    anomalies = []
    for point in x:
        is_anomaly = detector.is_anomaly(point)
        if is_anomaly:
            anomalies.append(point)
        else:
            non_anomalies.append(point)
    anomalies = np.array(anomalies)
    non_anomalies = np.array(non_anomalies)

    plt.scatter(anomalies.T[0], anomalies.T[1])
    plt.scatter(non_anomalies.T[0], non_anomalies.T[1])

    plt.show()

    # 8
    mat = loadmat("ex8data2")
    x = mat['X']
    x_val, y_val = mat['Xval'], mat['yval'].flatten()

    detector = AnomalyDetector()
    detector.fit(x)

    # 9
    fig, axs = plt.subplots(1, 11)

    for values, figure, mu, sigma in zip(x.T, axs, detector.mu, detector.sigma):
        plot_distribution(figure, values, mu, sigma)
        print("mu = " + str(mu) + ". sigma = " + str(sigma))
    plt.show()

    # 11
    threshold, score = find_optimal_threshold(detector, x_val, y_val, [1e-67, 1e-44, 1.6e-44, 1.65e-44])
    print("Best threshold is " + str(threshold) + ". It gives f1 score as " + str(score))
    detector.threshold = threshold

    # 12
    non_anomalies = 0
    anomalies = 0
    for point in x:
        is_anomaly = detector.is_anomaly(point)
        if is_anomaly:
            anomalies += 1
        else:
            non_anomalies += 1
    print(str(anomalies) + " anomalies were detected out of " + str(anomalies + non_anomalies) + " train samples")

    non_anomalies = 0
    anomalies = 0
    for point in x_val:
        is_anomaly = detector.is_anomaly(point)
        if is_anomaly:
            anomalies += 1
        else:
            non_anomalies += 1
    print(str(anomalies) + " anomalies were detected out of " + str(anomalies + non_anomalies) + " validation samples")
