import scipy.io
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import re
from nltk.stem import PorterStemmer
import urlextract
import sys


# 17
class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.url_extractor = urlextract.URLExtract()
        self.tag_regex = re.compile(r"<[^>]*>")
        self.email_regex = re.compile(r"[^\s]+@[^\s]+")
        self.number_regex = re.compile(r'\d+(?:\.\d*(?:[eE]\d+))?')
        self.dollar_regex = re.compile(r"[$]+")
        self.spaces_regex = re.compile(r"\s+")
        self.special_chars = [
            "<", "[", "^", ">", "+", "?", "!", "'", ".", ",", ":",
            "*", "%", "#", "_", "=", "-", "&", '/', '\\', '(', ')'
        ]

    def preprocess_text(self, text):
        text = text.lower()
        text = self.remove_html_tags(text)
        text = self.replace_urls(text)
        text = self.replace_emails(text)
        text = self.replace_numbers(text)
        text = self.replace_dollar_signs(text)
        text = self.stem_words(text)
        text = self.remove_special_characters(text)
        text = self.spaces_regex.sub(' ', text)
        return text.strip()

    def remove_html_tags(self, text):
        text = self.tag_regex.sub(" ", text).split(" ")
        text = filter(len, text)
        text = ' '.join(text)
        return text

    def replace_urls(self, text):
        urls = list(set(self.url_extractor.find_urls(text)))
        urls.sort(key=lambda u: len(u), reverse=True)
        for url in urls:
            text = text.replace(url, " httpaddr ")
        return text

    def replace_emails(self, text):
        return self.email_regex.sub(" emailaddr ", text)

    def replace_numbers(self, text):
        return self.number_regex.sub(" number ", text)

    def replace_dollar_signs(self, text):
        return self.dollar_regex.sub(" dollar ", text)

    def remove_special_characters(self, text):
        for char in self.special_chars:
            text = text.replace(str(char), "")
        text = text.replace("\n", " ")
        return text

    def stem_words(self, text):
        text = [self.stemmer.stem(token) for token in text.split(" ")]
        text = " ".join(text)
        return text


# 5
def gauss_kernel(x, l, gamma):
    matrix = np.power(x-l, 2)
    return np.exp(-np.sum(matrix, axis=1) * gamma)


def apply_kernel(x, gamma):
    return apply_kernel_with_points(x, x, gamma)


def apply_kernel_with_points(x, l, gamma):
    return np.array([gauss_kernel(x_i, l, gamma) for x_i in x])


def plot_data(x, y):
    accepted_idx = y == 1
    plt.scatter(x[accepted_idx].T[0], x[accepted_idx].T[1], color='green', label="class 1")
    plt.scatter(x[~accepted_idx].T[0], x[~accepted_idx].T[1], color='blue', label="class 2")
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()


def plot_divider(classifier, x_range, y_range, kernel=None):
    xx = np.linspace(*x_range, 20)
    yy = np.linspace(*y_range, 20)
    yy, xx = np.meshgrid(yy, xx)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    if kernel:
        xy = kernel(xy)
    z = classifier.decision_function(xy).reshape(xx.shape)
    plt.contour(xx, yy, z, colors='red', levels=[0], linestyles=['-'])


def find_better_params(x, y, x_val, y_val, c_values, sigma_values, kernel='rbf'):
    c_optimal = 0
    sigma_optimal = 0
    result_score = 0
    for c in c_values:
        for s in sigma_values:
            gamma = np.power(s, -2.)
            classifier = svm.SVC(C=c, gamma=gamma, kernel=kernel)
            classifier.fit(x, y)
            score = classifier.score(x_val, y_val)
            print("C: ", c, "sigma: ", s, "score: ", score)
            if score > result_score:
                result_score = score
                c_optimal = c
                sigma_optimal = s

    return c_optimal, sigma_optimal

# 19 20
def convert_text_to_feature_vector(text, mapping):
    preprocessor = TextPreprocessor()
    text = preprocessor.preprocess_text(text)
    words = text.split(' ')
    feature_vector = np.zeros(len(mapping))
    for word in words:
        if word in mapping:
            feature_vector[mapping[word] - 1] = 1
    return feature_vector


def check_is_spam(classifier, mapping, filename):
    mail = open(filename).read()
    feature_vector = convert_text_to_feature_vector(mail, mapping)
    prediction = classifier.predict([feature_vector])
    if prediction[0] == 0:
        print(filename + " is not a spam")
    else:
        print(filename + " is a spam")


def test_spam_detection(train_file, test_file, vocab_file):
    mat = scipy.io.loadmat(train_file)
    x = mat['X']
    y = mat['y'].flatten()

    mat = scipy.io.loadmat(test_file)
    x_test = mat['Xtest']
    y_test = mat['ytest'].flatten()

    # C = 30, gamma=0.001 ~ 99.1
    clf = svm.SVC(C=30, kernel='rbf', gamma=0.001)
    clf.fit(x, y)
    print("Training Accuracy (gaussian):", (clf.score(x, y)) * 100, "%")
    print("Test Accuracy (gaussian):", (clf.score(x_test, y_test)) * 100, "%")

    word_to_code = {}
    code_to_word = ["unknown"]
    with open(vocab_file) as f:
        for line in f:
            code, word = line.split('\t')
            word = word.strip()
            code = int(code)
            code_to_word.append(word)
            word_to_code[word] = code

    check_is_spam(clf, word_to_code, "data/emailSample1.txt")
    check_is_spam(clf, word_to_code, "data/emailSample2.txt")
    check_is_spam(clf, word_to_code, "data/spamSample1.txt")
    check_is_spam(clf, word_to_code, "data/spamSample2.txt")



if __name__ == "__main__":
    # 1
    mat = scipy.io.loadmat('data/ex5data1.mat')
    x = mat['X']
    y = mat['y'].flatten()

    # 2
    plot_data(x, y)
    plt.show()

    # 3 4
    for c in [1, 100]:
        plot_data(x, y)
        clf = svm.SVC(kernel='linear', C=c)
        clf.fit(x, y)
        plot_divider(clf, [0, 5], [0, 5])
        plt.show()

    # 6
    mat = scipy.io.loadmat('data/ex5data2.mat')
    x = mat['X']
    y = mat['y'].flatten()

    # 7
    sigma = 1
    gamma = 1 / (2 * np.power(sigma, 2))

    f = apply_kernel(x, gamma)

    # 8
    clf = svm.SVC(kernel='linear')
    clf.fit(f, y)

    # 9
    plot_data(x, y)
    plot_divider(clf, [0, 1], [0, 1], lambda v: apply_kernel_with_points(v, x, gamma))
    plt.show()

    # 10
    mat = scipy.io.loadmat('data/ex5data3.mat')
    x = mat['X']
    y = mat['y'].flatten()
    x_val = mat['Xval']
    y_val = mat['yval'].flatten()

    # 11
    sigma_values = [0.1, 0.3, 0.5, 1, 3, 10, 30, 50, 100]
    c_values = [0.01, 0.03, 0.1, 0.3, 0.5, 1, 3, 10, 30, 50, 100]
    c, s = find_better_params(x, y, x_val, y_val, c_values, sigma_values)
    print("Best C: ", c, ", best sigma: ", s)

    # 12
    clf = svm.SVC(kernel='rbf', C=c, gamma=np.power(s, -2.))
    clf.fit(x, y)
    plot_divider(clf, [-0.6, 0.3], [-0.7, 0.6])
    plot_data(x, y)
    plt.show()

    # 13
    mat = scipy.io.loadmat('data/spamTrain.mat')
    x = mat['X']
    y = mat['y'].flatten()

    # 15
    mat = scipy.io.loadmat('data/spamTest.mat')
    x_test = mat['Xtest']
    y_test = mat['ytest'].flatten()

    # 16
    # long running. c = 0.03, sigma = 0.1. score 0.99
    # c, s = find_better_params(x, y, x_test, y_test, c_values, sigma_values, kernel='linear')
    # print("Best C: ", c, ", best sigma: ", s)
    c = 0.03
    sigma = 0.1

    clf = svm.SVC(C=c, kernel='linear', gamma=np.power(sigma, -2))
    clf.fit(x, y)
    print("Training Accuracy:", (clf.score(x, y)) * 100, "%")
    print("Test Accuracy:", (clf.score(x_test, y_test)) * 100, "%")

    # 18
    word_to_code = {}
    code_to_word = ["unknown"]
    with open("data/vocab.txt") as f:
        for line in f:
            code, word = line.split('\t')
            word = word.strip()
            code = int(code)
            code_to_word.append(word)
            word_to_code[word] = code

    check_is_spam(clf, word_to_code, "data/emailSample1.txt")
    check_is_spam(clf, word_to_code, "data/emailSample2.txt")
    check_is_spam(clf, word_to_code, "data/spamSample1.txt")
    check_is_spam(clf, word_to_code, "data/spamSample2.txt")

    # 25
    mat = scipy.io.loadmat('data/customTrain.mat')
    x_custom = mat['X']
    y_custom = mat['y'].flatten()

    mat = scipy.io.loadmat('data/customTest.mat')
    x_custom_test = mat['Xtest']
    y_custom_test = mat['ytest'].flatten()

    clf.fit(x_custom, y_custom)
    print("Accuracy on custom training set:", (clf.score(x_custom, y_custom)) * 100, "%")
    print("Accuracy on custom test set:", (clf.score(x_custom_test, y_custom_test)) * 100, "%")

    word_to_code = {}
    code_to_word = ["unknown"]
    with open("data/vocab2.txt") as f:
        for line in f:
            code, word = line.split('\t')
            word = word.strip()
            code = int(code)
            code_to_word.append(word)
            word_to_code[word] = code

    check_is_spam(clf, word_to_code, "data/emailSample1.txt")
    check_is_spam(clf, word_to_code, "data/emailSample2.txt")
    check_is_spam(clf, word_to_code, "data/spamSample1.txt")
    check_is_spam(clf, word_to_code, "data/spamSample2.txt")