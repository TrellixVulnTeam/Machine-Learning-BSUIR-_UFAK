from scipy.io import loadmat
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt


class CollaborativeFiltering:
    eps = 0.000001
    learning_coeff = 0.0005

    def __init__(self, rating_matrix, ratings, n, reg_coeff=0):
        self.n = n
        self.n_movies, self.n_users = rating_matrix.shape
        self.theta = np.random.rand(self.n_users, self.n) * (2 * self.eps) - self.eps
        self.x = np.random.rand(self.n_movies, self.n) * (2 * self.eps) - self.eps
        self.r = rating_matrix
        self.rt = self.r.T
        self.y = ratings
        self.yt = self.y.T
        self.rate_idx = np.where(self.r == 1)
        self.reg_coeff = reg_coeff
        self.learning_history = []

    # 3
    def cost(self):
        ratings_vector = self.y[self.rate_idx]
        theta = self.theta[self.rate_idx[1]]
        x = self.x[self.rate_idx[0]]
        predictions = np.sum(theta * x, axis=1)
        error = ratings_vector - predictions
        cost = np.dot(error.T, error)

        reg_x = self.reg_coeff * np.sum(self.x ** 2)
        reg_theta = self.reg_coeff * np.sum(self.theta ** 2)
        return (cost + reg_x + reg_theta) * 0.5

    # 4
    def compute_gradient(self):
        x_grad = np.zeros(self.x.shape)
        theta_grad = np.zeros(self.theta.shape)

        for movie_id in range(self.n_movies):
            movie_rating_row = self.r[movie_id]
            x_i = self.x[movie_id]
            idx = np.where(movie_rating_row == 1)
            theta = self.theta[idx]
            theta_t = theta.T
            rates = self.y[movie_id][idx]
            predictions = np.sum(theta * x_i, axis=1)
            error = predictions - rates
            for k in range(self.n):
                t_k = theta_t[k]
                reg = self.reg_coeff * self.x[movie_id][k]
                x_grad[movie_id][k] = np.sum(np.multiply(error, t_k)) + reg

        for user_id in range(self.n_users):
            user_rating_row = self.rt[user_id]
            theta_j = self.theta[user_id]
            idx = np.where(user_rating_row == 1)
            x = self.x[idx]
            xt = x.T
            rates = self.yt[user_id][idx]
            predictions = np.sum(theta_j * x, axis=1)
            error = predictions - rates
            for k in range(self.n):
                x_k = xt[k]
                reg = self.reg_coeff * self.theta[user_id][k]
                theta_grad[user_id][k] = np.sum(np.multiply(error, x_k)) + reg

        return x_grad, theta_grad

    def do_gradient_descend(self):
        x_grad, theta_grad = self.compute_gradient()
        self.x -= self.learning_coeff * x_grad
        self.theta -= self.learning_coeff * theta_grad

    def recommend(self, user_id, top=10):
        user_theta = self.theta[user_id]
        predictions = np.sum(user_theta * self.x, axis=1)
        movie_predictions = ((movie_id, prediction) for movie_id, prediction in enumerate(predictions))
        movie_predictions = sorted(movie_predictions, key=lambda p: p[1], reverse=True)
        return movie_predictions[:top]

    def train(self, n_iterations):
        for i in range(n_iterations):
            collab.do_gradient_descend()
            cost = collab.cost()
            self.learning_history.append(cost)
            print(str(i) + ": " + str(collab.cost()))


def add_custom_rates(r, y, movie_ids, rate):
    new_r = np.zeros((r.shape[0], r.shape[1] + 1))
    new_r[:, :-1] = r
    new_y = np.zeros((r.shape[0], r.shape[1] + 1))
    new_y[:, :-1] = y

    new_r[:, -1][movie_ids] = 1
    new_y[:, -1][movie_ids] = rate
    return new_r, new_y


def load_movies():
    movies = []
    with open("movie_ids.txt") as f:
        for line in f:
            movie_id, *rest = line.split(' ')
            movies.append(' '.join(rest).strip())
    return movies


if __name__ == "__main__":
    # 1
    mat = loadmat("ex9_movies")
    rating_matrix = mat['R']
    y = mat['Y']
    movies = load_movies()

    # 8
    movie_idx = [0, 16, 41, 55, 68, 134, 176, 404, 1158]
    rating_matrix, y = add_custom_rates(rating_matrix, y, movie_idx, 5)
    new_user_id = rating_matrix.shape[1] - 1
    for movie_id in movie_idx:
        movie = movies[movie_id]
        print("User " + str(new_user_id) + " rating " + movie + " with 5.")

    # 2
    n = 4

    # 5
    collab = CollaborativeFiltering(rating_matrix, y, n)
    collab.train(500)
    plt.plot(collab.learning_history)
    plt.show()

    # 9
    recommendations = collab.recommend(new_user_id, 10)
    for rec in recommendations:
        movie = movies[rec[0]]
        predicted_rate = rec[1]
        print("User " + str(new_user_id) + " will rate movie " + movie + " with " + str(predicted_rate))

    # 10
    u, s, vt = svds(y.astype(float), k=n)
    collab = CollaborativeFiltering(rating_matrix, y, n)
    collab.theta = vt.T
    collab.x = u

    # 8
    recommendations = collab.recommend(new_user_id, 10)
    for rec in recommendations:
        movie = movies[rec[0]]
        predicted_rate = rec[1]
        print("User " + str(new_user_id) + " will rate movie " + movie + " with " + str(predicted_rate))
