from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import imageio


class HierarchicalCluster:
    def __init__(self, sub_clusters, centroid=None, index=None):
        self.centroid = centroid
        self.clusters = sub_clusters
        self.index = index
        if self.clusters:
            cluster_centroids = np.array([c.centroid for c in self.clusters])
            self.centroid = np.mean(cluster_centroids, axis=0)

    def unravel(self, level, container: list):
        if level > 1:
            for sub_cluster in self.clusters:
                sub_cluster.unravel(level - 1, container)
        elif level == 1:
            container.append(self)

    def get_all_children(self, container=None):
        if container == None:
            container = []
            for cluster in self.clusters:
                cluster.get_all_children(container)
            return container
        elif self.clusters != None:
            for cluster in self.clusters:
                cluster.get_all_children(container)
        else:
            container.append(self.centroid)


class HierarchicalClusterizer:
    def __init__(self, x):
        self.x = x
        self.n = len(x)
        self.distance_matrix = np.zeros((self.n, self.n))

    def clusterize(self):
        clusters = [HierarchicalCluster(None, point, i) for i, point in enumerate(self.x)]
        self.compute_distance_matrix(clusters)
        while len(clusters) > 1:
            print(len(clusters))
            c_1, c_2 = self.find_two_closest()
            parent_cluster = HierarchicalCluster([clusters[c_1], clusters[c_2]])
            del clusters[c_2]
            del clusters[c_1]
            clusters.append(parent_cluster)
            self.update_distance_matrix(c_1, c_2, clusters)
        return clusters[0]

    def compute_distance_matrix(self, clusters):
        for i in range(len(clusters)):
            self.distance_matrix[i][i] = 10000000000000000
            for j in range(i + 1, len(clusters)):
                dist = np.linalg.norm(clusters[i].centroid - clusters[j].centroid)
                self.distance_matrix[i][j] = dist
                self.distance_matrix[j][i] = dist

    def update_distance_matrix(self, i, j, clusters):
        self.distance_matrix = np.delete(self.distance_matrix, j, axis=0)
        self.distance_matrix = np.delete(self.distance_matrix, i, axis=0)
        self.distance_matrix = np.delete(self.distance_matrix, j, axis=1)
        self.distance_matrix = np.delete(self.distance_matrix, i, axis=1)
        n = len(self.distance_matrix)
        b = np.zeros((n + 1, n + 1))
        b[:-1, :-1] = self.distance_matrix
        b[n][n] = 10000000000000000
        for j in range(n):
            b[n][j] = b[j][n] = np.linalg.norm(clusters[n].centroid - clusters[j].centroid)
        self.distance_matrix = b

    def find_two_closest(self):
        i_min, j_min = np.where(self.distance_matrix == np.min(self.distance_matrix))[0]
        return sorted((i_min, j_min))


class KMeansClusterizer:
    def __init__(self, k):
        self.k = k
        self.centroid_history = []

    # 5
    def clusterize(self, x):
        n = len(x)
        number_of_iterations = 10
        clusters = np.zeros(n)
        centroids = self.initialize_centroids(x)
        self.centroid_history.append(centroids)
        for _ in range(number_of_iterations):
            for i, vector in enumerate(x):
                k = self.detect_cluster(vector, centroids)
                clusters[i] = k
            centroids = self.recalculate_centroids(clusters, x)
            self.centroid_history.append(centroids)

        return clusters, centroids

    # 2
    def initialize_centroids(self, x):
        idx = np.random.randint(0, len(x), self.k)
        centroids = x[idx, :]
        return centroids

    # 3
    @staticmethod
    def detect_cluster(point, centroids):
        k = np.argmin([np.linalg.norm(point - centroid) for centroid in centroids])
        return k

    # 4
    def recalculate_centroids(self, clusters, x):
        new_centroids = []
        for cluster in range(self.k):
            cluster_point_indexes = np.argwhere(clusters == cluster)
            if len(cluster_point_indexes) == 0:
                continue
            cluster_points = x[cluster_point_indexes]
            centroid = np.mean(cluster_points, axis=0)[0]
            new_centroids.append(centroid)
        return np.array(new_centroids)

def replace_colors(img_data, clusters, centroid_colors):
    for cluster in range(len(centroid_colors)):
        cluster_point_indexes = np.argwhere(clusters == cluster)
        img_data[cluster_point_indexes] = centroid_colors[cluster]


if __name__ == "__main__":
    # 1
    mat = loadmat("ex6data1")
    x = mat['X']

    # 6
    k = 3
    clusterizer = KMeansClusterizer(k)
    clusters, centroids = clusterizer.clusterize(x)

    for cluster in range(k):
        cluster_point_indexes = np.argwhere(clusters == cluster)
        cluster_points = x[cluster_point_indexes]
        plt.scatter(cluster_points.T[0], cluster_points.T[1])

    centroids_history = np.array(clusterizer.centroid_history)
    for i in range(k):
        history = centroids_history[:, i]
        ax = plt.plot(history.T[0], history.T[1], linestyle='--')
        plt.plot(history[-1][0], history[-1][1], 'b+', color=ax[0]._color, mew=4, ms=10)

    plt.show()

    # 7
    mat = loadmat("bird_small")
    data = mat['A']
    image = data.reshape(128, 128, 3)
    plt.imshow(image, interpolation='nearest')
    plt.show()

    # 8
    k = 16
    data = data.reshape(128 * 128, 3)
    clusterizer = KMeansClusterizer(k)
    clusters, centroid_colors = clusterizer.clusterize(data)
    replace_colors(data, clusters, centroid_colors)

    image = data.reshape(128, 128, 3)
    plt.imshow(image, interpolation='nearest')
    plt.show()

    imageio.imwrite("bird_16.png", image, "png")
    np.savetxt('output.txt', clusters)

    # 10
    image = imageio.imread('myc.png').reshape(128 * 128, 3)
    imageio.imwrite("myc.png", image.reshape(128, 128, 3), "png")
    clusters, centroid_colors = clusterizer.clusterize(image)
    replace_colors(image, clusters, centroid_colors)

    image = image.reshape(128, 128, 3)
    plt.imshow(image, interpolation='nearest')
    plt.show()

    imageio.imwrite("myc_16.png", image, "png")
