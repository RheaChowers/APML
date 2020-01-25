import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def digits_example():
    '''
    Example code to show you how to load the MNIST data and plot it.
    '''

    # load the MNIST data:
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target

    # plot examples:
    plt.gray()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        plt.imshow(np.reshape(data[i, :], (8, 8)))
        plt.title("Digit " + str(labels[i]))
    plt.show()


def swiss_roll_example():
    '''
    Example code to show you how to load the swiss roll data and plot it.
    '''

    # load the dataset:
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)

    # plot the data:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()


def faces_example(path):
    '''
    Example code to show you how to load the faces data.
    '''

    with open(path, 'rb') as f:
        X = pickle.load(f)

    num_images, num_pixels = np.shape(X)
    d = int(num_pixels**0.5)
    print("The number of images in the data set is " + str(num_images))
    print("The image size is " + str(d) + " by " + str(d))

    # plot some examples of faces:
    plt.gray()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(np.reshape(X[i, :], (d, d)))
    plt.show()


def plot_with_images(X, images, title, image_num=25):
    '''
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param num_to_plot: Number of images to plot along with the scatter plot.
    :return: the figure object.
    '''

    n, pixels = np.shape(images)
    img_size = int(pixels**0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # get the size of the embedded images for plotting:
    x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
    y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08

    # draw random images and plot them in their relevant place:
    for i in range(image_num):
        img_num = np.random.choice(n)
        x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
        x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
        img = images[img_num, :].reshape(img_size, img_size)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000,
                  extent=(x0, x1, y0, y1))

    # draw the scatter plot of the embedded data points:
    ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)

    return fig


def euclid(X, Y):
    """
    return the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """
    # (x-y)^2=x^2+y^2-2xy
    X_squared_sum = np.sum(np.square(X), axis=1)  # ||X||^2
    Y_squared_sum = np.sum(np.square(Y), axis=1)  # ||Y||^2
    dists = -2 * np.dot(X, Y.T) + X_squared_sum[:, np.newaxis] + Y_squared_sum
    dists = np.clip(dists, 0, None)
    dists = np.sqrt(dists)
    return dists


def MDS(X, d):
    '''
    Given a NxN pairwise distance matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.

    :param X: NxN distance matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix.
    '''
    n, _ = X.shape
    # D = euclid(X, X)
    H = np.eye(n) - np.full((n, n), 1/n)
    S = -0.5 * H * X * H
    original_eigvals, eigvecs = np.linalg.eigh(S)
    # choose d largest values
    indices = np.argsort(original_eigvals)
    eigvals = original_eigvals[indices][-d:]
    eigvecs = eigvecs[:, indices][:, -d:]
    diag_eigvals = np.diag(np.sqrt(eigvals))
    return original_eigvals, np.matmul(eigvecs, diag_eigvals)


def knn(X, k):
    """
    calculate the m nearest neighbors similarity of the given distance matrix.
    :param X: A Nxd data matrix.
    :param m: The number of nearest neighbors.
    :return: Nx(k+1) matrix of indices of nearest neighbors
    """
    distances = euclid(X, X)
    neighbs = np.argpartition(distances, k, axis=1)[:, :k+1]  # since the point itself will be included
    return neighbs


def LLE(X, d, k):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the LLE algorithm.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param k: the number of neighbors for the weight extraction.
    :return: Nxd reduced data matrix.
    '''
    samples, dim = X.shape
    KNN = knn(X, k)
    W = np.zeros((samples, samples))  # N x N weights matrix
    for i in range(samples):
        # get the indices of the k+1 neighbors of the i'th point (k+1 since the point is included)    
        neighbors = KNN[i, :]
        neighbors = neighbors[np.where(neighbors != i)]  # we don't want the data point itself
        Z = X[neighbors, :] - X[i, :]  # get the data of the neighbors and subtract the element itself
        gram = np.matmul(Z, Z.T)  # G_ij = z_i^Tz_j since Z is composed of row vectors
        gram_pinv = np.linalg.pinv(gram)
        weights = gram_pinv.sum(axis=1)  # multiplying by 1's vec is like summing rows
        W[i, neighbors] = weights / np.sum(weights)  # normalize
    W_laplacian = np.eye(samples) - W
    MTM = np.matmul(W_laplacian.T, W_laplacian)
    eigvals, eigvecs = np.linalg.eigh(MTM)
    return eigvecs[:, 1: d+1]  # disregard the first eigenvector


def gaussian_kernel(X, sigma):
    """
    calculate the gaussian kernel similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """

    return np.exp(np.divide(np.square(X), -2*sigma*sigma))


def DiffusionMap(X, d, sigma, t):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the Diffusion Map algorithm. The k parameter allows restricting the
    kernel matrix to only the k nearest neighbor of each data point.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param sigma: the sigma of the gaussian for the kernel matrix transformation.
    :param t: the scale of the diffusion (amount of time steps).
    :return: Nxd reduced data matrix.
    '''
    distances = euclid(X, X)
    K = gaussian_kernel(distances, sigma)
    D_norm = np.diag(1/np.sum(K, axis=1))
    A = np.matmul(D_norm, K)
    eigvals, eigvecs = np.linalg.eigh(A)
    # choose largest d excluding the dirst
    eigvals = eigvals[-(d+1):-1]
    eigvecs = eigvecs[:, -(d+1):-1]
    # the the t'th power
    eigvals = np.power(eigvals, t)
    return np.matmul(eigvecs, np.diag(eigvals))


def scree_plot(samples=500, dim=2, embed=64, num_noises=6, max_noise=0.5):
    """
    Shows scree plots for different noises for the MDA and PCA methods
    samples - the number of points we are embedding
    dim - the intrinsic dimension
    embed - the embedded dimension
    num_noise - for how many different noises we create plots
    max_noise - the maximal noise
    """
    rand = np.random.rand(samples, dim)  # random 2d dataset
    data = np.hstack((rand, np.zeros((samples, embed-dim))))  # pad with zeros
    embedded_data = np.matmul(data, get_rotation(embed))  # rotate the data
    noises = np.linspace(0, max_noise, num_noises)  # the standard deviation of the noises
    plot_eigvalues(noises, MDS, "MDS Scree Plots for Different Noises", embedded_data, dim, samples, embed, num_noises)
    plot_eigvalues(noises, None, "PCA Scree Plots for Different Noises", embedded_data, dim, samples, embed, num_noises)


def plot_eigvalues(noises, func, title, embedded_data, dim, samples, embed, num_noises):
    """
    Plots eigenvalues
    """
    plt.figure()
    plt.suptitle(title)
    for i, noise in enumerate(noises):  # varying noises
        noisey_embedded = embedded_data + np.random.normal(scale=noise, size=(samples, embed))
        if func == MDS:
            eigvals, _ = MDS(euclid(noisey_embedded, noisey_embedded), dim)
        else:
            eigvals = get_PCA_eigvals(noisey_embedded)
        eigvals = eigvals[::-1]  # sort from largest to smallest
        eigvals = eigvals[:10]
        plt.subplot(num_noises/2, 2, i+1)
        plt.title("sigma="+str(noise))
        plt.plot([i+1 for i in range(len(eigvals))], eigvals, '.')
    plt.show()


def get_PCA_eigvals(data):
    """
    returns the eigenvalues of the covariance matrix of a dataset
    """
    n, _ = data.shape
    cov = np.cov(data)
    eigvals, _ = np.linalg.eigh(cov)
    return eigvals


def get_rotation(dimension):
    """
    returns a random rotation matrix in the given dimension
    """
    rand_gaus = np.random.normal(size=(dimension, dimension))  # random rotation
    rotation, _ = np.linalg.qr(rand_gaus)  # embed X embed
    return rotation


def test_LLE(data, d=2, min_k=5, max_k=6, color=None, title=""):
    """
    Tests the LLE function on a given data for values of k between
    min_k and max_k, and plots the result
    """
    i = 1
    plt.figure()
    plt.suptitle(title)
    for k in range(min_k, max_k):
        reduced = LLE(data, d, k)
        ax = plt.subplot(int((max_k-min_k)/2), 2, i)
        if k < max_k - 2:
            ax.xaxis.set_visible(False)
        plt.title('k='+str(k))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=color, cmap=plt.cm.Spectral)
        i += 1
    plt.show()


def test_diffusion(data, d=2, color=None, title=""):
    """
    Tests the diffusion map on a given data set
    """
    i = 1
    plt.figure()
    plt.suptitle(title)
    for sigma in [1.5, 2, 2.5, 3]:
        for t in [0.5, 0.75, 1, 2]:
            reduced = DiffusionMap(data, d, sigma, t)
            ax = plt.subplot(4, 4, i)
            ax.xaxis.set_visible(False)
            plt.title("t="+str(t)+", sigma="+str(sigma))
            plt.scatter(reduced[:, 0], reduced[:, 1], c=color, cmap=plt.cm.Spectral)
            ax.set(xlim=(np.min(reduced[:, 0]) - 0.2*abs(np.min(reduced[:, 0])), np.max(reduced[:, 0]) + 0.2*abs(np.max(reduced[:, 0]))),
                   ylim=(np.min(reduced[:, 1]) - 0.2*abs(np.min(reduced[:, 1])), np.max(reduced[:, 1]) + 0.2*abs(np.max(reduced[:, 1]))))
            i += 1
    plt.show()


def test_MDS(data, color=None, title="", d=2):
    """
    Tests the MDS function on a given dataset and plots the result
    """
    _, reduced = MDS(euclid(data, data), d)
    plt.figure()
    plt.title(title)
    ax = plt.subplot(1, 1, 1)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=color, cmap=plt.cm.Spectral)
    ax.set(xlim=(np.min(reduced[:, 0]) - 0.2*abs(np.min(reduced[:, 0])), np.max(reduced[:, 0]) + 0.2*abs(np.max(reduced[:, 0]))),
           ylim=(np.min(reduced[:, 1]) - 0.2*abs(np.min(reduced[:, 1])), np.max(reduced[:, 1]) + 0.2*abs(np.max(reduced[:, 1]))))
    plt.show()


def plot(data, title="", color=None):
    """
    plots a simple 2d figure with given data, title and colors
    """
    plt.figure()
    plt.suptitle(title)
    plt.scatter(data[:, 0], data[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.show()


def test_faces(path="faces.pickle"):
    """
    Tests the different methods with the face dataset
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # Diffusion Map
    for sigma in np.linspace(8, 20, 5):
        for t in [1, 5, 10, 15]:
            reduced = DiffusionMap(data, 2, sigma, t)
            title = "Diffusion Map on Faces, t="+str(t)+", sigma="+str(sigma)
            fig = plot_with_images(reduced, data, title, image_num=25)
            plt.show()
    # LLE
    for k in range(10, 16):
        reduced = LLE(data, 2, k)
        title = "LLE on Faces, k="+str(k)
        fig = plot_with_images(reduced, data, title, image_num=25)
        plt.show()

    # MDS
    reduced = MDS(euclid(data, data), 2)
    title = "MDS on Faces"
    fig = plot_with_images(reduced, data, title, image_num=25)
    plt.show()


if __name__ == '__main__':
    # Un-comment any test you want to run
    swiss_roll, color = X, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)
    digits = datasets.load_digits()
    mnist = digits.data / 255.
    mnist_labels = digits.target

    ################## Swiss TESTS ##################
    # test_LLE(swiss_roll, d=2, min_k=9, max_k=17, color=color, title="LLE on Swiss Roll")
    # test_diffusion(swiss_roll, color=color, title="Diffusion Map on Swiss Roll")
    # test_MDS(swiss_roll, color=color, title="MDS on Swiss Roll")

    ################## MNIST TESTS ##################
    # test_LLE(mnist, d=2, min_k=9, max_k=17, color=mnist_labels, title="LLE on MNIST")
    test_diffusion(mnist, color=mnist_labels, title="Diffusion Map on MNIST")
    # test_MDS(mnist, color=mnist_labels, title="MDS on MNIST")

    ################## scree plots ##################
    # scree_plot()

    ################## Faces Test ##################
    # test_faces()
