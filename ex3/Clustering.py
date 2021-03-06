import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import numpy.linalg as linalg
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def generate_synthetic(k=5, sigma=0.5):
    """
    Generates k clusters, each normally distributed around a different point on the unit circle
    """
    X = []
    radius = 3
    centers = np.linspace(0, 2*np.pi, k, False)
    center_xy = (radius * np.stack([np.cos(centers), np.sin(centers)])).T
    for row in center_xy:
        # use the center as a center for a gaussian
        n = np.random.randint(100, 200)  # choose a random number of points
        points_x = np.random.normal(loc=row[0], scale=sigma, size=n)
        points_y = np.random.normal(loc=row[1], scale=sigma, size=n)
        X.append(np.stack([points_x, points_y]))
    X = np.hstack(X).T
    if k == 5:
        plt.figure()
        plt.title("5 Random Normally Distributed Clusters")
        plt.scatter(X[:,0], X[:,1])
        plt.savefig("random_clusters.png")
        plt.close()
    return X


def circles_example():
    """
    an example function for generating and plotting synthetic data.
    """

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.array([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.array([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.array([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.array([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    # circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)
    return np.concatenate((circle1, circle2, circle3, circle4), axis=1).T
    # plt.plot(circles[0,:], circles[1,:], '.k')
    # plt.show()


def apml_pic_example(path='APML_pic.pickle'):
    """
    an example function for loading and plotting the APML_pic data.

    :param path: the path to the APML_pic.pickle file
    """

    with open(path, 'rb') as f:
        apml = pickle.load(f)

    plt.plot(apml[:, 0], apml[:, 1], '.')
    plt.show()


def microarray_exploration(data_path='microarray_data.pickle',
                            genes_path='microarray_genes.pickle',
                            conds_path='microarray_conds.pickle'):
    """
    an example function for loading and plotting the microarray data.
    Specific explanations of the data set are written in comments inside the
    function.

    :param data_path: path to the data file.
    :param genes_path: path to the genes file.
    :param conds_path: path to the conds file.
    """

    # loading the data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(genes_path, 'rb') as f:
        genes = pickle.load(f)
    with open(conds_path, 'rb') as f:
        conds = pickle.load(f)

    # look at a single value of the data matrix:
    i = 128
    j = 63
    print("gene: " + str(genes[i]) + ", cond: " + str(
        conds[0, j]) + ", value: " + str(data[i, j]))

    # make a single bar plot:
    plt.figure()
    plt.bar(np.arange(len(data[i, :])) + 1, data[i, :])
    plt.show()

    # look at two conditions and their correlation:
    plt.figure()
    plt.scatter(data[:, 27], data[:, 29])
    plt.plot([-5,5],[-5,5],'r')
    plt.show()

    # see correlations between conds:
    correlation = np.corrcoef(np.transpose(data))
    plt.figure()
    plt.imshow(correlation, extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()

    # look at the entire data set:
    plt.figure()
    plt.imshow(data, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    plt.colorbar()
    plt.show()


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


def euclidean_centroid(X):
    """
    return the center of mass of data points of X.
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster.
    """
    # the centroid of the cluster is the average of all points in the cluster
    num_points, _ = X.shape
    return X.sum(axis=0) / num_points


def kmeans_pp_init(X, k, metric):
    """
    The initialization function of kmeans++, returning k centroids.
    :param X: The data matrix.
    :param k: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """
    samples, D = X.shape
    if k > samples:
        raise Exception("k needs to be smaller than the number of points in the data")
    centroids = np.zeros((k, D))
    random_row = np.random.randint(samples)
    centroids[0, :] = X[random_row, :]  # initialize the first centroid at random
    for i in range(1, k):
        # define a probability vector s.t. p()
        # compute distances of X and all centroids
        dist = metric(X, centroids[:i, :])  # a samples x i matrix
        dist = np.min(dist, axis=1)  # get the minimal distance from some centroid
        sq_dist = np.square(dist)  # the probability is proportionate to the squared distance
        p = sq_dist / np.sum(sq_dist)
        centroids[i, :] = X[np.random.choice(samples, 1, p=p)]
    return centroids


def kmeans(X, k, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init):
    """
    The K-Means function, clustering the data X into k clusters.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :return: a tuple of (clustering, centroids, statistics)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    """
    samples, D = X.shape
    centroids = init(X, k, metric)  # size of K x D
    clustering = np.zeros((samples))  # the i'th cell will tell us to which cluster the i'th sample belongs
    for i in range(iterations):
        # calculate the distance between the centroids and the data
        distances = metric(centroids, X)  # k, N array
        clustering = np.argmin(distances, axis=0)
        # plot_clusters(X, clustering, centroids)  - if we want to plot
        # update the centroids
        for j in range(k):
            # the j'th centroid is the center of all data points labeled j
            centroids[j, :] = center(X[np.where(clustering == j)])
    return clustering, centroids


def gaussian_kernel(X, sigma):
    """
    calculate the gaussian kernel similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """

    return np.exp(np.divide(np.square(X), -2*sigma*sigma))


def mnn(X, m):
    """
    calculate the m nearest neighbors similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """
    # get the indices of the m nearest neighbors not including the element itself
    indices = np.argsort(X, axis=1)[:, 1:m+1]
    similarity = np.zeros(X.shape)
    for i, cols in enumerate(indices):
        similarity[i, cols] = 1
    similarity = np.logical_and(similarity, similarity.T)
    return similarity


def spectral(X, k, similarity_param=0.5, similarity=gaussian_kernel,
             histogram=False, plot_eigenvalues=False, p=-1):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param similarity: The similarity transformation of the data.
    :param histogram: if to show a histogram of the distances
    :param plot_eigenvalues: wether to plot the eigenvalues graph
    :param p: the percentile used for choosing a similarity param in the gaussian kernel
    :return: clustering, as in the kmeans implementation.
    """
    samples, dim = X.shape
    distances = euclid(X, X)  # get the distances matrix
    if histogram:
        _ = plt.hist(distances[np.triu_indices_from(distances, 1)].flatten(), range=[0,0.1])
        plt.title("Distance Histogram")
        plt.show()
    # if we want we can get a similarity param based on percentiles of distances
    if similarity == gaussian_kernel and p != -1:
        similarity_param = np.percentile(distances[np.triu_indices_from(distances, 1)], p)
    W = similarity(distances, similarity_param)
    # construct a diagonal matrix from the sum of the rows and calculate D^-1/2
    D_norm = np.diag(1/np.sqrt(np.sum(W, axis=1)))
    L_sym = np.identity(samples)-np.matmul(np.matmul(D_norm, W), D_norm)  # the symmetric laplacian
    eigvals, eigvecs = linalg.eigh(L_sym)
    if plot_eigenvalues:  # plot the eigenvalues to see eigengap
        plt.plot([i for i in range(1, 21)], eigvals[np.argsort(eigvals)][:20], '.')
        plt.title("20 First Eigenvalues in Accending Order")
        plt.show()
    # choose k lowest values
    indices = np.argsort(eigvals)
    eigvals = eigvals[indices][:k]
    eigvecs = eigvecs[:, indices][:, :k]
    # normalize eigenvecs and send to k means
    eigvecs = eigvecs / np.sqrt(np.sum(np.square(eigvecs), axis=1))[:, None]
    return kmeans(eigvecs, k, iterations=10)


def cost(X, clustering, centroids, k):
    """
    Returns the value of the cost function of k-means given the
    :param X: data of size samples x dimension
    :param clustering: array of size samples denoting to which cluster each sample belongs
    :param centroids: array of size k x dimension of centroid locations
    :param k: number of clusters
    return - the cost value
    """
    cost = 0
    for i in range(k):
        rows = np.where(clustering == i)[0]  # get the samples clustered in the i'th cluster
        cost += np.sum(euclid(X[rows, :], centroids[i, :].reshape(1,-1)))  # sum the square distances form the center
    return cost


def plot_similarity(X, k, data_name="", similarity_param=0.5, similarity=gaussian_kernel, percentile=1):
    """
    Given data matrix X, a cluster number k, name of data, similarity parameter and 
    similarity function, plots the unsorted and sorted similarity matrix of the data
    """
    np.random.shuffle(X)
    samples, _ = X.shape
    distances = euclid(X, X)  # get the distances matrix
    sim_type = "Mnn"
    if similarity == gaussian_kernel:
        sim_type = "Gaussian"
        if percentile != -1:
            similarity_param = np.percentile(distances[np.triu_indices_from(distances, 1)], percentile)
    clustering, _ = spectral(X, k, similarity_param, similarity)  # run spectral clustering
    # plot_clusters(X, clustering)  # plot the clusters to see they are ok
    # recalculate similarity matrix and sorted similarity matrix and plot them
    W = similarity(distances, similarity_param)
    sorted_data = X[np.argsort(clustering)]
    sorted_distances = euclid(sorted_data, sorted_data)
    W_sorted = similarity(sorted_distances, similarity_param)
    # plot the heatmaps
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(W, cmap="hot")
    axs[0].set_title('Unsorted')
    fig.suptitle("Similarity Matrices, Data={0}, Kernel={1}, Similarity Param={2:.2f}".format(
                data_name, sim_type, similarity_param))
    axs[1].imshow(W_sorted, cmap="hot")
    axs[1].set_title('Sorted')
    plt.show()


def plot_clusters(X, clustering, centroids=-1, title="", show=False, save=False):
    """
    Plots clusters in 2d given the data, a vector signaling in which cluster is each sample
    the centroids of each cluster and the number of clusters
    """
    plt.scatter(X[:, 0], X[:, 1], c=clustering)
    if type(centroids) != int:
        plt.scatter(centroids[:, 0], centroids[:, 1], s=100, marker="X")
    if len(title) > 1:
        plt.title(title)
    if show:
        plt.show()
    if save:
        plt.savefig(title[:title.find("\n")]+".png")
        plt.close()


def test_elbow(X, plot_each_cluster=False):
    """
    Tests the elbow method: generates synthetit normally distributed clusters
    for k's in the range 2,7 and then runs kmeans for a variaty of k values 
    on the synthetic data.
    If plot=True, a graph of the elbow method will be shown.
    If plot_each_cluster=True, a graph of each clustering will be shown (not recommended)
    If save=True, the plot will be saved as "elbow.png"
    """

    costs = []
    ks = [k for k in range(2, 16)]  # ks on which we test
    for k in ks:
        np.random.shuffle(X)
        clustering, centroids = kmeans(X, k)  # cluster
        costs.append(cost(X, clustering, centroids, k))
        if plot_each_cluster:
            plot_clusters(X, clustering, centroids)
    return ks, costs


def elbow_on_data(X):
    """
    Runs the elbow method on a given dataset X
    """
    ks, costs = test_elbow(X)
    plt.plot(ks, costs, '.')
    plt.show()


def elbow_on_synthetic(plot=False, plot_each_cluster=False, save=False):
    """
    Tests the elbow method: generates synthetic normally distributed clusters
    for k's in the range 2,7 and then runs kmeans for a variaty of k values 
    on the synthetic data.
    If plot=True, a graph of the elbow method will be shown.
    If plot_each_cluster=True, a graph of each clustering will be shown (not recommended)
    If save=True, the plot will be saved as "elbow.png"
    """
    for true_k in range(2, 8):
        X = generate_synthetic(k=true_k)
        ks, costs = test_elbow(X)
        plt.subplot(2, 3, true_k-1)
        plt.title("True k="+str(true_k))
        plt.axvline(x=true_k, linewidth=1, linestyle="-")
        plt.plot(ks, costs, '.')
    if plot:
        plt.show()
    if save:
        plt.savefig("synthetic_elbow.png")


def biodata_tests(k=4, elbow=False, eigengap=False, histogram=False, similarity=False,
                  show=False, save=False):
    with open("microarray_data.pickle", 'rb') as f:
        biodata = pickle.load(f)
    # elbow for biodata
    if elbow:
        costs = test_elbow(biodata)
        plt.figure()
        plt.plot(np.array([i for i in range(len(costs))])+2, costs, '.')
        plt.title("Elbow Method for Biodata")
        plt.xlabel("K value")
        plt.ylabel("Cost")
        if show:
            plt.show()
        if save:
            plt.savefig("biodata_elbow.png")
    # eigengap for different percentiles for biodata
    if eigengap:
        sigma = 4.2095143425340655
        clus, cent = spectral(biodata, k, similarity_param=sigma, plot_eigenvalues=True)
    if histogram:
        _, _ = spectral(biodata, k, histogram=True)
    if similarity:
        for m in range(8, 21):
            plot_similarity(X, k, data_name="Biodata", similarity_param=m, similarity=mnn)


def silhouette_method(X, data_name="", save=False):
    """
    Tests the silhouette of some dataset
    :param X: the data
    """
    sil = []
    k_range = [k for k in range(2, 25)]
    for k in k_range:
        clustering, centroids = kmeans(X, k)
        title = "k="+str(k)
        # plot_clusters(X, clustering, centroids, title)
        sil.append(silhouette(X, clustering, centroids))
    plt.figure()
    plt.title("Silhouette as function of K "+data_name)
    plt.bar(k_range, sil)
    plt.show()
    if save:
        plt.savefig("Silhouette as function of K.png")
        plt.close()


def silhouette(X, clustering, centroids):
    """
    The silhouette method. I used the elbow in my answers but wrote this function as well.
    :param X: the data
    :param clustering: the clustering
    :param centroids: the locations of the k centroids.
    """
    samples, _ = X.shape
    k, _ = centroids.shape
    dists = euclid(X, X)
    sil = 0
    _, counts = np.unique(clustering, return_counts=True)
    for i in range(samples):
        icluster = clustering[i]
        cluster_size = counts[icluster] - 1
        rows = np.where(clustering == icluster)[0]
        ai = (1/cluster_size) * np.sum(dists[rows, i])
        cluster_distances = np.zeros((k,))
        for j in range(k):
            if j == icluster:
                cluster_distances[j] = np.inf  # ensuring b_i wont be the current cluster
            rows = np.where(clustering == j)[0]
            # calculate the distance between the j'th cluster's elements and x_i
            cluster_distances[j] += np.sum(dists[rows, i])
        bi = np.min(np.multiply(1/counts, cluster_distances))
        sil += (bi-ai)/max(ai, bi)
    return sil / samples


def compare_tsne_pca(dimension=2):
    """
    Compares the tsne and PCA methods of dimensionality reduction
    :param dimension: the dimension to which the data is reduced to
    """
    k = 10
    # load mnist dataset
    X, y = load_digits(n_class=k, return_X_y=True)
    num_samps, _ = X.shape
    # get the PCA reduction of the data
    X_pca = PCA(n_components=dimension).fit_transform(X)
    X_tsne = TSNE(n_components=dimension).fit_transform(X)
    pca_clustering, pca_centroids = kmeans(X_pca, k)
    tsne_clustering, tsne_centroids = kmeans(X_tsne, k)
    plt.figure()
    plt.subplot(2, 2, 1)
    plot_clusters(X_pca, pca_clustering, title="PCA kmeans")
    plt.subplot(2, 2, 2)
    plot_clusters(X_tsne, tsne_clustering, title="TSNE kmeans")
    plt.subplot(2, 2, 3)
    plot_clusters(X_pca, y, title="PCA True")
    plt.subplot(2, 2, 4)
    plot_clusters(X_tsne, y, title="TSNE True")
    plt.show()


if __name__ == '__main__':
    mpl.style.use('fivethirtyeight')  # prettier graphs
    # load the different datasets
    gaussians = generate_synthetic()
    with open('APML_pic.pickle', 'rb') as f:
        APML = pickle.load(f)
    circles = circles_example()
    # choose the dataset to work with
    X = APML
    data_name = "APML"
    k = 9  # choose some k
    # run the following line to plot kmeans clusters on the data
    kmeans_clustering, _ = kmeans(X, k)
    plot_clusters(X, kmeans_clustering, title=data_name, show=True, save=False)
    # run the following line to plot spectral clustering on the data
    spectral_clustering, _ = spectral(X, k, similarity=gaussian_kernel, p=0.75)
    plot_clusters(X, spectral_clustering, title=data_name, show=True, save=False)
    # run this to plot the eigengap
    _, _ = spectral(X, k, similarity=gaussian_kernel, plot_eigenvalues=True, p=0.75)
    # run this to plot similarity charts
    plot_similarity(X, k, data_name=data_name, similarity_param=11, similarity=mnn)
    # run this line to see the elbow method on synthetic data set
    elbow_on_synthetic(plot=True)
    # run this line to see the elbow method on given dataset
    elbow_on_data(X)
    # run this line to see the silhouette graph for the given data - I used elbow in my answers
    silhouette_method(X, data_name=data_name)
    # run the following function to create different graphs regarding the biological data
    biodata_tests(k=k, elbow=False, eigengap=False, histogram=False, similarity=False,
                  show=False, save=False)
    # run the following line to compare tSNE and PCA
    compare_tsne_pca(dimension=2)
