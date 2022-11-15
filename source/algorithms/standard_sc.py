import numpy as np
from scipy.cluster import vq as cl


def kMM(k, pts):
    """
    Performs k-means clustering on dataset

    Params:
        k (int): the number of clusters
        pts (np.ndarray): an m*n array. Each row is a point in
                          n dimensional Euclidean space.

    Returns:
        clusters (list): List of k lists. Each element of the list
                         is another list which contains the indices
                         of the points in the cluster.
    """
    clusters = [[] for i in range(k)]
    w_pts = cl.whiten(pts)
    centroids = cl.kmeans(w_pts, k)[0]

    for i, p in enumerate(w_pts):
        dists = []
        dists = np.array([
            np.linalg.norm(p-center) for center in centroids
        ])
        cl_id = np.argmin(dists)
        clusters[cl_id].append(i)
    return clusters


def _get_adj_mat(n, edges):
    adj_mat = np.zeros((n, n))

    for edge in edges:
        i, j = edge[0], edge[1]
        adj_mat[i,j] = 1
    return adj_mat

def _get_lap_mat(adj_mat):
    deg_mat = np.diag(np.sum(adj_mat, axis=0))
    lap_mat = deg_mat - adj_mat
    return lap_mat



def unnormalizedSC(n, k, edges):
    """
    Performs unnormalized Spectral clustering

    Params:
        n (int): the number of nodes in the graph.
        k (int): the number of clusters required.
        edges (np.ndarray): |E|*2 matrix with each row representing
                            an edge.

    Returns:
        clusters (list): a list whose element are list of labels of
                         nodes belonging to the same cluster.

    """
    adj_mat = _get_adj_mat(n, edges)
    lap_mat = _get_lap_mat(adj_mat)

    w, v = np.linalg.eigh(lap_mat)
    vecs = v[:, :k]

    clusters = kMM(k, vecs)

    return clusters


def normalizedSC(n, k, edges):
    """
    Performs normalized Spectral clustering

    Params:
        n (int): the number of nodes in the graph.
        k (int): the number of clusters required.
        edges (np.ndarray): |E|*2 matrix with each row representing
                            an edge.

    Returns:
        clusters (list): a list whose element are list of labels of
                         nodes belonging to the same cluster.

    Warnings:
        Might throw a warning when there are isolated edges
    """
    adj_mat = _get_adj_mat(n, edges)
    lap_mat = _get_lap_mat(adj_mat)

    deg_mat = np.sum(adj_mat, axis=0)
    deg_inv_half = np.diag(1/np.sqrt(deg_mat))
    norm_lap_mat = deg_inv_half @ (lap_mat @ deg_inv_half)
    w, v = np.linalg.eigh(lap_mat)
    vecs = v[:, :k]
    clusters = kMM(k, vecs)

    return clusters
