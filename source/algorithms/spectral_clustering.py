import numpy as np
from scipy.cluster import vq as cl
from scipy.linalg import null_space, sqrtm


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
    vecs = v[:, :k+1]

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
    norm_lap_mat = deg_inv_half@(lap_mat @ deg_inv_half)
    w, v = np.linalg.eigh(lap_mat)
    vecs = v[:, :k+1]
    clusters = kMM(k, vecs)

    return clusters


def unnormalizedConSC(n, k, edges, groups):
    """
    Performs unnormalized spectral clustering with population fairness
    constraint.

    Params:
        n (int): the number of vertices in the graph.
        k (int): the number of clusters required.
        edges (np.ndarray): |E|*2 matrix with each row representing
                            an edge.
        groups (list): list of lists. Each list is a collection
                       of nodes forming a group.

    Returns:
        clusters (list): list of lists. Each list is the collection
                         of nodes forming the cluster.
    """
    adj_mat = _get_adj_mat(n, edges)
    lap_mat = _get_lap_mat(adj_mat)

    h = len(groups)
    f_mat = np.zeros((n, h)) # each col represents group

    for i  in range(h):
        for j in groups[i]:
            f_mat[j][i] = 1 # jth vertex in ith group

    vs_by_n = np.array(list(map(lambda x: len(x)/n, groups))).reshape((1,h))
    F = f_mat - vs_by_n
    F = F[:, :-1] # all except the last column.

    Z = null_space(F.T)

    new_lap_mat = Z.T @ (lap_mat @ Z)
    w,v = np.linalg.eigh(new_lap_mat)
    Y = v[:, :k+1]
    H = Z @ Y
    clusters = kMM(k, H)

    return clusters



def normalizedConSC(n, k, edges, groups):
    """
    Performs normalized spectral clustering with population fairness
    constraint.

    Params:
        n (int): the number of vertices in the graph.
        k (int): the number of clusters required.
        edges (np.ndarray): |E|*2 matrix with each row representing
                            an edge.
        groups (list): list of lists. Each list is a collection
                       of nodes forming a group.

    Returns:
        clusters (list): list of lists. Each list is the collection
                         of nodes forming the cluster.

    Warnings:
        Shows error if there are isolated vertices.
    """
    adj_mat = _get_adj_mat(n, edges)
    lap_mat = _get_lap_mat(adj_mat)

    h = len(groups)

    f_mat = np.zeros((n, h)) # each col represents group

    for i  in range(h):
        for j in groups[i]:
            f_mat[j][i] = 1 # jth vertex in ith group

    vs_by_n = np.array(list(map(lambda x: len(x)/n, groups))).reshape((1,h))
    F = f_mat - vs_by_n
    F = F[:, :-1] # all except the last column.

    Z = null_space(F.T)

    ## calculations from the paper ##
    deg_mat = np.diag(np.sum(adj_mat, axis=0))
    Q2 = Z.T @ (deg_mat @ Z)
    Q = sqrtm(Q2)
    Q_inv = np.linalg.inv(Q)

    unorm_lap = Z.T @ (lap_mat @ Z)
    new_lap_mat = Q_inv @ (unorm_lap @ Q_inv)
    w, v = np.linalg.eigh(new_lap_mat)
    X = v[:, :k+1]
    H = Z @ (Q_inv @ X)
    clusters = kMM(k, H)

    return clusters



"""
The _<clustering function> perform the same task as
<clustering function> without underscore, but takes
adjacency matrix instead of list of edges.

_unnormalizedSC
_normalizedSC
_unnormalizedConSC
_normalizedConSC
"""

def _unnormalizedSC(n, k, adj_mat):
    """
    Performs unnormalized Spectral clustering

    Params:
        n (int): the number of nodes in the graph.
        k (int): the number of clusters required.
        adj_mat (np.ndarray): n*n adjacency matrix.

    Returns:
        clusters (list): a list whose element are list of labels of
                         nodes belonging to the same cluster.

    """
    lap_mat = _get_lap_mat(adj_mat)

    w, v = np.linalg.eigh(lap_mat)
    vecs = v[:, :k+1]

    clusters = kMM(k, vecs)

    return clusters


def _normalizedSC(n, k, adj_mat):
    """
    Performs normalized Spectral clustering

    Params:
        n (int): the number of nodes in the graph.
        k (int): the number of clusters required.
        adj_mat (np.ndarray): n*n adjacency matrix.

    Returns:
        clusters (list): a list whose element are list of labels of
                         nodes belonging to the same cluster.

    Warnings:
        Might throw a warning when there are isolated edges
    """
    lap_mat = _get_lap_mat(adj_mat)

    deg_mat = np.sum(adj_mat, axis=0)
    deg_inv_half = np.diag(1/np.sqrt(deg_mat))
    norm_lap_mat = deg_inv_half@(lap_mat @ deg_inv_half)
    w, v = np.linalg.eigh(lap_mat)
    vecs = v[:, :k+1]
    clusters = kMM(k, vecs)

    return clusters


def _unnormalizedConSC(n, k, adj_mat, groups):
    """
    Performs unnormalized spectral clustering with population fairness
    constraint.

    Params:
        n (int): the number of vertices in the graph.
        k (int): the number of clusters required.
        adj_mat (np.ndarray): n*n adjacency matrix.
        groups (list): list of lists. Each list is a collection
                       of nodes forming a group.

    Returns:
        clusters (list): list of lists. Each list is the collection
                         of nodes forming the cluster.
    """
    lap_mat = _get_lap_mat(adj_mat)

    h = len(groups)
    f_mat = np.zeros((n, h)) # each col represents group

    for i  in range(h):
        for j in groups[i]:
            f_mat[j][i] = 1 # jth vertex in ith group

    vs_by_n = np.array(list(map(lambda x: len(x)/n, groups))).reshape((1,h))
    F = f_mat - vs_by_n
    F = F[:, :-1] # all except the last column.

    Z = null_space(F.T)

    new_lap_mat = Z.T @ (lap_mat @ Z)
    w,v = np.linalg.eigh(new_lap_mat)
    Y = v[:, :k+1]
    H = Z @ Y
    clusters = kMM(k, H)

    return clusters



def _normalizedConSC(n, k, adj_mat, groups):
    """
    Performs normalized spectral clustering with population fairness
    constraint.

    Params:
        n (int): the number of vertices in the graph.
        k (int): the number of clusters required.
        adj_mat (np.ndarray): n*n adjacency matrix.
        groups (list): list of lists. Each list is a collection
                       of nodes forming a group.

    Returns:
        clusters (list): list of lists. Each list is the collection
                         of nodes forming the cluster.

    Warnings:
        Shows error if there are isolated vertices.
    """
    lap_mat = _get_lap_mat(adj_mat)

    h = len(groups)

    f_mat = np.zeros((n, h)) # each col represents group

    for i  in range(h):
        for j in groups[i]:
            f_mat[j][i] = 1 # jth vertex in ith group

    vs_by_n = np.array(list(map(lambda x: len(x)/n, groups))).reshape((1,h))
    F = f_mat - vs_by_n
    F = F[:, :-1] # all except the last column.

    Z = null_space(F.T)

    ## calculations from the paper ##
    deg_mat = np.diag(np.sum(adj_mat, axis=0))
    Q2 = Z.T @ (deg_mat @ Z)
    Q = sqrtm(Q2)
    Q_inv = np.linalg.inv(Q)

    unorm_lap = Z.T @ (lap_mat @ Z)
    new_lap_mat = Q_inv @ (unorm_lap @ Q_inv)
    w, v = np.linalg.eigh(new_lap_mat)
    X = v[:, :k+1]
    H = Z @ (Q_inv @ X)
    clusters = kMM(k, H)

    return clusters
