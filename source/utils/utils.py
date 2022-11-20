import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def genGraph(n, cluster_sizes, p, q):
    """
    Generates random graph

    Params:
            n (int): the number of vertices in the graph.
            cluster_sizes (list): list of sizes of each cluster in
                                the graph (must add up to n)
            p (float): probability of an edge within the cluster.
            q (float): probability of an edge between clusters.

    Returns:
        edges (np.ndarray): |E|*2 matrix with each rows representing an edge
        clusters (list): list of list of clusters forming the true clusters.
    """
    assert sum(cluster_sizes) == n, \
        "sum of cluster sizes is not equal to the number of vertices"


    vertices = np.arange(n, dtype=np.int32)
    clusters = []

    permuted_vertices = np.random.permutation(vertices)

    # get cumulative cluster sizes.
    l_index = 0

    for i in range(len(cluster_sizes)-1):
        r_index = l_index + cluster_sizes[i+1]
        cluster = permuted_vertices[l_index:r_index]
        clusters.append(cluster)
        l_index = r_index
    clusters.append(permuted_vertices[l_index:])

    edges = []

    for i in vertices:
        for j in vertices:
            if i != j:
                for c in clusters:
                    if i in c and j in c:
                        if np.random.random_sample()<p:
                            edges.append(np.array([i,j], dtype=np.int32))
                            break
                else:
                    if np.random.random_sample()<q:
                        edges.append(np.array([i,j], dtype=np.int32))

    return np.array(edges), clusters


def visualize(n, edges, clusters=[]):
    """
    displays the graph with clusters labelled

    Params:
        n (int): number of vertices in the graph
        edges (np.ndarray): |E|*2 matrix with each row representing
                            an edge.
        clusters (list): (optional) list of, list of vertices
                         forming the clusters.

    Returns:
        None
    """
    assert len(clusters)<7, "more clusters than colours given."

    vertices = range(n)
    G = nx.Graph()
    G.add_edges_from(edges)
    G.remove_nodes_from(list(nx.isolates(G)))
    pos=nx.spring_layout(G)
    colors = ['blue', 'red', 'cyan', 'green', 'magenta', 'black']


    for i,c in enumerate(clusters):
        nx.draw_networkx(
            G, pos=pos, nodelist=c,
            node_color=colors[i],
            with_labels=False
        )
    plt.show()


def genGroups(n, etas, clusters):
    """
    Assigns the groups for nodes in clusters.

    Params:
        n (int): number of vertices in the graph.
        etas (list): list of floats. Contains as many entries as
                     there are groups. ith entry is the fraction
                     of the ith group in each cluster.
        clutsters (list): list of list of collection of vertices
                          forming the clusters.

    Returns:
        groups (list): list of list of nodes in the same group.
    """
    assert sum(etas)==1, "sum of fractions should add up to 1"
    h = len(etas)
    groups = [[] for i in range(h)]
    _eta_cumul = etas.copy()

    for i in range(h-1):
        _eta_cumul[i+1] = _eta_cumul[i]+etas[i+1]

    for cluster in clusters:
        c = len(cluster)
        group = 0
        for i in range(c):
            if i < _eta_cumul[group]*c:
                groups[group].append(cluster[i])

            else:
                group += 1
                groups[group].append(cluster[i])

    return groups


def get_num_cuts(edges, cluster):
    """
    Gives the cut of a cluster (no. of edges to be removed for
    the collection of vertices to become isolated from the rest
    of the graph.)

    Params:
        edges (np.ndarray): |E|*2 matrix, each row is an
                            edge.

    Returns:
        cuts (int): the number of cuts for the cluster.
    """
    cuts = 0

    for edge in edges:
        a, b = edge[0], edge[1]

        if a in cluster and b not in cluster:
            cuts += 1

        if a not in cluster and b in cluster:
            cuts += 1

    return cuts


def get_group_cluster_matrix(clusters, groups):
    """
    Gives a C*G matrix whose ijth entry is the number of
    members of group j in the ith cluster.

    Params:
        clusters (list): List of list of vertices forming clusters.
        groups (list): List of list of vertices belonging to each
                       groups.

    Returns:
        mat (np.ndarray): (no. of clusters)*(no. of groups) matrix
                          ij th entry is the number of elements
                          of group j in the ith cluster.
    """
    mat = np.zeros((len(clusters), len(groups)), dtype=np.int32)

    for j, group in enumerate(groups):
        for i, cluster in enumerate(clusters):
            for v in cluster:
                if v in group:
                    mat[i][j] += 1
    return mat


def get_balance(mat):
    max_groups = np.max(mat, axis=1)
    min_groups = np.min(mat, axis=1)
    balances = (min_groups/max_groups).reshape(-1)

    return balances.reshape(-1)
