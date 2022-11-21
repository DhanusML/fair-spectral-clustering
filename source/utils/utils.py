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
    Helper function for genGraphWithGroups
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
    """
    Computes balance of each cluster from `group-cluster-matrix'

    Params:
        mat (np.ndarray): matrix with ijth entry being the number
                          of elements of group j in cluster i.

    Returns:
        balance (np.array): array of floats. ith entry is the
                            balance of the ith cluster.
    """
    max_groups = np.max(mat, axis=1)
    min_groups = np.min(mat, axis=1)
    balances = (min_groups/max_groups).reshape(-1)

    return balances.reshape(-1)


def _get_group(v, groups):
    for i, group in enumerate(groups):
        if v in group:
            return i


def visualizeGroups(clusters, groups, edges, title=''):
    """
    !!!!!!!!!!!!!!!!!!!!!!!!!
    !!! Needs improvement !!!
    !!!!!!!!!!!!!!!!!!!!!!!!!

    Visualizes each cluster with different group labelled
    by different colors.

    Params:
        clusters (list): List of list of vertices forming clusters.
        groups (list): list of list of vertices forming groups.
        edges (np.ndarray): |E|*2 matrix, each row is an edge.

    Returns:
        None
    """
    assert len(groups) < 7, "not enough colors to label with."

    G = nx.Graph()
    G.add_edges_from(edges)
    G.remove_nodes_from(list(nx.isolates(G)))
    pos=nx.spring_layout(G)
    colors = ['blue', 'red', 'cyan', 'green', 'magenta', 'black']

    plt.figure(0)
    plt.title("Clustering")
    for i,c in enumerate(clusters):
        nx.draw_networkx(
            G, pos=pos, nodelist=c,
            node_color=colors[i],
            with_labels=False
        )

    plt.figure(1)
    plt.suptitle("Clusters")
    colors = ['green', 'red', 'yellow', 'magenta', 'gray', 'black']
    num_clusters = len(clusters)
    for i, cluster in enumerate(clusters):
        new_edges = []
        for e in edges:
            if e[0] in cluster and e[1] in cluster:
                new_edges.append(e)
        G = nx.Graph()

        for v in cluster:
            G.add_nodes_from(
                [[v, {'color':colors[_get_group(v,groups)]}]]
            )
        G.add_edges_from(new_edges)
        c_map = [node[1]['color'] for node in G.nodes(data=True)]
        plt.subplot(1 , num_clusters, i+1)
        nx.draw(G, node_color=c_map)

    plt.show()


def genGraphWithGroups(n, cluster_sizes, etas, a, b, c, d):
    """
    Generate random graph using a variant of SBM.
    a, b, c, d are probabilities of edges between vertices:
        a -> same cluster, same group,
        b -> different cluster, same group,
        c -> same cluster, different group,
        d -> different cluster, different group.

    Params:
        n (int): number of vertices in the graph.
        cluster_sizes (list): list of integers, each entry represents
                              sizes of the clusters. Must add up to n
        etas (list): list of floats, ith entry represents the fraction
                     of elements of ith group in each cluster.
        a (float): probability of edge between same cluster, same group.
        b (float): probability of edge between different cluster, same group.
        c (float): probability of edge between same cluster, different group.
        d (float): probability of edge between different cluster, differnt group.

    Returns:
        edges (np.ndarray): |E|*2 array of np.int32's. Each row represents
                            an edge in the graph.
        clusters (list): list of list of vertices forming clusters.
        groups (list): list of list of vertices forming groups.

    Warnings:
        Throws error if the elements of cluster_sizes don't add up to n
        or if sum(etas) is not 1.
    """

    assert sum(cluster_sizes) == n, \
        "sum of cluster sizes is not equal to the number of vertices"

    assert sum(etas)==1, "sum of fractions should add up to 1"

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

    groups = genGroups(n, etas, clusters)

    edges = []

    # edges with prob a, b, c, d
    for v1 in vertices:
        for v2 in vertices:
            if v1 != v2:
                for cl in clusters:
                    if v1 in cl and v2 in cl: # sc
                        for gr in groups:
                            if v1 in gr and v2 in gr:
                                if np.random.random_sample()<a: # sg
                                    edges.append(
                                        np.array(
                                            [v1, v2], dtype=np.int32
                                        )
                                    )
                        else: #sc, not sg
                            if np.random.random_sample()<c:
                                edges.append(
                                    np.array(
                                        [v1, v2], dtype=np.int32
                                    )
                                )

                else: # not sc
                    for gr in groups:
                        if v1 in gr and v2 in gr: # sg
                            if np.random.random_sample()<b:
                                edges.append(
                                    np.array(
                                        [v1, v2], dtype=np.int32
                                    )
                                )

                    else: # not sg
                        if np.random.random_sample()<d:
                            edges.append(
                                np.array(
                                    [v1, v2], dtype=np.int32
                                )
                            )

    return np.array(edges), clusters, groups


def getMisclassificationMat(clusters1, clusters2):
    """
    Takes two clusterings as input and returns a matrix whose
    ijth entry is the cardinality of symmetric difference of
    the ith cluster based on the first clustering and the jth
    cluster based on the second clustering.

    Params:
        cluster1 (list): list of list of vertices forming cluster1.
        cluster2 (list): list of list of vertices forming cluster2.

    Returns:
        mis_mat (np.ndarray): ijth entry is the number of elements
                              in the symmetric difference of
                              cluster1[i] and cluster2[j].

    Warnings:
        Throws an error if the two clusterings have different number
        of clusters.
    """
    assert len(clusters1) == len(clusters2), "number of clusters must be same"
    num = len(clusters1)

    cs1 = [set(x) for x in clusters1]
    cs2 = [set(x) for x in clusters2]

    mis_mat = np.zeros((num, num), dtype=np.int32)

    for i in range(num):
        s1 = cs1[i]
        for j in range(num):
            s2 = cs2[j]
            mis_num = len(s1.symmetric_difference(s2))
            mis_mat[i][j] = mis_num

    return mis_mat
