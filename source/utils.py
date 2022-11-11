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
    pos=nx.spring_layout(G)
    colors = ['blue', 'red', 'cyan', 'green', 'magenta', 'black']


    for i,c in enumerate(clusters):
        nx.draw_networkx(
            G, pos=pos, nodelist=c,
            node_color=colors[i],
            with_labels=False
        )
    plt.show()
