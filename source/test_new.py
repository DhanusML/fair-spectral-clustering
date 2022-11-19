from algorithms.spectral_clustering import *
from utils.utils import *

def get_sizes(clusters):
    return list(map(lambda x: len(x), clusters))


if __name__ == "__main__":
    # cluster_sizes = [25,25,25,25]
    cluster_sizes = [10, 20, 30, 40]
    edges, clusters_original = genGraph(100, cluster_sizes, 0.9, 0.1)
    etas = [0.25,0.25,0.25,0.25]
    groups = genGroups(100, etas, clusters_original)

    ## unnormalized vanilla SC ##
    clusters_vanilla_u = unnormalizedSC(100, 4, edges)
    cuts_vanilla_u = []
    for c in clusters_vanilla_u:
        cuts_vanilla_u.append(get_num_cuts(edges, c))

    group_cluster_mat_vanilla_u = get_group_cluster_matrix(clusters_vanilla_u,
                                                           groups)

    ## normalized vanilla SC ##
    clusters_vanilla = normalizedSC(100, 4, edges)
    cuts_vanilla = []
    for c in clusters_vanilla_u:
        cuts_vanilla.append(get_num_cuts(edges, c))

    group_cluster_mat_vanilla = get_group_cluster_matrix(clusters_vanilla,
                                                         groups)

    ## unnormalized cons SC ##
    clusters_con_u = unnormalizedConSC(100, 4, edges, groups)
    cuts_con_u = []

    for c in clusters_con_u:
        cuts_con_u.append(get_num_cuts(edges, c))

    group_cluster_mat_con_u = get_group_cluster_matrix(clusters_con_u, groups)

    ## normalized cons SC ##
    clusters_con = normalizedConSC(100, 4, edges, groups)
    cuts_con = []

    for c in clusters_con:
        cuts_con.append(get_num_cuts(edges, c))

    group_cluster_mat_con = get_group_cluster_matrix(clusters_con, groups)


    clusters = normalizedConSC(100, 4, edges, groups)
    cuts = []


    print("\nvanilla unnormalized")
    print("clusters: ", get_sizes(clusters_vanilla_u))
    print("cuts: ", cuts_vanilla_u)
    print("group cluster matrix:\n", group_cluster_mat_vanilla_u)
    print("balance ", get_balance(group_cluster_mat_vanilla_u))

    print("\nvanilla normalized")
    print("clusters: ", get_sizes(clusters_vanilla))
    print("cuts: ", cuts_vanilla)
    print("group cluster matrix:\n", group_cluster_mat_vanilla)
    print("balance ", get_balance(group_cluster_mat_vanilla))

    print("\ncons unnormalized")
    print("clusters: ", get_sizes(clusters_con_u))
    print("cuts: ", cuts_con_u)
    print("group cluster matrix:\n", group_cluster_mat_con_u)
    print("balance ", get_balance(group_cluster_mat_con_u))

    print("\ncons normalized")
    print("clusters: ", get_sizes(clusters_con))
    print("cuts: ", cuts_con)
    print("group cluster matrix:\n", group_cluster_mat_con)
    print("balance ", get_balance(group_cluster_mat_con))

# visualize(100, edges, clusters)
