import algorithms.spectral_clustering as al
import utils.utils as ut
from utils import drug as dr
import numpy as np

def get_sizes(clusters):
    return list(map(lambda x: len(x), clusters))

def getEdges(mat):
    n = mat.shape[0]
    edges = []
    for i in range(n):
        for j in range(n):
            if mat[i][j] == 1:
                edges.append(np.array([i,j], dtype=np.int32))

    return np.array(edges, dtype=np.int32)


if __name__=="__main__":
    '''
    adj_tot, adj_1, adj_2, gender, ethinicity\
        = dr._get_adj_mat_and_groups('../data/drug/')
    #output = dr.get_grouplists_drugnet()
    '''
    adj_mat, gender, ethinicity\
        = dr.__get_adj_mat_and_groups('../data/drug/')


    edges = getEdges(adj_mat)
    groups = ethinicity

    num_clusters = 7
    num_nodes = len(adj_mat)

    ### vanilla unnormalized clustering ###
    clusters_vanilla_u = al._unnormalizedSC(num_nodes,
                                           num_clusters, adj_mat)
    cuts_vanilla_u = []
    for c in clusters_vanilla_u:
        pass

    group_cluster_mat_vanilla_u = ut.get_group_cluster_matrix(
        clusters_vanilla_u,
        groups
    )

    balance_vanilla_u = ut.get_balance(group_cluster_mat_vanilla_u)


    ### vanilla normalized clustering ###
    clusters_vanilla = al._normalizedSC(num_nodes,
                                     num_clusters, adj_mat)
    cuts_vanilla = []
    for c in clusters_vanilla:
        pass

    group_cluster_mat_vanilla = ut.get_group_cluster_matrix(
        clusters_vanilla,
        groups
    )

    balance_vanilla = ut.get_balance(group_cluster_mat_vanilla)



    ### constrained unnormalized clustering ###
    clusters_con_u = al._unnormalizedConSC(num_nodes,
                                          num_clusters, adj_mat,
                                          groups)
    cuts_con_u = []
    for c in clusters_con_u:
        pass

    group_cluster_mat_con_u = ut.get_group_cluster_matrix(
        clusters_con_u,
        groups
    )

    balance_con_u = ut.get_balance(group_cluster_mat_con_u)



    '''
    ### constrained normalized clustering ###
    clusters_con = al._normalizedConSC(num_nodes,
                                      num_clusters, adj_mat,
                                      groups)
    cuts_con = []
    for c in clusters_con:
        pass

    group_cluster_mat_con = ut.get_group_cluster_matrix(
        clusters_con,
        groups
    )

    balance_con = ut.get_balance(group_cluster_mat_con)
    '''

    #### printing and visualization ####
    ## vanilla unnormalized ##
    print("\nvanilla unnormalized")
    print("clusters: ", get_sizes(clusters_vanilla_u))
    print("cuts: ", cuts_vanilla_u)
    print("group cluster matrix:\n", group_cluster_mat_vanilla_u)
    print("balance:", ut.get_balance(group_cluster_mat_vanilla_u))
    ut.visualizeGroups(clusters_vanilla_u, groups, edges)

    ## vanilla normalized ##
    print("\nvanilla normalized")
    print("clusters: ", get_sizes(clusters_vanilla))
    print("cuts: ", cuts_vanilla)
    print("group cluster matrix:\n", group_cluster_mat_vanilla)
    print("balance:", ut.get_balance(group_cluster_mat_vanilla))
    #ut.visualizeGroups(clusters_vanilla, reduced_group_list, edges)


    ## constrained unnormalized ##
    print("\nconstrained unnormalized")
    print("clusters: ", get_sizes(clusters_con_u))
    print("cuts: ", cuts_con_u)
    print("group cluster matrix:\n", group_cluster_mat_con_u)
    print("balance:", ut.get_balance(group_cluster_mat_con_u))
    ut.visualizeGroups(clusters_con_u, groups, edges)

    '''
    ## constrained normalized ##
    print("\nconstrained normalized")
    print("clusters: ", get_sizes(clusters_con))
    print("cuts: ", cuts_con)
    print("group cluster matrix:\n", group_cluster_mat_con)
    print("balance:", ut.get_balance(group_cluster_mat_con))
    #ut.visualizeGroups(clusters_con, reduced_group_list, edges)
    '''
