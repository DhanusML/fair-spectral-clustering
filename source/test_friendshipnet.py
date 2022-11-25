import algorithms.spectral_clustering as al
import utils.utils as ut
from utils import preprocessing as fnp
import numpy as np

def get_sizes(clusters):
    return list(map(lambda x: len(x), clusters))


if __name__=="__main__":
    tot_num_nodes, num_conn_nodes, edges, orphan_nodes,\
    gender_group, reduced_group_list,\
        reduced_class_group_list = fnp.get_friendshipnet_data()

    #print(edges)

    num_clusters = 5

    groups = reduced_group_list
    num_nodes = num_conn_nodes

    ### vanilla unnormalized clustering ###
    clusters_vanilla_u = al.unnormalizedSC(num_nodes,
                                           num_clusters, edges)
    cuts_vanilla_u = []
    for c in clusters_vanilla_u:
        cuts_vanilla_u.append(ut.get_num_cuts(edges, c))

    group_cluster_mat_vanilla_u = ut.get_group_cluster_matrix(
        clusters_vanilla_u,
        groups
    )

    balance_vanilla_u = ut.get_balance(group_cluster_mat_vanilla_u)


    ### vanilla normalized clustering ###
    clusters_vanilla = al.normalizedSC(num_nodes,
                                     num_clusters, edges)
    cuts_vanilla = []
    for c in clusters_vanilla:
        cuts_vanilla.append(ut.get_num_cuts(edges, c))

    group_cluster_mat_vanilla = ut.get_group_cluster_matrix(
        clusters_vanilla,
        groups
    )

    balance_vanilla = ut.get_balance(group_cluster_mat_vanilla)



    ### constrained unnormalized clustering ###
    clusters_con_u = al.unnormalizedConSC(num_nodes,
                                          num_clusters, edges,
                                          groups)
    cuts_con_u = []
    for c in clusters_con_u:
        cuts_con_u.append(ut.get_num_cuts(edges, c))

    group_cluster_mat_con_u = ut.get_group_cluster_matrix(
        clusters_con_u,
        groups
    )

    balance_con_u = ut.get_balance(group_cluster_mat_con_u)



    ### constrained normalized clustering ###
    clusters_con = al.normalizedConSC(num_nodes,
                                      num_clusters, edges,
                                      groups)
    cuts_con = []
    for c in clusters_con:
        cuts_con.append(ut.get_num_cuts(edges, c))

    group_cluster_mat_con = ut.get_group_cluster_matrix(
        clusters_con,
        groups
    )

    balance_con = ut.get_balance(group_cluster_mat_con)

    #### printing and visualization ####
    ## vanilla unnormalized ##
    print("\nvanilla unnormalized")
    print("clusters: ", get_sizes(clusters_vanilla_u))
    print("cuts: ", cuts_vanilla_u)
    print("group cluster matrix:\n", group_cluster_mat_vanilla_u)
    print("balance:", ut.get_balance(group_cluster_mat_vanilla_u))
    ut.visualizeGroups(clusters_vanilla_u, reduced_group_list, edges)

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
    ut.visualizeGroups(clusters_con_u, reduced_group_list, edges)

    ## constrained normalized ##
    print("\nconstrained normalized")
    print("clusters: ", get_sizes(clusters_con))
    print("cuts: ", cuts_con)
    print("group cluster matrix:\n", group_cluster_mat_con)
    print("balance:", ut.get_balance(group_cluster_mat_con))
    #ut.visualizeGroups(clusters_con, reduced_group_list, edges)


