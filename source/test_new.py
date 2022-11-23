#from algorithms.spectral_clustering import al
#from utils.utils import ut
import algorithms.spectral_clustering as al
import utils.utils as ut

def get_sizes(clusters):
    return list(map(lambda x: len(x), clusters))


if __name__ == "__main__":
    cluster_sizes = [10, 10, 10, 10]
    etas = [0.25,0.25,0.25,0.25]
    #cluster_sizes = [50, 50]
    #etas = [0.5, 0.5]
    num_vertices = sum(cluster_sizes)
    num_clusters = len(cluster_sizes)

    edges, clusters_original, groups = \
        ut.genGraphWithGroups(
              num_vertices, cluster_sizes, etas,
              0.9, 0.8, 0.8, 0.1
              #0.9,0.2,0.8,0.1
        )
    '''
    cluster_sizes = [10, 20, 30, 40]
    edges, clusters_original = genGraph(100, cluster_sizes, 0.9, 0.1)
    etas = [0.25,0.25,0.25,0.25]
    groups = genGroups(100, etas, clusters_original)
    '''

    ## unnormalized vanilla SC ##
    clusters_vanilla_u = al.unnormalizedSC(num_vertices, num_clusters, edges)
    cuts_vanilla_u = []
    for c in clusters_vanilla_u:
        cuts_vanilla_u.append(ut.get_num_cuts(edges, c))

    misMat_vanilla_u = ut.getMisclassificationMat(clusters_vanilla_u, clusters_original)
    group_cluster_mat_vanilla_u = ut.get_group_cluster_matrix(clusters_vanilla_u,
                                                           groups)
    #ut.visualizeGroups(clusters_vanilla_u, groups, edges)

    ## normalized vanilla SC ##
    clusters_vanilla = al.normalizedSC(num_vertices, num_clusters, edges)
    cuts_vanilla = []
    for c in clusters_vanilla_u:
        cuts_vanilla.append(ut.get_num_cuts(edges, c))

    misMat_vanilla = ut.getMisclassificationMat(clusters_vanilla, clusters_original)
    group_cluster_mat_vanilla = ut.get_group_cluster_matrix(clusters_vanilla,
                                                         groups)

    ## unnormalized cons SC ##
    clusters_con_u = al.unnormalizedConSC(num_vertices, num_clusters, edges, groups)
    cuts_con_u = []

    for c in clusters_con_u:
        cuts_con_u.append(ut.get_num_cuts(edges, c))

    misMat_con_u = ut.getMisclassificationMat(clusters_con_u, clusters_original)
    group_cluster_mat_con_u = ut.get_group_cluster_matrix(clusters_con_u, groups)
    #ut.visualizeGroups(clusters_con_u, groups, edges)

    ## normalized cons SC ##
    clusters_con = al.normalizedConSC(num_vertices, num_clusters, edges, groups)
    cuts_con = []

    for c in clusters_con:
        cuts_con.append(ut.get_num_cuts(edges, c))

    misMat_con = ut.getMisclassificationMat(clusters_con, clusters_original)
    group_cluster_mat_con = ut.get_group_cluster_matrix(clusters_con, groups)


    clusters = al.normalizedConSC(num_vertices, num_clusters, edges, groups)
    cuts = []


    print("\nvanilla unnormalized")
    print("clusters: ", get_sizes(clusters_vanilla_u))
    print("cuts: ", cuts_vanilla_u)
    print("group cluster matrix:\n", group_cluster_mat_vanilla_u)
    print("balance:", ut.get_balance(group_cluster_mat_vanilla_u))
    print("misclassification matrix:\n", misMat_vanilla_u)
    print("errors", ut.getMisclassificationError(misMat_vanilla_u))

    print("\nvanilla normalized")
    print("clusters: ", get_sizes(clusters_vanilla))
    print("cuts: ", cuts_vanilla)
    print("group cluster matrix:\n", group_cluster_mat_vanilla)
    print("balance:", ut.get_balance(group_cluster_mat_vanilla))
    print("misclassification matrix:\n", misMat_vanilla)
    print("errors", ut.getMisclassificationError(misMat_vanilla))

    print("\ncons unnormalized")
    print("clusters: ", get_sizes(clusters_con_u))
    print("cuts: ", cuts_con_u)
    print("group cluster matrix:\n", group_cluster_mat_con_u)
    print("balance:", ut.get_balance(group_cluster_mat_con_u))
    print("misclassification matrix:\n", misMat_con_u)
    print("errors", ut.getMisclassificationError(misMat_con_u))

    print("\ncons normalized")
    print("clusters: ", get_sizes(clusters_con))
    print("cuts: ", cuts_con)
    print("group cluster matrix:\n", group_cluster_mat_con)
    print("balance:", ut.get_balance(group_cluster_mat_con))
    print("misclassification matrix:\n", misMat_con)
    print("errors", ut.getMisclassificationError(misMat_con))

# visualize(100, edges, clusters)
