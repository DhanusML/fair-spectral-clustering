# # Following function pre-procceses the FriendshipNET dataset and returns the list of edges (Nx2 matrix).

import numpy as np
import copy

def get_friendshipnet_data():
    """Returns the following in the same order:
        No_of_Nodes, 
        No_of_connected_nodes,
        edges,
        orphannodes,
        gender_group,
        reduced_group_list"""


    metadata_orig = np.genfromtxt(
            "../data/metadata_2013.txt",
            delimiter="\t",
            # skip_header=True,
            dtype="str",
        )
        # )[:,0]        #For only first column


    edges_orig = np.genfromtxt(
            "../data/Friendship-network_data_2013.csv",
            delimiter=" ", 
            # skip_header=True,
            # dtype="str",
        )



    # edges_orig.shape    #   (668,2)
    allnodes = np.unique(metadata_orig[:, 0].astype(int))   #329
    mapping = np.unique(edges_orig)        #only 134 nodes are interconnected

    orphannodes = np.setdiff1d(allnodes, mapping)   #195

    edges = np.copy(edges_orig)
    metadata = np.copy(metadata_orig[:, 0]).astype(int)
    for j in np.unique(mapping):
        indices = np.where(edges[:,0] == j)
        edges[indices, 0] = np.where(mapping == j)
        
        indices = np.where(edges[:,1] == j)
        edges[indices, 1] = np.where(mapping == j)

        index = np.where(metadata_orig[:,0].astype(float) == float(j))
        metadata[index] = np.where(mapping == j)

    for j in orphannodes:
        index = np.where(metadata_orig[:,0].astype(float) == float(j))
        metadata[index] = np.where(orphannodes == j)
        metadata[index] += 134

    # metadata now contains relabelled data, with 
    # values 0 to 133 indicating the nodes which 
    # are interconnected and indicated by edges, 
    # all other nodes are numbered 134 and onwards.

    metadata_orig[:, 0] = metadata

    gender_group = [[], []]
    for i,ID in enumerate(metadata_orig[:, 0]):
        if metadata_orig[i, 2] == 'F':
            gender_group[1].append(int(ID))
        elif metadata_orig[i, 2] == 'M':
            gender_group[0].append(int(ID))
        else:
            gender_group[1].append(int(ID))


    reduced_group_list = copy.deepcopy(gender_group)

    for j in gender_group[0]:
        # print(int(j))
        if int(j) >= 134:
            # print("removing")
            reduced_group_list[0].remove(j)
        
    for j in gender_group[1]:
        if int(j) >= 134:
            reduced_group_list[1].remove(j)
    
    No_of_Nodes = len(orphannodes) + len(mapping)
    No_of_connected_nodes = len(mapping)
    
    return No_of_Nodes, No_of_connected_nodes, edges, orphannodes, gender_group, reduced_group_list

