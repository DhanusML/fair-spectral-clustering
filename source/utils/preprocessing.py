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
            # "../../data/metadata_2013.txt",
            "../data/friendship_net/metadata_2013.txt",
            delimiter="\t",
            # skip_header=True,
            dtype="str",
        )
        # )[:,0]        #For only first column


    edges_orig = np.genfromtxt(
            # "../../data/Friendship-network_data_2013.csv",
            "../data/friendship_net/Friendship-network_data_2013.csv",
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


    class_group = [[], [], [], [], [], [], [], [], []]
    for i,ID in enumerate(metadata_orig[:, 0]):
        if metadata_orig[i, 1] == '2BIO1':
            class_group[0].append(int(ID))        #2BIO1

        elif metadata_orig[i, 1] == '2BIO2':
            class_group[1].append(int(ID))         #2BIO2

        elif metadata_orig[i, 1] == '2BIO3':
            class_group[2].append(int(ID))         #2BIO3

        elif metadata_orig[i, 1] == 'MP*1':
            class_group[3].append(int(ID))         #MP*1

        elif metadata_orig[i, 1] == 'MP*2':
            class_group[4].append(int(ID))         #MP*2

        elif metadata_orig[i, 1] == 'PSI*':
            class_group[5].append(int(ID))         #PSI*

        elif metadata_orig[i, 1] == 'PC':
            class_group[6].append(int(ID))         #PC

        elif metadata_orig[i, 1] == 'PC*':
            class_group[7].append(int(ID))         #PC*

        elif metadata_orig[i, 1] == 'MP':
            class_group[8].append(int(ID))         #MP
            # print("me")
        else:
            print(int(ID))




    reduced_gender_group_list = copy.deepcopy(gender_group)

    for j in gender_group[0]:
        # print(int(j))
        if int(j) >= 134:
            # print("removing")
            reduced_gender_group_list[0].remove(j)
        
    for j in gender_group[1]:
        if int(j) >= 134:
            reduced_gender_group_list[1].remove(j)



    reduced_class_group_list = copy.deepcopy(class_group)
    for k in range(9):
        for j in class_group[k]:
            if int(j) >= 134:
                reduced_class_group_list[k].remove(j) 




    No_of_Nodes = len(orphannodes) + len(mapping)
    No_of_connected_nodes = len(mapping)

    edges = edges.astype(np.int32)

    listofedges = edges.tolist()
    for (index, element) in enumerate(edges[:,0]):
        j = edges[index,1]
        for k in range(edges.shape[0]):
            if all(edges[k]==[j, element]):
                break

        else:
            listofedges.append([j, element])

    edges = np.array(listofedges, dtype=np.int32)

    return No_of_Nodes, No_of_connected_nodes, edges, orphannodes, gender_group, reduced_gender_group_list, reduced_class_group_list

