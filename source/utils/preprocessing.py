# Following function pre-procceses the FriendshipNET dataset and returns the list of edges (Nx2 matrix).

import numpy as np

def get_friendshipnet_data():
    """Returns two parameters,
        First: N (Number of Nodes in FriendshipNet)
        Second: list of edges (Nx2 Numpy Matrix)"""
    metadata_orig = np.genfromtxt(
            "../data/metadata_2013.txt",
            delimiter="\t",
            # skip_header=True,
            # dtype="str",
        )
        # )[:,0]        #For only first column


    edges_orig = np.genfromtxt(
            "../data/Friendship-network_data_2013.csv",
            delimiter=" ", 
            # skip_header=True,
            # dtype="str",
        )

    mapping, relabels = np.unique((metadata_orig[:,0]), return_inverse=True)
    metadata = np.copy(metadata_orig)
    metadata[:,0] = relabels        #Relabels the data 

    edges = np.copy(edges_orig)
    for j in np.unique(metadata_orig[:, 0]):
        indices = np.where(edges[:,0] == j)
        edges[indices, 0] = np.where(mapping == j)

        indices = np.where(edges[:,1] == j)
        edges[indices, 1] = np.where(mapping == j)

    No_of_Users = len(np.unique(metadata[:, 0]))
    
    return No_of_Users, edges
