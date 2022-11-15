import numpy as np
metadata = np.genfromtxt(
        "../data/metadata_2013.txt",
        delimiter="\t",
        # skip_header=True,
        # dtype="str",
    )
    # )[:,0]        #For only first column

# No_of_Nodes = len(np.unique(metadata[:, 0]))

_, relabels = np.unique((metadata[:,0]), return_inverse=True)     
metadata[:,0] = relabels        #Relabels the IDs in metadata as continous integers between 0 and No_of_Nodes
