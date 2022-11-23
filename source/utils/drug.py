import numpy as np


def isSymmetric(mat):
    if mat.shape[0] != mat.shape[1]:
        return False

    return np.all(mat == mat.T)


def symmetrize1(mat):
    n = mat.shape[0]
    new_mat = []
    for i in range(n):
        row = mat[i,:].reshape(-1)
        col = mat[:,i].reshape(-1)
        new_row = (row+col>0).astype(np.int32)
        new_mat.append(new_row)

    return np.array(new_mat)


def symmetrize2(mat):
    n = mat.shape[0]
    new_mat = []
    for i in range(n):
        row = mat[i,:].reshape(-1)
        col = mat[:, i].reshape(-1)
        new_row = (row*col>0).astype(np.int32)
        new_mat.append(new_row)

    return np.array(new_mat)


def __get_adj_mat_and_groups(path):
    adj_mat = []
    get_new_label = {}
    new_label = 0

    with open(path+'DRUGNET.csv') as f:
        f.readline()
        for line in f:
            line = line.strip().split(',')
            line = [int(x) for x in line]
            old_label = line[0]
            line = np.array(line[1:], dtype=np.int32)
            get_new_label[old_label] = new_label
            new_label+=1
            adj_mat.append(line)

    adj_mat = np.array(adj_mat, dtype=np.int32)
    adj_symm1 = symmetrize1(adj_mat)
    adj_symm2 = symmetrize2(adj_mat)

    # male # female
    gender = [[], [], []]

    # aff_am # latino # white/other
    ethinicity = [[], [], []]

    with open(path+'DRUGATTR.csv') as f:
        f.readline()
        for line in f:
            line = line.strip().split(',')[:-1]
            line = [int(x) for x in line]
            node, eth, gend = get_new_label[line[0]], line[1], line[2]

            if eth==2:
                ethinicity[0].append(node)

            elif eth==3:
                ethinicity[1].append(node)

            else:
                ethinicity[2].append(node)

            if gend==1:
                gender[0].append(node)

            elif gend==2:
                gender[1].append(node)

            else:
                gender[2].append(node)


    ## remove isolated vertices ##
    new_label_1 = {}
    n_lab = 0
    new_as1 = []
    num_vertices = len(adj_symm1)
    removed_vertices = []

    for i in range(num_vertices):
        if np.any(adj_symm1[i,:]!=0):
            new_row = adj_symm1[i].reshape(-1)
            new_as1.append(new_row)
            new_label_1[i] = n_lab
            n_lab += 1
        else:
            removed_vertices.append(i)

    new_as1 = np.array(new_as1, dtype=np.int32)

    delta = [False if i in removed_vertices else True
             for i in range(num_vertices)]

    new_as1 = new_as1[:, delta]

    #print(isSymmetric(new_as1))
    #print(new_as1.shape)
    #print(new_label_1.values())


    new_gender = [[], [], []]
    for g_num, gend in enumerate(gender):
        for v in gend:
            if v in new_label_1.keys():
                new_gender[g_num].append(new_label_1[v])

    new_ethinicity = [[], [], []]
    for e_num, eth in enumerate(ethinicity):
        for v in eth:
            if v in new_label_1.keys():
                new_ethinicity[e_num].append(new_label_1[v])
    f = lambda x: len(x)
    #print(list(map(f,new_gender)))
    #print(list(map(f, new_ethinicity)))

    return new_as1, new_gender, new_ethinicity
    #return adj_mat, adj_symm1, adj_symm2, gender, ethinicity



def _get_adj_mat_and_groups(path):
    adj_mat = []
    get_new_label = {}
    new_label = 0

    with open(path+'DRUGNET.csv') as f:
        f.readline()
        for line in f:
            line = line.strip().split(',')
            line = [int(x) for x in line]
            old_label = line[0]
            line = np.array(line[1:], dtype=np.int32)
            get_new_label[old_label] = new_label
            new_label+=1
            adj_mat.append(line)

    adj_mat = np.array(adj_mat, dtype=np.int32)
    adj_symm1 = symmetrize1(adj_mat)
    adj_symm2 = symmetrize2(adj_mat)

    # male # female
    gender = [[], [], []]

    # aff_am # latino # white/other
    ethinicity = [[], [], []]

    with open(path+'DRUGATTR.csv') as f:
        f.readline()
        for line in f:
            line = line.strip().split(',')[:-1]
            line = [int(x) for x in line]
            node, eth, gend = get_new_label[line[0]], line[1], line[2]

            if eth==2:
                ethinicity[0].append(node)

            elif eth==3:
                ethinicity[1].append(node)

            else:
                ethinicity[2].append(node)

            if gend==1:
                gender[0].append(node)

            elif gend==2:
                gender[1].append(node)

            else:
                gender[2].append(node)

    return adj_mat, adj_symm1, adj_symm2, gender, ethinicity


def get_adj_mat_and_groups(path):
    """
    Params:
        path (str): path to the folder containing DRUGNET.csv and
                    DRUGATTR.csv.

    Returns:
        adj_mat (np.ndarray): adjacency matrix for the directed graph
                              Is not symmetric.
        adj_symm1 (np.ndarray): symmetrization of the above using or
        adj_symm2 (np.ndarray): symmetrization of the above using and
        gender (list): list of list of nodes forming gender groups
                       (last position contains gender unspecified)
        ethinicity (list): list of list of nodes forming ethinic groups
    """
    adj_mat = []
    get_new_label = {}
    _get_temp_label = {}
    new_label = 0
    _temp_label = 0
    isolated_vertices = []

    with open(path+'DRUGNET.csv') as f:
        f.readline()
        for line in f:
            line = line.strip().split(',')
            line = [int(x) for x in line]
            old_label = line[0]
            line = np.array(line[1:], dtype=np.int32)
            _get_temp_label[old_label] = _temp_label
            _temp_label += 1
            if np.all(line==0):
                isolated_vertices.append(old_label)
                continue

            get_new_label[old_label] = new_label
            new_label+=1
            adj_mat.append(line)

    _num_vertices = len(adj_mat[0])
    iv_new = [_get_temp_label[i] for i in isolated_vertices]

    bool_labels = [False if x in iv_new
                   else True for x in range(_num_vertices)]

    adj_mat = np.array(adj_mat, dtype=np.int32)
    adj_mat = adj_mat[:, bool_labels]
    adj_symm1 = symmetrize1(adj_mat)
    adj_symm2 = symmetrize2(adj_mat)

    # male # female
    gender = [[], [], []]

    # aff_am # latino # white/other
    ethinicity = [[], [], []]

    with open(path+'DRUGATTR.csv') as f:
        f.readline()
        for line in f:
            line = line.strip().split(',')[:-1]
            line = [int(x) for x in line]
            node, eth, gend = line[0], line[1], line[2]

            if node in isolated_vertices:
                continue

            if eth==2:
                ethinicity[0].append(get_new_label[node])

            elif eth==3:
                ethinicity[1].append(get_new_label[node])

            else:
                ethinicity[2].append(get_new_label[node])

            if gend==1:
                gender[0].append(get_new_label[node])

            elif gend==2:
                gender[1].append(get_new_label[node])

            else:
                gender[2].append(get_new_label[node])

    return adj_mat, adj_symm1, adj_symm2, gender, ethinicity


if __name__ == "__main__":
    #adj_mat, a1, a2, gen, eth = __get_adj_mat_and_groups('../../data/drug/')
    adj_mat, gen, eth = __get_adj_mat_and_groups('../../data/drug/')
    print(adj_mat.shape)
    print(isSymmetric(adj_mat))

