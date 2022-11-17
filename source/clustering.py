from algorithms.spectral_clustering import *
from utils.preprocessing import get_friendshipnet_data
from utils.utils import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
_, n, edges, _, _, gender_groups  = get_friendshipnet_data()



edges = edges.astype(int)


# clusters = normalizedSC(n, 3, edges)
# clusters = unnormalizedSC(n, 3, edges)
clusters = normalizedConSC(n, 2, edges, gender_groups)

visualize(n, edges, clusters)