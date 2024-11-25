import numpy as np
from dgl.data.utils import load_graphs
from dgl.data.utils import load_labels
import matplotlib.pyplot as plt
import os
from scipy.io import savemat
graph_path='/data2/zlb/data/ext_feat/hovernet_consep/n*n_cell_graph_resnet/'
save_path="/data2/zlb/data/ext_feat/hovernet_consep/N*N_adj_resnet/"

for image_name in os.listdir(graph_path):

    (g0,),_= load_graphs(graph_path+image_name)

    n_node=g0.num_nodes()
    e=g0.edges()
    g=np.zeros([n_node,n_node])
    for i in range(len(e[0])):
        g[e[0][i]][e[1][i]]=1
        
        savemat(save_path+image_name.split(".")[0]+'.mat', {'N*N_adj':g})


