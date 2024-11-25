import os
from glob import glob
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch 
from dgl.data.utils import save_graphs
import h5py

from histocartography.preprocessing import (
    VahadaneStainNormalizer,         # stain normalizer
    NucleiExtractor,                 # nuclei detector 
    DeepFeatureExtractor,            # feature extractor 
    KNNGraphBuilder,                 # kNN graph builder
    ColorMergedSuperpixelExtractor,  # tissue detector
    DeepFeatureExtractor,            # feature extractor
    RAGGraphBuilder,                 # build graph
    AssignmnentMatrixBuilder         # assignment matrix 
)
centroids_path="/data2/zlb/data/ext_feat/senucls_consep/cellbox/"
features_path="/data2/zlb/data/ext_feat/newfeat_resnet/consep-cell/"
image_label = 1
def build_cg(image_name):

        boxes=np.load(centroids_path+image_name)
        features=np.load(features_path+image_name)
        centroids=[]
        for i in range(len(boxes)):
            centroids.append([(boxes[i][0]+boxes[i][2])/2,(boxes[i][1]+boxes[i][3])/2])
        knn_graph_builder = KNNGraphBuilder(k=5, thresh=50, add_loc_feats=True)

        #print('np.array(centroids).shape',np.array(centroids).shape)  np.array(centroids).shape (17, 2)
        
        graph = knn_graph_builder.process(np.array(centroids), features)
        return graph, centroids

for image_name in os.listdir(centroids_path):
      cell_graph, nuclei_centroid = build_cg(image_name)
      
      cg_out = os.path.join('/data2/zlb/data/ext_feat/hovernet_consep/n*n_cell_graph_resnet/', image_name.replace('.npy', '.bin'))
      
      save_graphs(
                        filename=cg_out,
                        g_list=[cell_graph],
                        labels={"label": torch.tensor([image_label])}
                    )