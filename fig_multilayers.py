#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:42:03 2024

@author: clarariegis
"""
import random

inter_edges = 800
intra_edges = 400

dict_layer1 = {"SourceNode" : np.concatenate((np.arange(1, inter_edges + 1), 
                                              np.arange(1, intra_edges + 1))),
               "TargetNode" : np.concatenate((np.repeat(np.arange(1, (inter_edges/4)+1),4),
                                             random.sample(range(1, inter_edges), intra_edges))),
               "SourceLayer": np.repeat("layer1",inter_edges+intra_edges), 
               "TargetLayer": np.concatenate((np.repeat("layer2",inter_edges), 
                                              np.repeat("layer1",intra_edges)))}

inter_edges = 200
intra_edges = 150

dict_layer2 = {"SourceNode" : np.concatenate((np.arange(1, inter_edges + 1), 
                                              np.arange(1, intra_edges + 1))),
               "TargetNode" : np.concatenate((np.repeat(np.arange(1, (inter_edges/4)+1),4),
                                             random.sample(range(1, inter_edges), intra_edges))),
               "SourceLayer": np.repeat("layer1",inter_edges+intra_edges), 
               "TargetLayer": np.concatenate((np.repeat("layer3",inter_edges), 
                                              np.repeat("layer2",intra_edges)))}


dict_layer3 = {"SourceNode" : np.arange(1, 41), 
               "TargetNode" : np.repeat(np.arange(1, (40/5)+1),5),
               "SourceLayer": np.repeat("layer3",40), 
               "TargetLayer": np.repeat("layer3",40)}

df_layer1 = pd.DataFrame(dict_layer1)
df_layer2 = pd.DataFrame(dict_layer2)
df_layer3 = pd.DataFrame(dict_layer3)

os.chdir('/Users/clarariegis/Desktop/Cambridge')
MyLayers = pd.concat([df_layer1,df_layer2,df_layer3])
MyLayers.to_csv('MyLayers.tsv', sep='\t', index=False)
