#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:01:37 2024

@author: clarariegis

Want to create connectivity matrices threhsolded using different approaches. 

"""

#%% Libraries

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import matplotlib.colors as colors
import bct as bct
import _pickle as cPickle
import networkx as nx



#%% NONE

os.chdir('/Users/clarariegis/Desktop/Kings/Y3_2022_23/project/publication/data and code')

# test:
all_pcp_nr2 = pd.read_pickle("ts_nr.pickle")


#_________________________ NO threshold _________________________
 

#fig2, ax2 = plt.subplots(3, 4, figsize=(20, 15))
fig, ax = plt.subplot_mosaic("""
                              ABCD
                              EFGH
                              """, figsize=(20, 10))



corr_matrix = np.corrcoef(all_pcp_nr2.loc[0]['ts'].T)

nonzero = np.count_nonzero(~np.isnan(corr_matrix))

ax['A'].set_title('Unthresholded, D = 12100', size = size)

cmap = plt.get_cmap('coolwarm') # The colormap (red/blue)
pc = ax['A'].pcolormesh(corr_matrix,  # Plot the Rank biserial values. 
                    norm=colors.CenteredNorm(), # Center the colorbar around 0.
                    cmap=cmap)  # Colorbar previoulsy selected. 

cbar = plt.colorbar(pc, ax = ax['D'], location='left', pad = 0.9999999, aspect=10)                         # Add colorbar. 
#cbar.ax['D'].tick_params(labelsize=12)             # Make colobar ticks bigger.

#plt.gca().set_aspect('equal')  # Set aspect ratio to make the plot square

mini = pd.DataFrame(corr_matrix.flatten())
mini.plot(kind = "kde", ax = ax['H'], legend=False, color = "darkslategrey")


# Transform it in a links data frame (3 columns only):
links = pd.DataFrame(corr_matrix).stack().reset_index()
links.columns = ['var1', 'var2', 'value']
 

# ____ NETWORK ____
# Build your graph
G=nx.from_pandas_edgelist(links, 'var1', 'var2')
 
# Plot the network:
nx.draw_networkx(G, ax = ax['E'],pos = nx.random_layout(G, seed = 1), 
                 with_labels=False, node_color='black', 
                 node_size=15, edge_color='black', linewidths=0.01, 
                 font_size=15, width = 0.05)

corr_matrix[corr_matrix > 0] = 1

# # ____ GRAPH MEASURES ____

clustering = np.mean(bct.clustering_coef_bu(corr_matrix))
            
# # No need to average it since it's a global measure. 
char_path = bct.charpath(bct.distance_bin(corr_matrix))[0]

# betweenness = np.mean(bct.betweenness_bin(corr_matrix))


#_________________________ Proportional threshold _________________________
 
corr_matrix = np.corrcoef(all_pcp_nr2.loc[0]['ts'].T)
bct.threshold_proportional(corr_matrix, 0.05, copy = False)

corr_matrix[corr_matrix == 0] = np.nan

nonzero = np.count_nonzero(~np.isnan(corr_matrix))

ax['B'].set_title('T$_{prop.}$ = 5%, D = 600', size = size)

cmap = plt.get_cmap('coolwarm') # The colormap (red/blue)
pc = ax['B'].pcolormesh(corr_matrix,  # Plot the Rank biserial values. 
                    norm=colors.CenteredNorm(), # Center the colorbar around 0.
                    cmap=cmap)  # Colorbar previoulsy selected. 


mini = pd.DataFrame(corr_matrix.flatten())
mini.plot(kind = "kde", ax = ax['H'], legend=False, color = "darkred")




 
# Transform it in a links data frame (3 columns only):
links = pd.DataFrame(corr_matrix).stack().reset_index()
links.columns = ['var1', 'var2', 'value']
 
# Build your graph
G=nx.from_pandas_edgelist(links, 'var1', 'var2')
 
# Plot the network:
nx.draw_networkx(G, ax = ax['F'], pos = nx.random_layout(G, seed = 1), 
                 with_labels=False, node_color='black', 
                 node_size=15, edge_color='black', linewidths=0.01, 
                 font_size=15, width = 0.2)


corr_matrix[corr_matrix > 0] = 1
# # ____ GRAPH MEASURES ____

# clustering = np.mean(bct.clustering_coef_bu(corr_matrix))
            
# # # No need to average it since it's a global measure. 
# char_path = bct.charpath(bct.distance_bin(corr_matrix))[0]

# # betweenness = np.mean(bct.betweenness_bin(corr_matrix))






#_________________________________ Negative _________________________________

corr_matrix = np.corrcoef(all_pcp_nr2.loc[0]['ts'].T)

# Set the values below the threshold to NaN
corr_matrix[-0.3 < corr_matrix] = np.nan
# corr_matrix[1 == corr_matrix] = np.nan
nonzero = np.count_nonzero(~np.isnan(corr_matrix))
size = 15
ax['C'].set_title('T$_{abs.-}$ = -0.3, D = 42', size = size)

# Plot the modified correlation matrix
cmap = plt.get_cmap('coolwarm')  # The colormap (red/blue)
pc = ax['C'].pcolormesh(corr_matrix,
                    norm=colors.CenteredNorm(),  # Center the colorbar around 0.
                    cmap=cmap)  # Colormap previously selected.


mini = pd.DataFrame(corr_matrix.flatten())
mini.plot(kind = "kde", ax = ax['H'], legend=False, color = "navy")



# Transform it in a links data frame (3 columns only):
links = pd.DataFrame(corr_matrix).stack().reset_index()
links.columns = ['var1', 'var2', 'value']
 
# Build your graph
G=nx.from_pandas_edgelist(links, 'var1', 'var2')
 
# Plot the network:
nx.draw_networkx(G, ax = ax['G'], pos = nx.random_layout(G, seed = 1), 
                 with_labels=False, node_color='black', 
                 node_size=15, edge_color='black', 
                 linewidths=0.01, font_size=15, width = 0.3)


ax['H'].set_title('Edge weights distributions', size = size)

ax['H'].legend(labels=["Unthesholded" , "T$_{prop.}$", "T$_{abs.}$"])

# # ____ GRAPH MEASURES ____

# clustering = np.mean(bct.clustering_coef_bu(corr_matrix))
            
# # No need to average it since it's a global measure. 
# char_path = bct.charpath(bct.distance_bin(corr_matrix))[0]

# betweenness = np.mean(bct.betweenness_bin(corr_matrix))


# ax.tick_params(axis='both', labelsize=12)
# ax.tick_params(axis='x', labelsize=30)
# ax.tick_params(axis='y', labelsize=30)

plt.show()

# #_________________ Distribution-guided network  _________________


# corr_matrix = np.corrcoef(all_pcp_nr2.loc[0]['ts'].T)
# corr_matrix2 = np.corrcoef(all_pcp_nr2.loc[2]['ts'].T)

# std1 = np.std(corr_matrix)
# std2 = np.std(corr_matrix2)

# # Set the values below the threshold to NaN
# corr_matrix[-0.2 < corr_matrix] = np.nan
# # corr_matrix[1 == corr_matrix] = np.nan

# # Plot the modified correlation matrix
# cmap = plt.get_cmap('coolwarm')  # The colormap (red/blue)
# pc = ax['G'].pcolormesh(corr_matrix,
#                     norm=colors.CenteredNorm(),  # Center the colorbar around 0.
#                     cmap=cmap)  # Colormap previously selected.


# mini = pd.DataFrame(corr_matrix.flatten())
# mini.plot(kind = "kde", ax = ax['I'], legend=False, color = "navy")



# # Transform it in a links data frame (3 columns only):
# links = pd.DataFrame(corr_matrix).stack().reset_index()
# links.columns = ['var1', 'var2', 'value']
 
# # Build your graph
# G=nx.from_pandas_edgelist(links, 'var1', 'var2')
 
# # Plot the network:
# nx.draw_networkx(G, ax = ax['C'], pos = nx.random_layout(G, seed = 1), 
#                  with_labels=False, node_color='black', 
#                  node_size=15, edge_color='navy', 
#                  linewidths=0.01, font_size=15, width = 0.3)
# plt.show()













































