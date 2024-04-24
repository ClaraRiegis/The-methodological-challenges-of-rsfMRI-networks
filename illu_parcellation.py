#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 11:50:25 2024

@author: clarariegis
"""

#%% Libraries

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import matplotlib.colors as colors
import bct as bct
import _pickle as cPickle
import networkx as nx
from scipy.ndimage import zoom
import os
import nibabel as nib


#%% FCs

os.chdir('/Users/clarariegis/Desktop/Kings/Y3_2022_23/project/atlas')

fig, ax = plt.subplot_mosaic("""
                              CBAL
                              GFEM
                              JIHK
                              """, figsize=(20, 15))
                              
size = 15           
                              
# ________________________ Voxel ________________________


atlas = nib.load("Caltech_0051456_func_preproc.nii")
test = atlas.get_fdata()
ax['L'].imshow(np.rot90(test[:,:,28, 50]), cmap='grey' )
ax['L'].tick_params(axis='both', colors='white')


resampled_data = zoom(test, (0.2,0.2,0.2,1))


arr_2d = resampled_data.reshape(-1, resampled_data.shape[-1])

# Create a mask of non-zero rows
non_zero_rows_mask = ~np.all(arr_2d < 10**-17, axis=1)
# Filter the array based on the mask
arr_filtered = arr_2d[non_zero_rows_mask]



corr_matrix = pd.DataFrame(arr_filtered.T).corr(method='pearson').values


# ax['L'].imshow(np.rot90(resampled_data[:,:,4, 50]), cmap='magma' )
# ax['L'].tick_params(axis='both', colors='white')



cmap = plt.get_cmap('coolwarm') # The colormap (red/blue)
pc = ax['M'].pcolormesh(corr_matrix,  # Plot the Rank biserial values. 
                    norm=colors.CenteredNorm(), # Center the colorbar around 0.
                    cmap=cmap)  # Colorbar previoulsy selected. 




mini = pd.DataFrame(corr_matrix.flatten())
mini.plot(kind = "kde", ax = ax['K'], legend=False, color = "darkred")



# ________________________ cc400 ________________________

atlas = nib.load("atl cc400/cc400_roi_atlas.nii")
test = atlas.get_fdata()
ax['A'].imshow(np.rot90(atlas.get_fdata()[:,:,30]), cmap='grey' )
ax['A'].tick_params(axis='both', colors='white')


cc400 = np.loadtxt("atl cc400/Caltech_0051456_rois_cc400.1D")
corr_matrix = pd.DataFrame(cc400).corr(method='pearson').values

cmap = plt.get_cmap('coolwarm') # The colormap (red/blue)
pc = ax['E'].pcolormesh(corr_matrix,  # Plot the Rank biserial values. 
                    norm=colors.CenteredNorm(), # Center the colorbar around 0.
                    cmap=cmap)  # Colorbar previoulsy selected. 




mini = pd.DataFrame(corr_matrix.flatten())
mini.plot(kind = "kde", ax = ax['K'], legend=False, color = "darkorange")




# Transform it in a links data frame (3 columns only):
links = pd.DataFrame(corr_matrix).stack().reset_index()
links.columns = ['var1', 'var2', 'value']
 
# Build your graph
G=nx.from_pandas_edgelist(links, 'var1', 'var2')
 
# Plot the network:
nx.draw_networkx(G, ax = ax['H'], pos = nx.random_layout(G, seed = 1), 
                 with_labels=False, node_color='black', 
                 node_size=15, edge_color='black', linewidths=0.01, 
                 font_size=15, width = 0.01)




# ________________________ cc200 ________________________

atlas = nib.load("atl cc200/cc200_roi_atlas.nii")
test = atlas.get_fdata()

ax['B'].imshow(np.rot90(atlas.get_fdata()[:,:,30]), cmap='grey' )
ax['B'].tick_params(axis='both', colors='white')


cc200 = np.loadtxt("atl cc200/Caltech_0051456_rois_cc200.1D")
corr_matrix = pd.DataFrame(cc200).corr(method='pearson').values

cmap = plt.get_cmap('coolwarm') # The colormap (red/blue)
pc = ax['F'].pcolormesh(corr_matrix,  # Plot the Rank biserial values. 
                    norm=colors.CenteredNorm(), # Center the colorbar around 0.
                    cmap=cmap)  # Colorbar previoulsy selected. 




mini = pd.DataFrame(corr_matrix.flatten())
mini.plot(kind = "kde", ax = ax['K'], legend=False, color = "darkgreen")



# Transform it in a links data frame (3 columns only):
links = pd.DataFrame(corr_matrix).stack().reset_index()
links.columns = ['var1', 'var2', 'value']
 
# Build your graph
G=nx.from_pandas_edgelist(links, 'var1', 'var2')
 
# Plot the network:
nx.draw_networkx(G, ax = ax['I'], pos = nx.random_layout(G, seed = 1), 
                 with_labels=False, node_color='black', 
                 node_size=15, edge_color='black', linewidths=0.01, 
                 font_size=15, width = 0.01)




# ________________________ atl AAL ________________________


atlas = nib.load("atl AAL/aal_roi_atlas.nii")
test = atlas.get_fdata()

ax['C'].imshow(np.rot90(atlas.get_fdata()[:,:,30]), cmap='grey' )
ax['C'].tick_params(axis='both', colors='white')


cc200 = np.loadtxt("atl AAL/Caltech_0051456_rois_aal.1D")
corr_matrix = pd.DataFrame(cc200).corr(method='pearson').values

cmap = plt.get_cmap('coolwarm') # The colormap (red/blue)
pc = ax['G'].pcolormesh(corr_matrix,  # Plot the Rank biserial values. 
                        norm=colors.CenteredNorm(), # Center the colorbar around 0.
                        cmap=cmap)  # Colorbar previoulsy selected. 




mini = pd.DataFrame(corr_matrix.flatten())
mini.plot(kind = "kde", ax = ax['K'], legend=False, color = "indigo")



# Transform it in a links data frame (3 columns only):
links = pd.DataFrame(corr_matrix).stack().reset_index()
links.columns = ['var1', 'var2', 'value']
 
# Build your graph
G=nx.from_pandas_edgelist(links, 'var1', 'var2')
 
# Plot the network:
nx.draw_networkx(G, ax = ax['J'], pos = nx.random_layout(G, seed = 1), 
                 with_labels=False, node_color='black', 
                 node_size=15, edge_color='black', linewidths=0.01, 
                 font_size=15, width = 0.01)


ax['K'].legend(labels=["Voxel" , "CC400", "CC200", "AAL"])


ax['A'].set_title('CC400', size = size)
ax['B'].set_title('CC200', size = size)
ax['C'].set_title('AAL', size = size)
ax['L'].set_title('Voxels', size = size)






