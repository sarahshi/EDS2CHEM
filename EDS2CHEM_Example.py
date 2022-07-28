# %% -*- coding: utf-8 -*-
""" Created on July 20, 2022 // @author: Sarah Shi """

# %% 
import os, sys, math, time, random
import mc3
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader

import skimage 
from skimage import segmentation, morphology, measure, transform, io
from skimage.feature import peak_local_max
import scipy.interpolate as interpolate
from scipy.ndimage import distance_transform_edt
from scipy.special import softmax
from sklearn import decomposition
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import train_test_split

#own code:
import QEMSCAN_functions as qf
import QEMSCAN_plotting as qp
import MCMC as mcmc
import MC3_Plotting as mc3plot

import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc, cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 18})
plt.rcParams['pdf.fonttype'] = 42

# %% 

output_dir = ["./FIGURES/", "./PLOTFILES/", "./NPZFILES/", "./LOGFILES/"] 
for ii in range(len(output_dir)):
    if not os.path.exists(output_dir[ii]):
        os.makedirs(output_dir[ii])

path_parent = os.path.dirname(os.getcwd())
path_grandparent = os.path.dirname(path_parent)

output_dir = ["/OutputData/030122Run_NoNa"] 

for ii in range(len(output_dir)):
    if not os.path.exists(output_dir[ii]):
       os.makedirs(path_grandparent + output_dir[ii], exist_ok=True)

DF_NEW = pd.read_csv('./CleanedProfilesDF.csv', index_col= ['Comment', 'DataSet/Point'],)

samplename = DF_NEW.index.levels[0]

# %%

name = 'K8_4mu_allphases'

conc_data = np.load(path_grandparent + "/InputData/NPZ/" + name + ".npz", allow_pickle = True)
conc_map = conc_data["conc_map"]
mask = conc_data["data_mask"]
elements = conc_data["elements"]
conc_map = np.delete(conc_map, np.s_[0], axis = 2)
elements = np.delete(elements, np.s_[0], axis=0)

z_tanh = np.load(path_grandparent + output_dir[0] + '/' + name + "_tanh.npz", allow_pickle = True)['z']
labs_data = np.load(path_grandparent + output_dir[0] + "/" + name + '_clustered_15.npz', allow_pickle = True)
labs_tanh = labs_data['labs_tanh']

# mc3_output1 = np.load(path_grandparent + "/Code/030122Run_NoNa/K8_PL1_25e5.npz", allow_pickle = True)
# mc3_output2 = np.load(path_grandparent + "/Code/030122Run_NoNa/K8_PL2_25e5.npz", allow_pickle = True)
# mc3_output3 = np.load(path_grandparent + "/Code/030122Run_NoNa/K8_PL3_25e5.npz", allow_pickle = True)
# mc3_output4 = np.load(path_grandparent + "/Code/030122Run_NoNa/K8_PL4_25e5.npz", allow_pickle = True)

pnames   = ['ax','ay', 'bx', 'by', 'ww', 'm', 'b']
texnames = ['ax','ay', 'bx', 'by', 'ww', 'm', 'b']

# %%


labs_tanh_update = labs_tanh

labs_tanh_update[labs_tanh == 3.0] = np.nan
labs_tanh_update[labs_tanh == 4.0] = 0.0
labs_tanh_update[labs_tanh == 6.0] = 2.0
labs_tanh_update[labs_tanh == 8.0] = 0.0
labs_tanh_update[labs_tanh == 9.0] = 0.0
labs_tanh_update[labs_tanh == 10.0] = 1.0
labs_tanh_update[labs_tanh == 11.0] = np.nan
labs_tanh_update[labs_tanh == 12.0] = np.nan
labs_tanh_update[labs_tanh == 13.0] = 1.0
labs_tanh_update[labs_tanh == 14.0] = np.nan

glass = labs_tanh_update == 2.0
plag = labs_tanh_update == 1.0
spinel = labs_tanh_update == 5.0
cpx = labs_tanh_update == 7.0
olivine = labs_tanh_update == 0.0

glass = morphology.remove_small_objects(glass, 20)
plag = morphology.remove_small_objects(plag, 20)
spinel = morphology.remove_small_objects(spinel, 20)
cpx = morphology.remove_small_objects(cpx, 20)
olivine = morphology.remove_small_objects(olivine, 20)

unique_update, counts_update = np.unique(labs_tanh_update, return_counts = True)

# %% 

glass_conc = conc_map.copy()
plag_conc = conc_map.copy()
cpx_conc = conc_map.copy()
spinel_conc = conc_map.copy()
olivine_conc = conc_map.copy()

glass_conc[~glass] = np.nan
plag_conc[~plag] = np.nan
spinel_conc[~spinel] = np.nan
cpx_conc[~cpx] = np.nan
olivine_conc[~olivine] = np.nan


# %%% 

name = 'K8_4mu'

plag_pca_scores, plag_pca, plag_pca_mean, plag_pca_std, gauss_plag_pca, gauss_plag_pca_mean, gauss_plag_pca_std = qp.plag_pca(plag_conc, plag, 1, name, True, True)
cpx_pca_scores, cpx_pca, cpx_pca_mean, cpx_pca_std, gauss_cpx_pca, gauss_cpx_pca_mean, gauss_cpx_pca_std = qp.cpx_pca(cpx_conc, cpx, 1, name, True, True)
ol_pca_scores, ol_pca, ol_pca_mean, ol_pca_std, gauss_ol_pca, gauss_ol_pca_mean, gauss_ol_pca_std = qp.ol_pca(olivine_conc, olivine, 1, name, True, True)
glass_pca_scores, glass_pca, glass_pca_mean, glass_pca_std, gauss_glass_pca, gauss_glass_pca_mean, gauss_glass_pca_std = qp.glass_pca(glass_conc, glass, 1, name, True, True)


# %% 

# anmcmc = np.load(path_grandparent + output_dir[0] + "/K8_4mu_allphases_anmcmc_proper_F.npz")
# an_mean_mcmc = anmcmc['gauss_plag_an_mean_mc']
# an_std_mcmc = anmcmc['gauss_plag_an_std_mc']

# gauss_plag_an_mean = np.nanmean(an_mean_mcmc)
# gauss_plag_an_std = np.nanstd(an_mean_mcmc)

# gauss_plag_std_mean = np.nanmean(an_std_mcmc)
# gauss_plag_std_std = np.nanstd(an_std_mcmc)

N = 256
glasscmap = ListedColormap(['#FFFFFF00', '#F9C300'])
plagcmap = ListedColormap(['#FFFFFF00', '#009988'])
spinelcmap = ListedColormap(['#FFFFFF00', '#2E2DCE'])
cpxcmap = ListedColormap(['#FFFFFF00', '#8E021F'])
olivinecmap = ListedColormap(['#FFFFFF00', '#666633'])

plag_cmap_arr = np.ones((N, 4))
plag_cmap_arr[:, 0] = np.linspace(204/256, 0/256, N)
plag_cmap_arr[:, 1] = np.linspace(238/256, 153/256, N)
plag_cmap_arr[:, 2] = np.linspace(255/256, 136/256, N)
plag_cmap = ListedColormap(plag_cmap_arr)

cpx_cmap_arr = np.ones((N, 4))
cpx_cmap_arr[:, 0] = np.linspace(255/256, 142/256, N)
cpx_cmap_arr[:, 1] = np.linspace(204/256, 2/256, N)
cpx_cmap_arr[:, 2] = np.linspace(204/256, 31/256, N)
cpx_cmap = ListedColormap(cpx_cmap_arr)

ol_cmap_arr = np.ones((N, 4))
ol_cmap_arr[:, 0] = np.linspace(239/256, 102/256, N)
ol_cmap_arr[:, 1] = np.linspace(238/256, 102/256, N)
ol_cmap_arr[:, 2] = np.linspace(187/256, 51/256, N)
ol_cmap = ListedColormap(ol_cmap_arr)

glass_cmap_arr = np.ones((N, 4))
glass_cmap_arr[:, 0] = np.linspace(255/256, 255/256, N)
glass_cmap_arr[:, 1] = np.linspace(255/256, 174/256, N)
glass_cmap_arr[:, 2] = np.linspace(255/256, 13/256, N)
glass_cmap = ListedColormap(glass_cmap_arr)

bw_cmap_arr = np.ones((N, 4))
bw_cmap_arr[:, 0] = np.linspace(12/256, 228/256, N)
bw_cmap_arr[:, 1] = np.linspace(123/256, 34/256, N)
bw_cmap_arr[:, 2] = np.linspace(220/256, 17/256, N)
bw_cmap = ListedColormap(bw_cmap_arr)


# %% 

xl1, xl_pca1, new_img1 = qp.segment_xl(plag, gauss_plag_pca, 1000, 1400, 100, 625)
xl2, xl_pca2, new_img2 = qp.segment_xl(plag, gauss_plag_pca, 3000, 3550, 2000, 2450)
xl3, xl_pca3, new_img3 = qp.segment_xl(plag, gauss_plag_pca, 2650, 3050, 2450, 2850)
xl4, xl_pca4, new_img4 = qp.segment_xl(plag, gauss_plag_pca, 1950, 2300, 1275, 1750)

# %% 

k8_plag1 = DF_NEW.loc['K8_plag1_rtoc']

distance1 = distance_transform_edt(xl1.copy())
local_max1 = peak_local_max(distance_transform_edt(xl1.copy()), indices=False, footprint=np.ones((100, 100)), labels=xl1.copy())
markers1 = morphology.label(local_max1, connectivity = 2)

labels_ws1 = segmentation.watershed(-distance1, markers1, mask=xl1.copy())
labels_ws_plotting1 = labels_ws1.astype('float')
labels_ws_plotting1[~xl1] = np.nan

labels_ws_plotting1[labels_ws_plotting1 == 0] = np.nan
labels_ws_plotting1[labels_ws_plotting1 == 1] = np.nan
labels_ws_plotting1[labels_ws_plotting1 == 9] = np.nan
labels_ws_plotting1[labels_ws_plotting1 == 11] = np.nan
labels_ws_plotting1[labels_ws_plotting1 == 12] = np.nan
labels_ws_plotting1[labels_ws_plotting1 == 14] = np.nan

new_mask1 = ~np.isnan(labels_ws_plotting1)
xl_pca1[~new_mask1] = np.nan

rot_img1, rot_pca1 = qp.rotate(new_mask1, xl1, xl_pca1, 150)

lim1 = rot_pca1[300:400, 150:410]
plt.figure(figsize = (8, 2))
plt.imshow(lim1, interpolation = 'None', origin = 'upper')
plt.plot(60, 30, 'ro')
plt.plot(225, 32, 'ro')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

an_lim1 = np.nanmean(lim1, axis = 0)
an_lim1 = an_lim1[~np.isnan(an_lim1)]
x_lim1 = np.linspace(len(an_lim1), 1, len(an_lim1)) * 4

lim1 = -rot_pca1[300:400, 150:410]
inputmatrix1 = lim1
plag1 = k8_plag1[k8_plag1['MgO']*10000 < 2500]

indparams_1 = [inputmatrix1, plag1]

# %% 

# MCMC parameter setup -- a x and y coordinates (left point), b x and y coordinates (right point), window width, slope, intercept
# pnames   = ["ax","ay", "bx", "by", "ww", "m", "b"]
# change step sizes to better suit the ranges

params1 = np.array([65, 25, 225, 26, 10, 0.78, 0.8])
pstep1 = 0.075 * params1
pstep1[0:5] = 1.
pstep1[-2] = 0.05*params1[-2]

# minimum and maximum parameter values 
pmin1 =   np.array([60, 10, 200, 10,  1, 0.00, 0.00])
pmax1 =   np.array([70, 30, 230, 30, 20, 1.00, 1.00])

files = 'K8_PL1'
an_1 = plag1.Anorthite.values / 100
mc3_output1 = mcmc.MCMC(an_1, an_1*0.05, params = params1, pstep = pstep1, 
                pmin = pmin1, pmax = pmax1, indparams = indparams_1, 
                log='K8_PL1_25e5.log', savefile='K8_PL1_25e5.npz')
fig1 = mc3plot.trace(mc3_output1['posterior'], title = files, zchain=mc3_output1['zchain'], burnin=mc3_output1['burnin'], pnames=texnames, savefile = files+'_trace.pdf')
fig2 = mc3plot.histogram(mc3_output1['posterior'], title = files, pnames=texnames, bestp=mc3_output1['bestp'], savefile = files+'_histogram.pdf', quantile=0.683)
fig3 = mc3plot.pairwise(mc3_output1['posterior'], title = files, pnames=texnames, bestp=mc3_output1['bestp'], savefile = files+'_pairwise.pdf')



# %% 

# N = 100000

# m1 = np.random.normal(bestp1[-2], 1*stdp1[-2], N)
# m2 = np.random.normal(bestp2[-2], 1*stdp2[-2], N)
# m3 = np.random.normal(bestp3[-2], 1*stdp3[-2], N)
# m4 = np.random.normal(bestp4[-2], 1*stdp4[-2], N)
# m_t = np.concatenate([m1, m2, m3, m4])

# b1 = np.random.normal(bestp1[-1], 1*stdp1[-1], N)
# b2 = np.random.normal(bestp2[-1], 1*stdp2[-1], N)
# b3 = np.random.normal(bestp3[-1], 1*stdp3[-1], N)
# b4 = np.random.normal(bestp4[-1], 1*stdp4[-1], N)
# b_t = np.concatenate([b1, b2, b3, b4])

# m_ave_mc, m_std_mc = np.mean(m_t), np.std(m_t)
# b_ave_mc, b_std_mc = np.mean(b_t), np.std(b_t)


# np.savez(path_grandparent + output_dir[0] + "/" + name + '_pca_mb_mcmc.npz', gauss_plag_pca = -gauss_plag_pca, m_ave_mc = m_ave_mc, m_std_mc = m_std_mc, b_ave_mc = b_ave_mc, b_std_mc = b_std_mc)
