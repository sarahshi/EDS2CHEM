# %% 
import os, sys, math, time, random
import numpy as np
import pandas as pd
from skimage import segmentation, morphology, measure, transform, io


#own code:
import QEMSCAN_functions as qf
import QEMSCAN_plotting as qp
import PyTorch_autoencoder as pt
from PyTorch_autoencoder import Autoencoder, Shallow_Autoencoder, Tanh_Autoencoder


# %% 

path_parent = os.path.dirname(os.getcwd())
path_grandparent = os.path.dirname(path_parent)

output_dir = ["/OutputData/030122Run_NoNa"] 

for ii in range(len(output_dir)):
    if not os.path.exists(output_dir[ii]):
       os.makedirs(path_grandparent + output_dir[ii], exist_ok=True)

DF_NEW = pd.read_csv('./CleanedProfilesDF.csv', index_col= ['Comment', 'DataSet/Point'],)
samplename = DF_NEW.index.levels[0] 

names = ["K8_4mu_allphases", ""]
i = int(sys.argv[1]) - 1
name = names[i]

conc_data = np.load(path_grandparent + "/InputData/NPZ/" + name + ".npz", allow_pickle = True)
conc_map = conc_data["conc_map"]
mask = conc_data["data_mask"]
elements = conc_data["elements"]
conc_map = np.delete(conc_map, np.s_[0:2], axis = 2)
elements = np.delete(elements, np.s_[0:2], axis=0)
z_tanh = np.load(path_grandparent + output_dir[0] + '/' + name + "_tanh.npz", allow_pickle = True)['z']
labs_data = np.load(path_grandparent + output_dir[0] + "/" + name + '_clustered.npz', allow_pickle = True)
labs_tanh = labs_data['labs_tanh']


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

# %% 

loaded = np.load(path_grandparent + output_dir[0] + '/' + name + "_pca_mb_mcmc.npz", allow_pickle = True)
gauss_plag_pca = loaded['gauss_plag_pca']
m_ave, m_std = loaded['m_ave_mc'], loaded['m_std_mc']
b_ave, b_std = loaded['b_ave_mc'], loaded['b_std_mc']

# %% 

N = 50000

gauss_plag_an_mean_mc = np.zeros_like(gauss_plag_pca)
gauss_plag_an_mean_mc[:, :] = np.nan
gauss_plag_an_std_mc = np.zeros_like(gauss_plag_pca)
gauss_plag_an_std_mc[:, :] = np.nan

for i in range(0, np.shape(gauss_plag_pca)[0]): 
    for j in range(0, np.shape(gauss_plag_pca)[1]): 
        if np.isnan(gauss_plag_pca[i, j]) == False: 
            temp = gauss_plag_pca[i, j] * np.random.normal(m_ave, m_std, N) + np.random.normal(b_ave, b_std, N)
            gauss_plag_an_mean_mc[i, j] = np.nanmean(temp)
            gauss_plag_an_std_mc[i, j] = np.nanstd(temp)

np.savez(path_grandparent + output_dir[0] + "/" + name + '_anmcmc_proper_F.npz', gauss_plag_an_mean_mc = gauss_plag_an_mean_mc, gauss_plag_an_std_mc = gauss_plag_an_std_mc)
