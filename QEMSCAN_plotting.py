# %% 
import numpy as np
import matplotlib.pyplot as plt
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import random
import sys
import time
import os 
from scipy.special import softmax

import skimage as ski
from sklearn import decomposition
from sklearn.mixture import BayesianGaussianMixture
from scipy.special import softmax
from skimage import segmentation, morphology, measure, transform, io
from sklearn.model_selection import train_test_split
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max

#own code:
import QEMSCAN_functions as qf
import QEMSCAN_plotting as qp
import PyTorch_autoencoder as pt
from PyTorch_autoencoder import Autoencoder, Shallow_Autoencoder, Tanh_Autoencoder


import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# %% 

def histplotting(conc_map, labels, name, appendix):

    unique = np.unique(labels)

    if len(unique[~np.isnan(unique)]) == 9: 
        fig, ax = plt.subplots(3, 3, figsize = (24, 24))

    if len(unique[~np.isnan(unique)]) == 12: 
        fig, ax = plt.subplots(4, 3, figsize = (24, 32))
        
    if len(unique[~np.isnan(unique)]) == 15: 
        fig, ax = plt.subplots(5, 3, figsize = (24, 40))

    if len(unique[~np.isnan(unique)]) == 18: 
        fig, ax = plt.subplots(6, 3, figsize = (24, 48))

    for ii in unique[~np.isnan(unique)]: 
            mg = conc_map[:, :, 0]
            al = conc_map[:, :, 1]
            si = conc_map[:, :, 2]
            ca = conc_map[:, :, 3]
            fe = conc_map[:, :, 4]
            cr = conc_map[:, :, 5]
            o  = conc_map[:, :, 6]

            boolean = labels == ii
            mg_ave = np.average(mg[boolean])
            al_ave = np.average(al[boolean])
            si_ave = np.average(si[boolean])
            ca_ave = np.average(ca[boolean])
            fe_ave = np.average(fe[boolean])
            cr_ave = np.average(cr[boolean])
            o_ave  = np.average(o[boolean])

            elements = ['Mg.1', 'Al.1', 'Si.1', 'Ca.1', 'Fe.1', 'Cr.1', 'O.1']
            elementalarray = np.array([mg_ave, al_ave, si_ave, ca_ave, fe_ave, cr_ave, o_ave])

            ax = ax.flatten()
            ax[int(ii)].bar(elements, elementalarray)
            ax[int(ii)].set_title(int(ii))
            ax[int(ii)].set_ylim([0, 300])
            ax[int(ii)].set_title(name+appendix+'label = '+str(ii))
            plt.tight_layout()

def plotting_explore(mineral_conc): 

    fig, ax = plt.subplots(2, 3, figsize = (24, 16))
    ax = ax.flatten()
    ax[0].scatter(mineral_conc[:, :, 2], mineral_conc[:, :, 0], s = 0.001, rasterized = True)
    ax[0].set_xlabel('Si')
    ax[0].set_ylabel('Mg')
    ax[1].scatter(mineral_conc[:, :,2], mineral_conc[:, :,1], s = 0.001, rasterized = True)
    ax[1].set_xlabel('Si')
    ax[1].set_ylabel('Al')
    ax[2].scatter(mineral_conc[:, :,2], mineral_conc[:, :,3], s = 0.001, rasterized = True)
    ax[2].set_xlabel('Si')
    ax[2].set_ylabel('Ca')
    ax[3].scatter(mineral_conc[:, :,2], mineral_conc[:, :,4], s = 0.001, rasterized = True)
    ax[3].set_xlabel('Si')
    ax[3].set_ylabel('Fe')
    ax[4].scatter(mineral_conc[:, :,2], mineral_conc[:, :,5], s = 0.001, rasterized = True)
    ax[4].set_xlabel('Si')
    ax[4].set_ylabel('Cr')
    ax[5].scatter(mineral_conc[:, :,2], mineral_conc[:, :,6], s = 0.001, rasterized = True)
    ax[5].set_xlabel('Si')
    ax[5].set_ylabel('O')
    plt.tight_layout()

def plag_pca(plag_conc, plag, std, name, plotting, saving):

    plag_data = plag_conc[:, :, 1:4]
    plag_data_mask = plag
    plag_array = plag_data[plag_data_mask.astype('bool')]
    plag_array_pca, plag_params_pca = qf.feature_normalisation(plag_array, return_params = True)

    plag_pca_fit = decomposition.PCA(n_components = 2)
    plag_pca_scores = plag_pca_fit.fit_transform(plag_array_pca)
    plag_pca_components = plag_pca_fit.components_

    plag_pca = qf.get_img(plag_pca_scores[:, 0], plag*1.)
    plag_pca_mean, plag_pca_std = np.mean(plag_pca_scores[:, 0]), np.std(plag_pca_scores[:, 0])

    gauss_plag_pca = qf.gaussian_filter(plag_pca, plag*1., std)
    gauss_plag_pca_mean, gauss_plag_pca_std = np.nanmean(gauss_plag_pca.ravel()), np.nanstd(gauss_plag_pca.ravel())


    if plotting == True: 
        fig, ax = plt.subplots(2, 4, figsize = (32, 16))
        ax = ax.flatten()
        ax[0].scatter(plag_array[:, 1], plag_array[:, 0], c = 'k', alpha = 0.5,s = 0.001, rasterized = True)
        ax[0].set_title(name+'_PL')
        ax[0].set_xlabel('Si')
        ax[0].set_ylabel('Al')
        ax[1].scatter(plag_array[:, 1], plag_array[:, 2], c = 'k', alpha = 0.5,s = 0.001, rasterized = True)
        ax[1].set_title(name+'_PL')
        ax[1].set_xlabel('Si')
        ax[1].set_ylabel('Ca')
        ax[2].set_title(name+'_PL')
        ax[2].scatter(plag_array[:, 0], plag_array[:, 2], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[2].set_xlabel('Al')
        ax[2].set_ylabel('Ca')

        ax[3].scatter(plag_pca_scores[:, 0], plag_pca_scores[:, 1], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[3].set_title(name+'_PL')
        ax[3].set_xlabel('PCA1')
        ax[3].set_ylabel('PCA2')

        ax[4].scatter(plag_pca_scores[:, 0], plag_array[:, 2], c = 'k', alpha = 0.5,s = 0.001, rasterized = True)
        ax[4].set_xlabel('PC1')
        ax[4].set_ylabel('Ca')
        ax[5].scatter(plag_pca_scores[:, 0], plag_array[:, 0], c = 'k', alpha = 0.5,s = 0.001, rasterized = True)
        ax[5].set_xlabel('PC1')
        ax[5].set_ylabel('Al')
        ax[6].scatter(plag_pca_scores[:, 0], plag_array[:, 1], c = 'k', alpha = 0.5,s = 0.001, rasterized = True)
        ax[6].set_xlabel('PC1')
        ax[6].set_ylabel('Si')
        fig.delaxes(ax[7])

        plt.tight_layout()
        if saving == True: 
            plt.savefig(name+'_plagxplot.pdf')

        fig, ax = plt.subplots(1, 2, figsize = (14, 6))
        y, x, _ = ax[0].hist(plag_pca_scores[:, 0], bins = 100, histtype='step', density = True)
        ax[0].vlines(plag_pca_mean - 2*plag_pca_std, 0, y.max(), 'k')
        ax[0].vlines(plag_pca_mean + 2*plag_pca_std, 0, y.max(), 'k')
        ax[0].set_title(name+'_PL')
        ax[0].set_xlabel('PC1 Score')
        ax[0].set_ylabel('Density')
        g_y, g_x, _ = ax[1].hist(gauss_plag_pca.ravel(), bins = 100, histtype='step', density = True)
        ax[1].vlines(gauss_plag_pca_mean - 2*gauss_plag_pca_std, 0, g_y.max(), 'k')
        ax[1].vlines(gauss_plag_pca_mean + 2*gauss_plag_pca_std, 0, g_y.max(), 'k')
        ax[1].set_title(name+'_PL')
        ax[1].set_xlabel('Gaussian Smoothed PC1 Score')
        ax[1].set_ylabel('Density')
        plt.tight_layout()
        if saving == True: 
            plt.savefig(name+'_plagCI.pdf')

    return plag_pca_scores, plag_pca, plag_pca_mean, plag_pca_std, gauss_plag_pca, gauss_plag_pca_mean, gauss_plag_pca_std

def cpx_pca(cpx_conc, cpx, std, name, plotting, saving):

    cpx_data = cpx_conc[:, :, [0, 2, 3]]
    cpx_data_mask = cpx
    cpx_array_orig = cpx_data[cpx_data_mask.astype('bool')]
    cpx_array_pca, cpx_params_pca = qf.feature_normalisation(cpx_array_orig, return_params = True)

    cpx_pca_fit = decomposition.PCA(n_components = 2)
    cpx_pca_scores = cpx_pca_fit.fit_transform(cpx_array_pca)
    cpx_pca_components = cpx_pca_fit.components_

    cpx_pca = qf.get_img(cpx_pca_scores[:, 0], cpx*1.)
    cpx_pca_mean, cpx_pca_std = np.mean(cpx_pca_scores[:, 0]), np.std(cpx_pca_scores[:, 0])
    gauss_cpx_pca = qf.gaussian_filter(cpx_pca, cpx*1., std)
    gauss_cpx_pca_mean, gauss_cpx_pca_std = np.nanmean(gauss_cpx_pca.ravel()), np.nanstd(gauss_cpx_pca.ravel())

    if plotting == True: 

        fig, ax = plt.subplots(2, 4, figsize = (32, 16))
        ax = ax.flatten()
        ax[0].scatter(cpx_array_orig[:, 1], cpx_array_orig[:, 0], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[0].set_title(name+'_CPX')
        ax[0].set_xlabel('Si')
        ax[0].set_ylabel('Mg')
        ax[1].scatter(cpx_array_orig[:, 1], cpx_array_orig[:, 2], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[1].set_title(name+'_CPX')
        ax[1].set_xlabel('Si')
        ax[1].set_ylabel('Ca')
        ax[2].scatter(cpx_array_orig[:, 0], cpx_array_orig[:, 2], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[2].set_title(name+'_CPX')
        ax[2].set_xlabel('Mg')
        ax[2].set_ylabel('Ca')
        ax[3].scatter(cpx_pca_scores[:, 0], cpx_pca_scores[:, 1], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[3].set_title(name+'_CPX')
        ax[3].set_xlabel('PCA1')
        ax[3].set_ylabel('PCA2')

        ax[4].scatter(cpx_pca_scores[:, 0], cpx_array_orig[:, 0],c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[4].set_xlabel('PC1')
        ax[4].set_ylabel('Mg')
        ax[5].scatter(cpx_pca_scores[:, 0], cpx_array_orig[:, 1], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[5].set_xlabel('PC1')
        ax[5].set_ylabel('Si')
        ax[6].scatter(cpx_pca_scores[:, 0], cpx_array_orig[:, 2], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[6].set_xlabel('PC1')
        ax[6].set_ylabel('Ca')
        fig.delaxes(ax[7])

        plt.tight_layout()
        if saving == True: 
            plt.savefig(name+'_cpxxplot.pdf')

        fig, ax = plt.subplots(1, 2, figsize = (14, 6))
        y, x, _ = ax[0].hist(cpx_pca_scores[:, 0], bins = 100, histtype='step', density = True)
        ax[0].vlines(cpx_pca_mean - 2*cpx_pca_std, 0, y.max(), 'k')
        ax[0].vlines(cpx_pca_mean + 2*cpx_pca_std, 0, y.max(), 'k')
        ax[0].set_title(name+'_CPX')
        ax[0].set_xlabel('PC1 Score')
        ax[0].set_ylabel('Density')
        g_y, g_x, _ = ax[1].hist(gauss_cpx_pca.ravel(), bins = 100, histtype='step', density = True)
        ax[1].vlines(gauss_cpx_pca_mean - 2*gauss_cpx_pca_std, 0, g_y.max(), 'k')
        ax[1].vlines(gauss_cpx_pca_mean + 2*gauss_cpx_pca_std, 0, g_y.max(), 'k')
        ax[1].set_title(name+'_CPX')
        ax[1].set_xlabel('Gaussian Smoothed PC1 Score')
        ax[1].set_ylabel('Density')
        plt.tight_layout()
        if saving == True: 
            plt.savefig(name+'_cpxCI.pdf')

    return cpx_pca_scores, cpx_pca, cpx_pca_mean, cpx_pca_std, gauss_cpx_pca, gauss_cpx_pca_mean, gauss_cpx_pca_std

def ol_pca(ol_conc, ol, std, name, plotting, saving):

    ol_data = ol_conc[:, :, [0, 2, 4]]
    ol_data_mask = ol
    ol_array_orig = ol_data[ol_data_mask.astype('bool')]
    ol_array_pca, ol_params_pca = qf.feature_normalisation(ol_array_orig, return_params = True)

    ol_pca_fit = decomposition.PCA(n_components = 2)
    ol_pca_scores = ol_pca_fit.fit_transform(ol_array_pca)
    ol_pca_components = ol_pca_fit.components_

    ol_pca = qf.get_img(ol_pca_scores[:, 0], ol*1.)
    ol_pca_mean, ol_pca_std = np.mean(ol_pca_scores[:, 0]), np.std(ol_pca_scores[:, 0])
    gauss_ol_pca = qf.gaussian_filter(ol_pca, ol*1., std)
    gauss_ol_pca_mean, gauss_ol_pca_std = np.nanmean(gauss_ol_pca.ravel()), np.nanstd(gauss_ol_pca.ravel())

    if plotting == True: 
        fig, ax = plt.subplots(2, 4, figsize = (32, 16))
        ax = ax.flatten()
        ax[0].scatter(ol_array_orig[:, 1], ol_array_orig[:, 0], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[0].set_title(name+'_OL')
        ax[0].set_xlabel('Si')
        ax[0].set_ylabel('Mg')
        ax[1].scatter(ol_array_orig[:, 1], ol_array_orig[:, 2], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[1].set_title(name+'_OL')
        ax[1].set_xlabel('Si')
        ax[1].set_ylabel('Fe')
        ax[2].scatter(ol_array_orig[:, 0], ol_array_orig[:, 2], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[2].set_title(name+'_OL')
        ax[2].set_xlabel('Mg')
        ax[2].set_ylabel('Fe')

        ax[3].scatter(ol_pca_scores[:, 0], ol_pca_scores[:, 1], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[3].set_title(name+'_OL')
        ax[3].set_xlabel('PCA1')
        ax[3].set_ylabel('PCA2')

        ax[4].scatter(ol_pca_scores[:, 0], ol_array_orig[:, 0], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[4].set_xlabel('PC1')
        ax[4].set_ylabel('Mg')
        ax[5].scatter(ol_pca_scores[:, 0], ol_array_orig[:, 1],  c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[5].set_xlabel('PC1')
        ax[5].set_ylabel('Si')
        ax[6].scatter(ol_pca_scores[:, 0], ol_array_orig[:, 2], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[6].set_xlabel('PC1')
        ax[6].set_ylabel('Fe')
        fig.delaxes(ax[7])

        plt.tight_layout()
        if saving == True: 
            plt.savefig(name+'_olxplot.pdf')

        fig, ax = plt.subplots(1, 2, figsize = (14, 6))
        y, x, _ = ax[0].hist(ol_pca_scores[:, 0], bins = 100, histtype='step', density = True)
        ax[0].vlines(ol_pca_mean - 2*ol_pca_std, 0, y.max(), 'k')
        ax[0].vlines(ol_pca_mean + 2*ol_pca_std, 0, y.max(), 'k')
        ax[0].set_title(name+'_OL')
        ax[0].set_xlabel('PC1 Score')
        ax[0].set_ylabel('Density')
        g_y, g_x, _ = ax[1].hist(gauss_ol_pca.ravel(), bins = 100, histtype='step', density = True)
        ax[1].vlines(gauss_ol_pca_mean - 2*gauss_ol_pca_std, 0, g_y.max(), 'k')
        ax[1].vlines(gauss_ol_pca_mean + 2*gauss_ol_pca_std, 0, g_y.max(), 'k')
        ax[1].set_title(name+'_OL')
        ax[1].set_xlabel('Gaussian Smoothed PC1 Score')
        ax[1].set_ylabel('Density')
        if saving == True: 
            plt.savefig(name+'_olCI.pdf')

    return ol_pca_scores, ol_pca, ol_pca_mean, ol_pca_std, gauss_ol_pca, gauss_ol_pca_mean, gauss_ol_pca_std


def glass_pca(glass_conc, glass, std, name, plotting, saving):

    glass_data = glass_conc[:, :, [0, 2, 4]]
    glass_data_mask = glass
    glass_array_orig = glass_data[glass_data_mask.astype('bool')]
    # glass_array_orig = glass_array_orig[glass_array_orig[:, 0] > 5]

    glass_array_pca, glass_params_pca = qf.feature_normalisation(glass_array_orig, return_params = True)

    glass_pca_fit = decomposition.PCA(n_components = 2)
    glass_pca_scores = glass_pca_fit.fit_transform(glass_array_pca)
    glass_pca_components = glass_pca_fit.components_

    glass_pca = qf.get_img(glass_pca_scores[:, 0], glass*1.)
    glass_pca_mean, glass_pca_std = np.mean(glass_pca_scores[:, 0]), np.std(glass_pca_scores[:, 0])
    gauss_glass_pca = qf.gaussian_filter(glass_pca, glass*1., std)
    gauss_glass_pca_mean, gauss_glass_pca_std = np.nanmean(gauss_glass_pca.ravel()), np.nanstd(gauss_glass_pca.ravel())

    if plotting == True: 
        fig, ax = plt.subplots(2, 4, figsize = (32, 16))
        ax = ax.flatten()
        ax[0].scatter(glass_array_orig[:, 1], glass_array_orig[:, 0], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[0].set_title(name+'_GL')
        ax[0].set_xlabel('Si')
        ax[0].set_ylabel('Mg')
        ax[1].scatter(glass_array_orig[:, 1], glass_array_orig[:, 2], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[1].set_title(name+'_GL')
        ax[1].set_xlabel('Si')
        ax[1].set_ylabel('Fe')
        ax[2].scatter(glass_array_orig[:, 0], glass_array_orig[:, 2], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[2].set_title(name+'_GL')
        ax[2].set_xlabel('Mg')
        ax[2].set_ylabel('Fe')

        ax[3].scatter(glass_pca_scores[:, 0], glass_pca_scores[:, 1], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[3].set_title(name+'_GL')
        ax[3].set_xlabel('PCA1')
        ax[3].set_ylabel('PCA2')
        ax[4].scatter(glass_pca_scores[:, 0], glass_array_orig[:, 0], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[4].set_xlabel('PC1')
        ax[4].set_ylabel('Mg')
        ax[5].scatter(glass_pca_scores[:, 0], glass_array_orig[:, 1], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[5].set_xlabel('PC1')
        ax[5].set_ylabel('Si')
        ax[6].scatter(glass_pca_scores[:, 0], glass_array_orig[:, 2], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[6].set_xlabel('PC1')
        ax[6].set_ylabel('Fe')
        fig.delaxes(ax[7])

        plt.tight_layout()
        if saving == True: 
            plt.savefig(name+'_glassxplot.pdf')

        fig, ax = plt.subplots(1, 2, figsize = (14, 6))
        y, x, _ = ax[0].hist(glass_pca_scores[:, 0], bins = 100, histtype='step', density = True)
        ax[0].vlines(glass_pca_mean - 2*glass_pca_std, 0, y.max(), 'k')
        ax[0].vlines(glass_pca_mean + 2*glass_pca_std, 0, y.max(), 'k')
        ax[0].set_title(name+'_GL')
        ax[0].set_xlabel('PC1 Score')
        ax[0].set_ylabel('Density')
        g_y, g_x, _ = ax[1].hist(gauss_glass_pca.ravel(), bins = 100, histtype='step', density = True)
        ax[1].vlines(gauss_glass_pca_mean - 2*gauss_glass_pca_std, 0, g_y.max(), 'k')
        ax[1].vlines(gauss_glass_pca_mean + 2*gauss_glass_pca_std, 0, g_y.max(), 'k')
        ax[1].set_title(name+'_GL')
        ax[1].set_xlabel('Gaussian Smoothed PC1 Score')
        ax[1].set_ylabel('Density')
        if saving == True: 
            plt.savefig(name+'_glassCI.pdf')

    return glass_pca_scores, glass_pca, glass_pca_mean, glass_pca_std, gauss_glass_pca, gauss_glass_pca_mean, gauss_glass_pca_std

# %% 

def segment_xl(mask, pca, x1, x2, y1, y2):

    xl = mask[y1:y2, x1:x2]
    xl_pca = pca[y1:y2, x1:x2]

    watershed_mask = xl.copy()
    distance = distance_transform_edt(watershed_mask)
    local_max = peak_local_max(distance, indices=False, footprint=np.ones((100, 100)), labels=watershed_mask)
    markers = morphology.label(local_max, connectivity = 2)
    labels_ws = segmentation.watershed(-distance, markers, mask=watershed_mask)
    labels_ws_plotting = labels_ws.astype('float')
    labels_ws_plotting[~xl] = np.nan

    # plt.figure(figsize = (12, 12))
    # plt.imshow(labels_ws_plotting, cmap = 'turbo')
    # plt.gca().set_aspect('equal', adjustable='box')

    properties = measure.regionprops(labels_ws)

    #calculate size and get indices in descending order of size
    size = [crystal.area for crystal in properties]
    sorted_indices = np.argsort(size)[::-1]

    slices = []
    for crystal in properties:        
        #slice to create image of only the particular region/crystal labelled
        img = crystal.image
        #get orientation
        angle = -(crystal.orientation/np.pi)*180
        #rotate
        rot_img = transform.rotate(img, angle, resize = True)
        #create binary image again; value of 0.5 is arbitrary
        rot_img[rot_img < 0.5] = 0
        rot_img[rot_img >= 0.5] = 1
        #apply padding so crystal isn't touching edge - to result in closed contours if applied at a later stage
        new_img = np.pad(rot_img, ((1,1), (1,1)), 'constant', constant_values = ((0,0),(0,0)) )
        new_img = new_img.astype('uint8')
        #append the rotated images 
        slices.append(new_img)

    sorted_slices = []
    #create list of crystals in descending order of area
    for i in range(len(sorted_indices)):
        sorted_slices.append(slices[sorted_indices[i]])

    #show number of crystals present
    # print("Number of crystals: " + str(len(sorted_slices)))

    return xl, xl_pca, new_img


def rotate(mask, xl, xl_pca, inputangle):
    properties = measure.regionprops(mask.astype(int))

    #calculate size and get indices in descending order of size
    size = [crystal.area for crystal in properties]
    sorted_indices = np.argsort(size)[::-1]

    slices = []

    for crystal in properties:        
        #slice to create image of only the particular region/crystal labelled
        img = xl #crystal.image
        pca = xl_pca
        #get orientation
        angle = -inputangle # -(-inputangle/np.pi)*180
        #rotate
        rot_img = transform.rotate(img, angle, resize = True)
        rot_pca = transform.rotate(xl_pca, angle, resize = True)
        #create binary image again; value of 0.5 is arbitrary
        rot_img[rot_img < 0.5] = 0
        rot_img[rot_img >= 0.5] = 1
        rot_pca[rot_img == 0] = np.nan

        #apply padding so crystal isn't touching edge - to result in closed contours if applied at a later stage
        new_img = np.pad(rot_img, ((1,1), (1,1)), 'constant', constant_values = ((0,0),(0,0)) )
        new_img = new_img.astype('uint8')
        #append the rotated images 
        slices.append(new_img)

    sorted_slices = []
    #create list of crystals in descending order of area
    for i in range(len(sorted_indices)):
        sorted_slices.append(slices[sorted_indices[i]])

    n = len(sorted_slices)
    fig, ax = plt.subplots(1, 2, figsize = (24, 12))
    ax = ax.flatten()

    ax[0].imshow(rot_img, interpolation = 'None')
    ax[1].imshow(rot_pca, interpolation = 'None')

    return rot_img, rot_pca


# %% 

def cpx_nmf(cpx_conc, cpx, std, plotting):

    cpx_data = cpx_conc[:, :, [0, 2, 3]]
    cpx_data_mask = cpx
    cpx_array_orig = cpx_data[cpx_data_mask.astype('bool')]
    cpx_array_nmf, cpx_params_nmf = qf.feature_normalisation(cpx_array_orig, return_params = True, mean_norm = False)

    cpx_nmf_fit = decomposition.NMF(n_components = 2, tol = 0.05)
    cpx_nmf_scores = cpx_nmf_fit.fit_transform(cpx_array_nmf)
    cpx_nmf_components = cpx_nmf_fit.components_

    cpx_nmf = qf.get_img(cpx_nmf_scores[:, 0], cpx*1.)
    cpx_nmf_mean, cpx_nmf_std = np.mean(cpx_nmf_scores[:, 0]), np.std(cpx_nmf_scores[:, 0])
    gauss_cpx_nmf = qf.gaussian_filter(cpx_nmf, cpx*1., std)
    gauss_cpx_nmf_mean, gauss_cpx_nmf_std = np.nanmean(gauss_cpx_nmf.ravel()), np.nanstd(gauss_cpx_nmf.ravel())

    if plotting == True:
        fig, ax = plt.subplots(2, 4, figsize = (32, 16))
        ax = ax.flatten()
        ax[0].scatter(cpx_nmf_scores[:, 0], cpx_nmf_scores[:, 1], s = 0.001, rasterized = True)
        ax[0].set_xlabel('NMF1')
        ax[0].set_ylabel('NMF2')
        ax[1].scatter(cpx_nmf_scores[:, 0], cpx_array_orig[:, 0], s = 0.001, rasterized = True)
        ax[1].set_xlabel('NMF1')
        ax[1].set_ylabel('Mg')
        ax[2].scatter(cpx_nmf_scores[:, 0], cpx_array_orig[:, 1], s = 0.001, rasterized = True)
        ax[2].set_xlabel('NMF1')
        ax[2].set_ylabel('Si')
        ax[3].scatter(cpx_nmf_scores[:, 0], cpx_array_orig[:, 2], s = 0.001, rasterized = True)
        ax[3].set_xlabel('NMF1')
        ax[3].set_ylabel('Ca')
        ax[4].scatter(cpx_array_orig[:, 1], cpx_array_orig[:, 0], s = 0.001, rasterized = True)
        ax[4].set_xlabel('Si')
        ax[4].set_ylabel('Mg')
        ax[5].scatter(cpx_array_orig[:, 1], cpx_array_orig[:, 2], s = 0.001, rasterized = True)
        ax[5].set_xlabel('Si')
        ax[5].set_ylabel('Ca')
        ax[6].scatter(cpx_array_orig[:, 0], cpx_array_orig[:, 2], s = 0.001, rasterized = True)
        ax[6].set_xlabel('Mg')
        ax[6].set_ylabel('Ca')
        plt.tight_layout()
        
        fig, ax = plt.subplots(1, 2, figsize = (14, 6))
        y, x, _ = ax[0].hist(cpx_nmf_scores[:, 0], bins = 100, histtype='step')
        ax[0].vlines(cpx_nmf_mean - 2*cpx_nmf_std, 0, y.max(), 'k')
        ax[0].vlines(cpx_nmf_mean + 2*cpx_nmf_std, 0, y.max(), 'k')
        g_y, g_x, _ = ax[1].hist(gauss_cpx_nmf.ravel(), bins = 100, histtype='step')
        ax[1].vlines(gauss_cpx_nmf_mean - 2*gauss_cpx_nmf_std, 0, g_y.max(), 'k')
        ax[1].vlines(gauss_cpx_nmf_mean + 2*gauss_cpx_nmf_std, 0, g_y.max(), 'k')

    return cpx_nmf_scores, cpx_nmf, cpx_nmf_mean, cpx_nmf_std, gauss_cpx_nmf, gauss_cpx_nmf_mean, gauss_cpx_nmf_std





def ol_nmf(ol_conc, ol, std, plotting):

    ol_data = ol_conc[:, :, [0, 2, 4]]
    ol_data_mask = ol
    ol_array_orig = ol_data[ol_data_mask.astype('bool')]
    ol_array_nmf, ol_params_nmf = qf.feature_normalisation(ol_array_orig, return_params = True, mean_norm = False)

    ol_nmf_fit = decomposition.NMF(n_components = 2, tol = 0.05)
    ol_nmf_scores = ol_nmf_fit.fit_transform(ol_array_nmf)
    ol_nmf_components = ol_nmf_fit.components_

    ol_nmf = qf.get_img(ol_nmf_scores[:, 0], ol*1.)
    ol_nmf_mean, ol_nmf_std = np.mean(ol_nmf_scores[:, 0]), np.std(ol_nmf_scores[:, 0])
    gauss_ol_nmf = qf.gaussian_filter(ol_nmf, ol*1., std)
    gauss_ol_nmf_mean, gauss_ol_nmf_std = np.nanmean(gauss_ol_nmf.ravel()), np.nanstd(gauss_ol_nmf.ravel())

    if plotting == True: 
        fig, ax = plt.subplots(2, 4, figsize = (32, 16))
        ax = ax.flatten()
        ax[0].scatter(ol_nmf_scores[:, 0], ol_nmf_scores[:, 1],  c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[0].set_xlabel('NMF1')
        ax[0].set_ylabel('NMF2')
        ax[1].scatter(ol_nmf_scores[:, 0], ol_array_orig[:, 0], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[1].set_xlabel('NMF1')
        ax[1].set_ylabel('Mg')
        ax[2].scatter(ol_nmf_scores[:, 0], ol_array_orig[:, 1], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[2].set_xlabel('NMF1')
        ax[2].set_ylabel('Si')
        ax[3].scatter(ol_nmf_scores[:, 0], ol_array_orig[:, 2], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[3].set_xlabel('NMF1')
        ax[3].set_ylabel('Fe')
        ax[4].scatter(ol_array_orig[:, 1], ol_array_orig[:, 0], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[4].set_xlabel('Si')
        ax[4].set_ylabel('Mg')
        ax[5].scatter(ol_array_orig[:, 1], ol_array_orig[:, 2], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[5].set_xlabel('Si')
        ax[5].set_ylabel('Fe')
        ax[6].scatter(ol_array_orig[:, 0], ol_array_orig[:, 2], c = 'k', alpha = 0.5, s = 0.001, rasterized = True)
        ax[6].set_xlabel('Mg')
        ax[6].set_ylabel('Fe')
        plt.tight_layout()
        
        fig, ax = plt.subplots(1, 2, figsize = (14, 6))
        y, x, _ = ax[0].hist(ol_nmf_scores[:, 0], bins = 100, histtype='step')
        ax[0].vlines(ol_nmf_mean - 2*ol_nmf_std, 0, y.max(), 'k')
        ax[0].vlines(ol_nmf_mean + 2*ol_nmf_std, 0, y.max(), 'k')
        ax[0].set_xlabel('PC1 Score')
        ax[0].set_ylabel('Counts')
        g_y, g_x, _ = ax[1].hist(gauss_ol_nmf.ravel(), bins = 100, histtype='step')
        ax[1].vlines(gauss_ol_nmf_mean - 2*gauss_ol_nmf_std, 0, g_y.max(), 'k')
        ax[1].vlines(gauss_ol_nmf_mean + 2*gauss_ol_nmf_std, 0, g_y.max(), 'k')


    return ol_nmf_scores, ol_nmf, ol_nmf_mean, ol_nmf_std, gauss_ol_nmf, gauss_ol_nmf_mean, gauss_ol_nmf_std




def plag_nmf(plag_conc, plag, std, plotting):

    plag_data = plag_conc[:, :, 1:4]
    plag_data_mask = plag
    plag_array = plag_data[plag_data_mask.astype('bool')]
    plag_array_nmf, plag_params_nmf = qf.feature_normalisation(plag_array, return_params = True, mean_norm = False)

    plag_nmf_fit = decomposition.NMF(n_components = 2, tol = 0.05)
    plag_nmf_scores = plag_nmf_fit.fit_transform(plag_array_nmf)
    plag_nmf_components = plag_nmf_fit.components_

    plag_nmf = qf.get_img(plag_nmf_scores[:, 0], plag*1.)
    plag_nmf_mean, plag_nmf_std = np.mean(plag_nmf_scores[:, 0]), np.std(plag_nmf_scores[:, 0])
    gauss_plag_nmf = qf.gaussian_filter(plag_nmf, plag*1., std)
    gauss_plag_nmf_mean, gauss_plag_nmf_std = np.nanmean(gauss_plag_nmf.ravel()), np.nanstd(gauss_plag_nmf.ravel())

    if plotting == True: 

        fig, ax = plt.subplots(2, 4, figsize = (32, 16))
        ax = ax.flatten()
        ax[0].scatter(plag_nmf_scores[:, 0], plag_nmf_scores[:, 1], s = 0.001, rasterized = True)
        ax[0].set_xlabel('NMF1')
        ax[0].set_ylabel('NMF2')
        ax[1].scatter(plag_nmf_scores[:, 0], plag_array[:, 2], s = 0.001, rasterized = True)
        ax[1].set_xlabel('NMF1')
        ax[1].set_ylabel('Ca')
        ax[2].scatter(plag_nmf_scores[:, 0], plag_array[:, 0], s = 0.001, rasterized = True)
        ax[2].set_xlabel('NMF1')
        ax[2].set_ylabel('Al')
        ax[3].scatter(plag_nmf_scores[:, 0], plag_array[:, 1], s = 0.001, rasterized = True)
        ax[3].set_xlabel('NMF1')
        ax[3].set_ylabel('Si')
        ax[4].scatter(plag_array[:, 1], plag_array[:, 0], s = 0.001, rasterized = True)
        ax[4].set_xlabel('Si')
        ax[4].set_ylabel('Al')
        ax[5].scatter(plag_array[:, 1], plag_array[:, 2], s = 0.001, rasterized = True)
        ax[5].set_xlabel('Si')
        ax[5].set_ylabel('Ca')
        ax[6].scatter(plag_array[:, 0], plag_array[:, 2], s = 0.001, rasterized = True)
        ax[6].set_xlabel('Al')
        plt.tight_layout()
        
        fig, ax = plt.subplots(1, 2, figsize = (14, 6))
        y, x, _ = ax[0].hist(plag_nmf_scores[:, 0], bins = 100, histtype='step')
        ax[0].vlines(plag_nmf_mean - 2*plag_nmf_std, 0, y.max(), 'k')
        ax[0].vlines(plag_nmf_mean + 2*plag_nmf_std, 0, y.max(), 'k')
        g_y, g_x, _ = ax[1].hist(gauss_plag_nmf.ravel(), bins = 100, histtype='step')
        ax[1].vlines(gauss_plag_nmf_mean - 2*gauss_plag_nmf_std, 0, g_y.max(), 'k')
        ax[1].vlines(gauss_plag_nmf_mean + 2*gauss_plag_nmf_std, 0, g_y.max(), 'k')
        plt.tight_layout()

    return plag_nmf_scores, plag_nmf, plag_nmf_mean, plag_nmf_std, gauss_plag_nmf, gauss_plag_nmf_mean, gauss_plag_nmf_std

