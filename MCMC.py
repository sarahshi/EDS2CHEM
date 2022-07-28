# %% 
import os, sys, math, time, random
import mc3
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate

# %% 

def projection(P, inputmatrix, plag, Nvalues = 7): 
    """The projection function takes the required inputs of the initial input parameters of     
    the (ax, ay), (bx, by) coordinates of the search rectangle, window width ww for a moving 
    average of the data, and slope and intercept scaling factors of the input PCA array. 
    The function outputs the scaled data that is then input into the MCMC sampling function."""

    ax, ay, bx, by, ww, slope, intercept = P[0:Nvalues]
    a = np.array([int(ax), int(ay)])
    b = np.array([int(bx), int(by)])
    ww = int(ww)
    dist_emp = plag["Distance (µ)"]
    an_emp = plag["Anorthite"] / 100

    if a[1] < b[1]: 

        h = np.abs(a[1] - b[1]) + 10
        bb_l = np.array([a[0], a[1] - h])
        bb_r = np.array([b[0], b[1] + h])

        line_size = np.array([np.abs(b[0] - a[0]), np.abs(b[1] - a[1])])
        bb_size = np.array([np.abs(bb_r[0] - bb_l[0]), np.abs(bb_r[1] - bb_l[1])])
        m = np.abs(b[1]-a[1]) / np.abs(b[0]-a[0])

        theta1 = np.arctan(m)

        x_search = np.linspace(bb_l[0], bb_r[0], np.abs(bb_r[0]-bb_l[0])+1)
        y_search = np.linspace(bb_l[1], bb_r[1], np.abs(bb_r[1]-bb_l[1])+1)
        X_search, Y_search = np.meshgrid(x_search, y_search)
        X = (X_search.reshape((np.prod(X_search.shape),))).astype("int")
        Y = (Y_search.reshape((np.prod(Y_search.shape),))).astype("int")

        isabove = lambda p, a, b: np.cross(p-a, b-a) < 0
        xy = np.column_stack((X, Y))
        above = isabove(xy, a, b)

        xyabove = xy[above, :]
        xybelow = xy[~above, :]

        allocate = np.zeros(len(xy))
        for i in range(0, len(xy)): 
            allocate[i] = inputmatrix[xy[i, 1], xy[i, 0]]

        pcaabove = allocate[above]
        pcabelow = allocate[~above]

        rect_below = xybelow[xybelow[:, 1] < a[1]]
        tri_below = xybelow[xybelow[:, 1] >= a[1]]
        rect_above = xyabove[xyabove[:, 1] > b[1]]
        tri_above = xyabove[xyabove[:, 1] <= b[1]]

        pca_rect_below = pcabelow[xybelow[:, 1] < a[1]]
        pca_tri_below = pcabelow[xybelow[:, 1] >= a[1]]
        pca_rect_above = pcaabove[xyabove[:, 1] > b[1]]
        pca_tri_above = pcaabove[xyabove[:, 1] <= b[1]]

        onehyp = 1 / np.sin(theta1)
        dy = np.abs((b[1]-a[1]) / (b[1]-a[1]))
        dx = np.abs((b[0]-a[0]) / (b[1]-a[1]))
        total_hyp = np.sqrt((bb_size[1]**2) + ((bb_size[0]+(dx*(bb_size[1]-line_size[1])))**2))

        rect_xout_below = np.abs(rect_below[:, 1] - a[1]) *  dx # (dx/dy)
        rect_x_below = np.abs(rect_below[:, 0] - a[0]) + rect_xout_below
        tri_x_below = tri_below[:, 0] - a[0] - ((tri_below[:, 1] - a[1]) * dx)

        rect_adj_below = np.cos(theta1) * rect_x_below
        tri_adj_below = np.cos(theta1) * tri_x_below

        addhyp_rect_below = np.abs(rect_below[:, 1]-bb_l[1]) * onehyp
        addhyp_tri_below = np.abs(tri_below[:, 1]-bb_l[1]) * onehyp

        rect_adj_b = rect_adj_below + addhyp_rect_below
        tri_adj_b = tri_adj_below + addhyp_tri_below

        xy_concat_below = np.concatenate([rect_below, tri_below])
        projdist_concat_below = np.concatenate([rect_adj_b, tri_adj_b])
        pca_concat_below = np.concatenate([pca_rect_below, pca_tri_below])

        df_below = pd.DataFrame(columns = ["X", "Y", "PD", "PCA"])
        df_below["X"] = xy_concat_below[:, 0]
        df_below["Y"] = xy_concat_below[:, 1]
        df_below["PD"] = projdist_concat_below 
        df_below["PCA"] = pca_concat_below

        rect_xout_above = np.abs(rect_above[:, 1] - b[1]) * dx #  (dx/dy)
        rect_x_above = np.abs(rect_above[:, 0] - b[0]) + rect_xout_above
        tri_x_above = np.abs(tri_above[:, 0] - b[0]) - (np.abs(tri_above[:, 1] - b[1]) * dx)

        rect_adj_above = np.cos(theta1) * rect_x_above
        tri_adj_above = np.cos(theta1) * tri_x_above

        onehyp = 1 / np.sin(theta1)
        addhyp_rect_above = np.abs(rect_above[:, 1]-bb_r[1]) * onehyp
        addhyp_tri_above = np.abs(tri_above[:, 1]-bb_r[1]) * onehyp

        rect_adj_a = rect_adj_above + addhyp_rect_above
        tri_adj_a = tri_adj_above + addhyp_tri_above

        xy_concat_above = np.concatenate([rect_above, tri_above])
        projdist_concat_above = np.concatenate([rect_adj_a, tri_adj_a])
        pca_concat_above = np.concatenate([pca_rect_above, pca_tri_above])

        df_above = pd.DataFrame(columns = ["X", "Y", "PD", "SPD", "PCA"])
        df_above["X"] = xy_concat_above[:, 0]
        df_above["Y"] = xy_concat_above[:, 1]
        df_above["PD"] = total_hyp - projdist_concat_above
        df_above["PCA"] = pca_concat_above

        df_i = pd.concat([df_above, df_below])
        df_i = df_i.sort_values(by=["PD"], axis=0, ascending=True, inplace=False, na_position="last")
        df_i["SPD"] = df_i["PD"] - np.nanmin(df_i["PD"])
        df_i["SPD"] = df_i["SPD"] * 4
        df_i = df_i[df_i["PCA"].notna()]

        # https://stackoverflow.com/questions/11352047/finding-moving-average-from-data-points-in-python 
        dist_val = df_i.SPD.ravel()
        cumsum_dist = np.cumsum(np.insert(dist_val, 0, 0))
        dist_v = (cumsum_dist[ww:] - cumsum_dist[:-ww]) / ww
        pca_val = df_i.PCA.ravel()
        cumsum_pca = np.cumsum(np.insert(pca_val, 0, 0))
        pca_v = (cumsum_pca[ww:] - cumsum_pca[:-ww]) / ww

        df_ma = pd.DataFrame(columns = ["PD_i", "PCA_i"])
        df_ma["PD_i"] = dist_v - dist_v[0]
        df_ma["PCA_i"] = pca_v

        dist_int = dist_emp
        pca_int = interpolate.interp1d(df_ma.PD_i, df_ma.PCA_i)(dist_int)
        df = pd.DataFrame(columns = ["D", "PCA"])
        df["D"] = dist_int
        df["PCA"] = pca_int

    elif a[1] > b[1]: 

        h = np.abs(a[1] - b[1]) + 10
        bb_l = np.array([a[0], a[1] - h])
        bb_r = np.array([b[0], b[1] + h])

        line_size = np.array([np.abs(b[0] - a[0]), np.abs(b[1] - a[1])])
        bb_size = np.array([np.abs(bb_r[0] - bb_l[0]), np.abs(bb_r[1] - bb_l[1])])
        m = np.abs(b[1]-a[1]) / np.abs(b[0]-a[0])

        theta1 = np.arctan(m)

        x_search = np.linspace(bb_l[0], bb_r[0], np.abs(bb_r[0]-bb_l[0])+1)
        y_search = np.linspace(bb_l[1], bb_r[1], np.abs(bb_r[1]-bb_l[1])+1)
        X_search, Y_search = np.meshgrid(x_search, y_search)
        X = (X_search.reshape((np.prod(X_search.shape),))).astype("int")
        Y = (Y_search.reshape((np.prod(Y_search.shape),))).astype("int")

        isabove = lambda p, a, b: np.cross(p-a, b-a) < 0
        xy = np.column_stack((X, Y))
        above = isabove(xy, a, b)

        xyabove = xy[above, :]
        xybelow = xy[~above, :]

        allocate = np.zeros(len(xy))
        for i in range(0, len(xy)): 
            allocate[i] = inputmatrix[xy[i, 1], xy[i, 0]]

        pcaabove = allocate[above]
        pcabelow = allocate[~above]

        rect_below = xybelow[xybelow[:, 1] < b[1]]
        tri_below = xybelow[xybelow[:, 1] >= b[1]]
        rect_above = xyabove[xyabove[:, 1] > a[1]]
        tri_above = xyabove[xyabove[:, 1] <= a[1]]

        pca_rect_below = pcabelow[xybelow[:, 1] < b[1]]
        pca_tri_below = pcabelow[xybelow[:, 1] >= b[1]]
        pca_rect_above = pcaabove[xyabove[:, 1] > a[1]]
        pca_tri_above = pcaabove[xyabove[:, 1] <= a[1]]

        onehyp = 1 / np.sin(theta1)
        dy = np.abs((b[1]-a[1]) / (b[1]-a[1]))
        dx = np.abs((b[0]-a[0]) / (b[1]-a[1]))
        total_hyp = np.sqrt((bb_size[1]**2) + ((bb_size[0]+(dx*(bb_size[1]-line_size[1])))**2))

        rect_xout_below = np.abs(rect_below[:, 1] - b[1]) * dx
        rect_x_below = np.abs(rect_below[:, 0] - b[0]) + rect_xout_below
        tri_x_below = np.abs(tri_below[:, 0] - b[0]) - ((tri_below[:, 1] - b[1]) * dx)
        rect_adj_below = np.cos(theta1) * rect_x_below
        tri_adj_below = np.cos(theta1) * tri_x_below

        addhyp_rect_below = np.abs(rect_below[:, 1]-bb_l[1]) * onehyp
        addhyp_tri_below = np.abs(tri_below[:, 1]-bb_l[1]) * onehyp

        rect_adj_b = rect_adj_below + addhyp_rect_below
        tri_adj_b = tri_adj_below + addhyp_tri_below

        xy_concat_below = np.concatenate([rect_below, tri_below])
        projdist_concat_below = np.concatenate([rect_adj_b, tri_adj_b])
        pca_concat_below = np.concatenate([pca_rect_below, pca_tri_below])

        df_below = pd.DataFrame(columns = ["X", "Y", "PD", "PCA"])
        df_below["X"] = xy_concat_below[:, 0]
        df_below["Y"] = xy_concat_below[:, 1]
        df_below["PD"] = total_hyp - projdist_concat_below 
        df_below["PCA"] = pca_concat_below

        rect_xout_above = np.abs(rect_above[:, 1] - a[1]) * dx 
        rect_x_above = np.abs(rect_above[:, 0] - a[0]) + rect_xout_above
        tri_x_above = np.abs(tri_above[:, 0] - a[0]) - (np.abs(tri_above[:, 1] - a[1]) * dx)

        rect_adj_above = np.cos(theta1) * rect_x_above
        tri_adj_above = np.cos(theta1) * tri_x_above

        onehyp = 1 / np.sin(theta1)
        addhyp_rect_above = np.abs(rect_above[:, 1]-bb_r[1]) * onehyp
        addhyp_tri_above = np.abs(tri_above[:, 1]-bb_r[1]) * onehyp

        rect_adj_a = rect_adj_above + addhyp_rect_above
        tri_adj_a = tri_adj_above + addhyp_tri_above

        xy_concat_above = np.concatenate([rect_above, tri_above])
        projdist_concat_above = np.concatenate([rect_adj_a, tri_adj_a])
        pca_concat_above = np.concatenate([pca_rect_above, pca_tri_above])

        df_above = pd.DataFrame(columns = ["X", "Y", "PD", "PCA"])
        df_above["X"] = xy_concat_above[:, 0]
        df_above["Y"] = xy_concat_above[:, 1]
        df_above["PD"] = total_hyp - projdist_concat_above
        df_above["PCA"] = pca_concat_above

        df_i = pd.concat([df_above, df_below])
        df_i = df_i.sort_values(by=["PD"], axis=0, ascending=True, inplace=False, na_position="last")
        df_i["SPD"] = df_i["PD"] - np.nanmin(df_i["PD"])
        df_i["SPD"] = df_i["SPD"] * 4
        df_i = df_i[df_i["PCA"].notna()]

        dist_val = df_i.SPD.ravel()
        cumsum_dist = np.cumsum(np.insert(dist_val, 0, 0))
        dist_v = (cumsum_dist[ww:] - cumsum_dist[:-ww]) / ww
        pca_val = df_i.PCA.ravel()
        cumsum_pca = np.cumsum(np.insert(pca_val, 0, 0))
        pca_v = (cumsum_pca[ww:] - cumsum_pca[:-ww]) / ww

        df_ma = pd.DataFrame(columns = ["PD_i", "PCA_i"])
        df_ma["PD_i"] = dist_v - dist_v[0]
        df_ma["PCA_i"] = pca_v

        dist_int = dist_emp
        pca_int = interpolate.interp1d(df_ma.PD_i, df_ma.PCA_i)(dist_int)
        df = pd.DataFrame(columns = ["D", "PCA"])
        df["D"] = dist_int
        df["PCA"] = pca_int

    elif a[1] == b[1]: 

        h = 10
        lim = inputmatrix[a[1]-h:b[1]+h, a[0]:b[0]]
        pca_int = np.nanmean(lim, axis = 0)
        pca = pca_int[~np.isnan(pca_int)]
        dist = np.linspace(1, len(pca), len(pca)) * 4 

        df = pd.DataFrame(columns = ["D", "PCA"])
        df["D"] = dist
        df["PCA"] = pca

    mx = df.PCA.values * slope
    barr = np.ones_like(mx) * intercept
    scaled_data = mx + barr

    return scaled_data

# %% 

def MCMC(data, uncert, params, pstep, pmin, pmax, indparams, log, savefile):
    
    """The MCMC function takes the required inputs and runs the Monte Carlo-Markov Chain. The function outputs the 
    mc3_output which contains all of the best fit parameters and standard deviations."""

    inputmatrix, plag = indparams[0:2]
    func = projection

    priorlow = np.array([5.0, 5.0, 5.0, 5.0, 0.0, 0.1, 0.1])
    priorup  = np.array([5.0, 5.0, 5.0, 5.0, 0.0, 0.1, 0.1])

    pnames   = ["ax","ay", "bx", "by", "ww", "m", "b"]
    texnames = ["ax","ay", "bx", "by", "ww", "m", "b"]

    mc3_output = mc3.sample(data, uncert, func=func, params=params, indparams=indparams, 
        pmin=pmin, pmax=pmax, priorlow=priorlow, priorup=priorup, 
        pstep = pstep, pnames=pnames, texnames=texnames, sampler="snooker", rms=False,
        nsamples=2.25e5, nchains=9, ncpu=3, burnin=2500, thinning=1,
        leastsq="trf", chisqscale=False, grtest=True, grbreak=1.01, grnmin=0.5,
        hsize=10, kickoff="normal", wlike=False, plots=False, log=log, savefile=savefile)

    return mc3_output