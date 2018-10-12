import numpy as np
from math import pi
import matplotlib.pyplot as plt


def CrossCompare(dict0, dict1, box_size, plot=False):
    imgs0 = np.squeeze(dict0['data'])
    imgs1 = np.squeeze(dict1['data'])

    k_min = 2*pi/box_size[0]
    k_max = 2*pi/box_size[0]*imgs0.shape[1]/2
    n_k_bin = 20

    pCl_real0, Pk_err_true0, ell0, bins, n_mode = calculate_pseudo_Cl(imgs0[0],
                                                                      imgs0[1],
                                                                      box_size,
                                                                      ell_min=k_min,
                                                                      ell_max=k_max,
                                                                      n_bin=n_k_bin,
                                                                      logspaced=True)

    pCl_real1, Pk_err_true1, ell1, bins, n_mode = calculate_pseudo_Cl(imgs1[0],
                                                                      imgs1[1],
                                                                      box_size,
                                                                      ell_min=k_min,
                                                                      ell_max=k_max,
                                                                      n_bin=n_k_bin,
                                                                      logspaced=True)


    power = {}
    power[dict0['name']] = [ell0, pCl_real0]
    power[dict1['name']] = [ell1, pCl_real1]

    x0, y0 = power[dict0['name']]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.scatter(ell0, pCl_real0, label=dict0['name'])
        plt.scatter(ell1, pCl_real1, label=dict1['name'])
        ax.legend()

    else:
        return power


def AutoCompare(dict0, dict1, box_size, plot=False):
    img0 = np.squeeze(dict0['data'])
    img1 = np.squeeze(dict1['data'])

    k_min = 2*pi/box_size[0]
    k_max = 2*pi/box_size[0]*img0.shape[1]/2
    n_k_bin = 20


    pCl_real0, Pk_err_true0, ell0, bins, n_mode = calculate_pseudo_Cl(img0,
                                                                      img0,
                                                                      box_size,
                                                                      ell_min=k_min,
                                                                      ell_max=k_max,
                                                                      n_bin=n_k_bin,
                                                                      logspaced=True)

    pCl_real1, Pk_err_true1, ell1, bins, n_mode = calculate_pseudo_Cl(img1,
                                                                      img1,
                                                                      box_size,
                                                                      ell_min=k_min,
                                                                      ell_max=k_max,
                                                                      n_bin=n_k_bin,
                                                                      logspaced=True)

    power = {}
    power[dict0['name']] = [ell0, pCl_real0]
    power[dict1['name']] = [ell1, pCl_real1]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xscale('log')
        ax.set_yscale('log')
        for i, key in enumerate(power):
            plt.plot(power[key][0], power[key][1], label=key)
        ax.legend()
        return power

    else:
        return power


def PowerDistance(imgs0, imgs1, box_size, batch=True):
    '''
    Required arguments:
    imgs0          Array of size (N, 1, H, W) or (N, H, W)
    imgs1          same as imgs0
    box_size        Physical size (L1, L2) of the maps.

    Returns:
    TODO
    '''
    assert(len(imgs0[0]) == len(imgs1[0]))

    imgs0 = np.squeeze(imgs0)
    imgs1 = np.squeeze(imgs1)

    pCl_real0, _, _, _, _ = calculate_pseudo_Cl(imgs0[0],
                                                imgs0[0],
                                                box_size)
    pCl_real1, _, _, _, _ = calculate_pseudo_Cl(imgs1[0],
                                                imgs1[0],
                                                box_size)
    dim0 = len(pCl_real0)
    dim1 = len(pCl_real0)
    assert(dim0 == dim1)

    square_diffs = np.zeros(shape=(len(imgs0[0]), dim0))

    for i, val in enumerate(imgs0):
        pCl_real0, _, _, _, _ = calculate_pseudo_Cl(imgs0[i],
                                                    imgs0[i],
                                                    box_size)
        pCl_real1, _, _, _, _ = calculate_pseudo_Cl(imgs1[i],
                                                    imgs1[i],
                                                    box_size)

        square_diff = np.square(np.subtract(pCl_real0, pCl_real1))

        square_diffs[i] = square_diff

    mse = np.mean(square_diffs, axis=0)
    tmse = np.sum(mse)

    return mse, tmse


def calculate_pseudo_Cl(map1,
                        map2,
                        box_size,
                        n_bin=None,
                        ell_min=None,
                        ell_max=None,
                        logspaced=False):
    """Estimates the cross-power spectrum of two maps.

    Required arguments:
    map1            Array of size (N, M).
    map2            Array of same shape as map1.
    box_size        Physical size (L1, L2) of the maps.

    Optional arguments:
    n_bin           Number of ell bins. If None, no binning is performed.
    ell_min         Minimum ell.
    ell_max,        Maximum ell.
    logspaced       Log-spaced bins. Default is False.

    Returns:
    Tuple (pCl_real, pCl_real_err, ell_mean, bin_edges, n_mode) with
        pCl_real        Estimated cross-power spectrum,
        pCl_real_err    Error on the mean, estimated from the scatter of the
                        individual modes,
        ell_mean        Mean ell per bin,
        bin_edges       Edges of the ell bins,
        n_mode          Number of modes per bin.
    """

    if map1.shape != map2.shape:
        raise ValueError(
            "Map dimensions don't match: {}x{} vs {}x{}".format(
                *(map1.shape + map2.shape)))

    # This can be streamlined alot
    map1_ft = np.fft.fft2(map1) * (box_size[0]/map1.shape[0])*(box_size[1]/map1.shape[1])
    map1_ft = np.fft.fftshift(map1_ft, axes=0)
    map2_ft = np.fft.fft2(map2) * (box_size[0]/map1.shape[0])*(box_size[1]/map1.shape[1])
    map2_ft = np.fft.fftshift(map2_ft, axes=0)
    map_1x2_ft = map1_ft.conj()*map2_ft

    ell_x_min_box = 2.0*pi/box_size[0]
    ell_y_min_box = 2.0*pi/box_size[1]
    ell_x = np.fft.fftshift(
        np.fft.fftfreq(map1.shape[0], d=1.0/map1.shape[0]))*ell_x_min_box
    ell_y = np.fft.fftfreq(
        map1.shape[1], d=1.0/map1.shape[1])*ell_y_min_box
    x_idx, y_idx = np.meshgrid(ell_x, ell_y, indexing="ij")
    ell_grid = np.sqrt((x_idx)**2 + (y_idx)**2)

    if n_bin==None:
        bin_edges = np.arange(
            start=np.min([ell_x_min_box, ell_y_min_box])/1.00001,
            stop=np.max(ell_grid),
            step=np.min([ell_x_min_box, ell_y_min_box]))
        n_bin = len(bin_edges) - 1
    else:
        if ell_max > np.max(ell_grid):
            raise RuntimeWarning(
                "Maximum ell is {}, where as ell_max was set as {}.".format(
                    np.max(ell_grid), ell_max))
        if ell_min < np.min([ell_x_min_box, ell_y_min_box]):
            raise RuntimeWarning(
                "Minimum ell is {}, where as ell_min was set as {}.".format(
                    np.min([ell_x_min_box, ell_y_min_box]), ell_min))
        if logspaced:
            bin_edges = np.logspace(np.log10(ell_min),
                                    np.log10(ell_max),
                                    n_bin+1, endpoint=True)
        else:
            bin_edges = np.linspace(ell_min, ell_max, n_bin+1, endpoint=True)

    pCl_real = np.zeros(n_bin)
    pCl_imag = np.zeros(n_bin)
    pCl_real_err = np.zeros(n_bin)
    pCl_imag_err = np.zeros(n_bin)
    ell_mean = np.zeros(n_bin)
    n_mode = np.zeros(n_bin)
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2

    ell_sort_idx = np.argsort(ell_grid.flatten())
    map_1x2_ft_sorted = map_1x2_ft.flatten()[ell_sort_idx]
    ell_grid_sorted = ell_grid.flatten()[ell_sort_idx]
    bin_idx = np.searchsorted(ell_grid_sorted, bin_edges)

    for i in range(n_bin):
        P = map_1x2_ft_sorted[bin_idx[i]:bin_idx[i+1]]/(box_size[0]*box_size[1])
        ell = ell_grid_sorted[bin_idx[i]:bin_idx[i+1]]
        pCl_real[i] = np.mean(P.real)
        pCl_imag[i] = np.mean(P.imag)
        pCl_real_err[i] = np.sqrt(np.var(P.real)/len(P))
        pCl_imag_err[i] = np.sqrt(np.var(P.imag)/len(P))
        ell_mean[i] = np.mean(ell)
        n_mode = len(P)

    return pCl_real, pCl_real_err, ell_mean, bin_edges, n_mode
