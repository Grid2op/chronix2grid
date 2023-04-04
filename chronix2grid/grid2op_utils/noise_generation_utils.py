# Copyright (c) 2019-2023, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Chronix2Grid, A python package to generate "en-masse" chronics for loads and productions (thermal, renewable)

import itertools
import numpy as np
import warnings

from chronix2grid.generation.generation_utils import get_nx_ny_nt
from numpy.random import default_rng


def generate_coords_mesh(Nx_comp, Ny_comp, Nt_comp, Nh_comp, size,
                         ratio_adjust=1.0):
    # distort the distance for the mesh (because we use kNN later on)
    max_mesh = max(Nx_comp, Ny_comp, Nt_comp, Nh_comp)
    rho_mesh_x = max_mesh / Nx_comp * ratio_adjust
    rho_mesh_y = max_mesh / Ny_comp * ratio_adjust
    rho_mesh_t = max_mesh / Nt_comp * ratio_adjust
    rho_mesh_h = max_mesh / Nh_comp * ratio_adjust
    
    # generate coordinates of the mesh
    coords_mesh = np.empty((size, 4))
    for i, row in enumerate(itertools.product(range(Nx_comp), range(Ny_comp), range(Nt_comp), range(Nh_comp))):
        coords_mesh[i,:] = row
    coords_mesh[:,0] /= (Nx_comp - 1) * rho_mesh_x
    coords_mesh[:,1] /= (Ny_comp - 1) * rho_mesh_y
    coords_mesh[:,2] /= (Nt_comp - 1) * rho_mesh_t
    coords_mesh[:,3] /= (Nh_comp - 1) * rho_mesh_h
    return coords_mesh, rho_mesh_x, rho_mesh_y, rho_mesh_t, rho_mesh_h
    

def get_load_mesh_tmp(nb_t, hs_, hs_mins, rho_mesh_t, rho_mesh_h):
    nb_h = len(hs_)
    # now find the noise for all the loads
    load_mesh_tmp = np.empty((nb_t * nb_h, 4))
    for i, row in enumerate(itertools.product(range(nb_t), hs_)):
        load_mesh_tmp[i,:] = (0, 0, *row)
    # compute where the "point" is on the mesh
    load_mesh_tmp[:,2] += 1
    load_mesh_tmp[:,2] /= (nb_t + 1)
    load_mesh_tmp[:,3] /= (np.max(hs_mins) + 5)
    
    # don't forget to re normalize it
    load_mesh_tmp[:,2] /= rho_mesh_t
    load_mesh_tmp[:,3] /= rho_mesh_h
    return load_mesh_tmp


def compute_noise(load_mesh_tmp,
                  loads_charac,
                  model,
                  range_x, range_y,
                  delta_x, delta_y,
                  rho_mesh_x, rho_mesh_y,
                  nb_t, nb_h, nb_load):
    
    loads_noise = np.zeros((nb_load, nb_t, nb_h))
    for row_id, (load_id, (load_x, load_y)) in enumerate(loads_charac[["x", 'y']].iterrows()):
        # compute where the "point" is on the mesh
        load_mesh = 1.0 * load_mesh_tmp
        load_mesh[:,0] = (load_x - range_x[0])/delta_x
        load_mesh[:,1] = (load_y - range_y[0])/delta_y
        
        # don't forget to re normalize it
        load_mesh[:,0] /= rho_mesh_x
        load_mesh[:,1] /= rho_mesh_y
        
        noise_load = model.predict(load_mesh)
        loads_noise[row_id, :, :] = noise_load.reshape(nb_t, nb_h)
        
    # renormalize the noise
    loads_noise -= loads_noise.mean()
    loads_noise /= loads_noise.std()  
    return loads_noise


def get_knn_fitted(forecasts_params, coords_mesh, noise_mesh):
    from sklearn.neighbors import KNeighborsRegressor
    if "n_neighbors" in forecasts_params:
        n_neighbors = int(forecasts_params["n_neighbors"])
        assert n_neighbors > 0, f"n_neighbors parameters should be > 0 and is currently {n_neighbors}"
    else:
        n_neighbors = 16 * 15 - 32  # 16 = les 16 arretes du cube 4d dans lequel il est
    
    if "leaf_size_knn" in forecasts_params:
        leaf_size_knn = int(forecasts_params["leaf_size_knn"])
    else:
        leaf_size_knn = 100  # found to be the best compromise
    
    if "algorithm_knn" in forecasts_params:
        algorithm_knn = str(forecasts_params["algorithm_knn"])
    else:
        algorithm_knn = "auto"  # should be good
        
    model = KNeighborsRegressor(n_neighbors=n_neighbors,  
                                weights="distance",
                                leaf_size=leaf_size_knn,
                                algorithm=algorithm_knn)
    model.fit(coords_mesh, noise_mesh.reshape(-1))
    return model


def get_forecast_parameters(forecasts_params, load_params, data_type="load"):    
    hs_mins = [int(el) for el in forecasts_params["h"]]
    # convert the h in "number of steps" (and not in minutes)
    hs = [h // load_params['dt'] for h in hs_mins]
    # check consistency
    for h, h_min in zip(hs, hs_mins):
        if h * load_params['dt'] != h_min:
            # TODO logger
            warnings.warn("Some forecast horizons are not muliple of the duration of a steps. They will be rounded")
    
    # load the parameters "h" for the forecasts
    std_hs = [float(el) for el in forecasts_params[f"h_std_{data_type}"]]
    for std in std_hs:
        assert std > 0, f"all parameters of 'h_std_{data_type}' should be >0 we found {std}"
    assert len(std_hs) == len(hs), "you should provide as much 'error' as there are forecasts horizon"
    return hs_mins, hs, std_hs


def get_iid_noise(load_seed, load_params, forecasts_params,
                  loads_charac, data_type, get_add_dim,
                  prng=None):
    if prng is None:
        prng = default_rng(load_seed)
    add_dim = get_add_dim(load_params, loads_charac)
    Nx_comp, Ny_comp, Nt_comp = get_nx_ny_nt(data_type, load_params, add_dim)
    Nh_comp = forecasts_params["nb_h_iid"]
    noise_mesh = prng.normal(0, 1, (Nx_comp, Ny_comp, Nt_comp, Nh_comp))
    return noise_mesh, (Nx_comp, Ny_comp, Nt_comp, Nh_comp)


def resize_mesh_factor(loads_charac, gen_charac, ratio_border=20.):
    # in the generation we will map this mesh to [0., 1.]
    range_x = min(loads_charac["x"].min(), gen_charac["x"].min()), max(loads_charac["x"].max(), gen_charac["x"].max())
    range_y = min(loads_charac["y"].min(), gen_charac["y"].min()), max(loads_charac["y"].max(), gen_charac["y"].max())
    delta_x_ = range_x[1] - range_x[0]
    range_x = (range_x[0] - delta_x_ / ratio_border, range_x[1] + delta_x_ / ratio_border)
    delta_y_ = range_y[1] - range_y[0]
    range_y = (range_y[0] - delta_y_/ ratio_border, range_y[1] + delta_y_ / ratio_border)
    # the (+/- delta_x_ / 20.) is to handle side effect at the border of the mesh
    
    delta_x = range_x[1] - range_x[0]
    delta_y = range_y[1] - range_y[0]
    return delta_x, delta_y, range_x, range_y

    
def get_forecast(load_p, loads_noise, hs, std_hs, loads_charac, reshape=True, keep_first_dim=False):
    nb_load = load_p.shape[0]
    nb_t = load_p.shape[1]
    nb_h = len(hs)
    
    load_p_for = np.stack([np.roll(load_p, -h) for h in hs], axis=2)
    
    if keep_first_dim:
        loads_noise_forecast = 1.0 * loads_noise[:,:,:]
    else:
        # in case of load, first dimension is thrown out (used to
        # compute the real time series data)
        loads_noise_forecast = 1.0 * loads_noise[:,:,1:]
    loads_noise_forecast *= np.array(std_hs).reshape(1, len(hs))
    loads_noise_forecast *= loads_charac["Pmax"].values.reshape(nb_load, 1, 1)
    load_p_for += loads_noise_forecast
    # keep the last value for the "forecasts outside the episode"
    for col, ts in enumerate(hs):
        load_p_for[:, (nb_t - ts):, col] = load_p_for[:, (nb_t - ts) - 1, col].reshape(nb_load, 1)
    
    # put everything in the right shape
    if reshape:
        load_p_for = load_p_for.reshape(nb_load, nb_t * nb_h)
    return load_p_for


def generate_noise(loads_charac,
                   gen_charac,
                   forecasts_params,
                   load_params,
                   load_seed,
                   data_type,
                   get_add_dim,
                   nb_t,
                   elem_params,
                   elem_charac,
                   nb_elem,
                   add_h0=True,
                   prng_noise=None):
    # compute the "real" size of the mesh 
    delta_x, delta_y, range_x, range_y = resize_mesh_factor(loads_charac, gen_charac)
    
    # retrieve the forecast parameters
    if data_type=='temperature':
        hs_mins, hs, std_hs = get_forecast_parameters(forecasts_params, load_params) 
    else:
        hs_mins, hs, std_hs = get_forecast_parameters(forecasts_params, load_params, data_type)
             
    # generate the independant data on the mesh 
    tmp_ = get_iid_noise(load_seed, elem_params, forecasts_params,
                         loads_charac, data_type, get_add_dim,
                         prng_noise)
    noise_mesh, (Nx_comp, Ny_comp, Nt_comp, Nh_comp) = tmp_
    
    # resize the mesh
    res_mesh = generate_coords_mesh(Nx_comp,
                                    Ny_comp,
                                    Nt_comp,
                                    Nh_comp,
                                    noise_mesh.size) 
    coords_mesh, rho_mesh_x, rho_mesh_y, rho_mesh_t, rho_mesh_h = res_mesh
    
    # "fit" the kNN    
    model = get_knn_fitted(forecasts_params, coords_mesh, noise_mesh)
    
    # get the "temporary" for the load coordinates (whether or not to generate the
    # also the "env" value with this function)
    if add_h0:
        hs_ = [0] + hs
    else:
        hs_ = hs
    
    # retrieve the "template" of coordinates of the mesh for the output vector
    mesh_tmp = get_load_mesh_tmp(nb_t, hs_, hs_mins,
                                 rho_mesh_t, rho_mesh_h)
    
    # now retrieve the real noise for each load
    this_noise = compute_noise(mesh_tmp,
                               elem_charac,
                               model,
                               range_x, range_y,
                               delta_x, delta_y,
                               rho_mesh_x, rho_mesh_y,
                               nb_t, len(hs_), nb_elem)
    
    return this_noise, hs, std_hs