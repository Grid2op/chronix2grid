*********************************
Description of implemented models
*********************************

Introduction
============

Some generation models are already implemented within chronix2grid as examples for other model implementation.
This chapter describes the methods they include and how to set their configuration

.. _correlated-noise:

Generation of correlated noise
================================
In load, solar and wind current modelling, some spatial and temporal correlated noises are generated.
These functions  are written :math:`f_t^c(x,y)`, c being the model category (solar, short medium or long term for wind, temperature for load).
They are based on:

* A coarse 3-dimensional mesh (x,y,t) with independent noise
* The spatial and temporal interpolation of this noise at the specific location of generators and specific moments of time

In params.json you can find all the required parameter for this correlated noise generation:

* **Lx**, **Ly** the total length of the mesh
* **dx_corr**, **dy_corr** the granularity of the coarse mesh. it represents the distance at which we consider that spatial phenomenons are independent
* **solar_corr**, **short_wind_corr**, **medium_wind_corr**, **long_wind_corr** and **temperature_corr** which define the coarse time resolution for each type of noise
* **dt** the real time resolution of the final chronics that will be modeled.

Spatial correlation
^^^^^^^^^^^^^^^^^^^^

For each coarse time step t, a 2-dimensional coarse mesh is built.
At each node (x,y,t) an independent random gaussian noise :math:`N(0,1)` is computed

Then a spatial interpolation is made at the specific location (x,y) of the generator,
weighted by the distance ot its nearest neighbour in the mesh

.. image:: ../pictures/spatial_correlation.PNG


Temporal correlation
^^^^^^^^^^^^^^^^^^^^

Then a temporal auto-correlation structure is achieved. For each category c, we go from
resolution **[c]_corr** (at which noises have been generated independently in time)
to resolution **dt** thanks to spline interpolation

Solar generation
=================

For solar generation, some additional parameters are provided:

* A yearly smooth solar pattern file at .npy format. It will be marked as :math:`pattern` and it doesn't depend on x and y
* In params.json:
    * **std_solar_noise** - standard deviation of the spatial and temporal correlated noise. It will be marked as :math:`\sigma`
    * **smooth_dist** - standard deviation of additional centered gaussian noise (will be normalized by Pmax). It will be marked as :math:`s`

For each solar generator located at x, y and with max power generation of :math:`P_m`

.. math:: prod_t(x,y) = P_m * smooth(pattern * (0,75+\sigma f_t^c(x,y)) + n_s(x,y,t,P_m))

Where :

* :math:`f_t^c(x,y)` is the solar correlated noise (see section :ref:`correlated-noise`)
* smooth is an exponential smoothing function
* :math:`n_s(x,y,t,P_m)` is an independent additional noise following distribution :math:`N(0,s/P_m)`

Wind generation
=================

C

Load generation
=================

C


Loss generation
=================

A simple module is actually implemented.
It reads a csv containing a yearly loss pattern chronic (5 min time step in the example provided), given as an absolute power value in MW.
Two inputs are necessary, with example provided in *getting_started/example/input*:

* A csv file containing the yearly loss pattern  in *patterns/loss_pattern.csv*
* A json parameter file that indicates the path to loss pattern in *case118_l2rpn_wcci/generation/params_loss.json*


Economic dispatch generation (hydro, nuclear and thermic generators)
====================================================================

ATTENTION HYDRO + params.json pour le nucléaire !!

