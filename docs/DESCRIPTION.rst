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
These functions  are written :math:`f_t^\text{category}(x,y)`, :math:`category` being the model category (solar, short medium or long term for wind, temperature for load).
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

.. image:: ../pictures/spatial_correlation.png


Temporal correlation
^^^^^^^^^^^^^^^^^^^^

Then a temporal auto-correlation structure is achieved. For each category, we go from
resolution **[category]_corr** (at which noises have been generated independently in time)
to resolution **dt** thanks to spline interpolation

Solar generation
=================

For solar generation, some additional parameters are provided:

* A yearly smooth solar pattern file at .npy format. It will be marked as :math:`pattern_t` and it doesn't depend on x and y
* In params.json:
    * **solar_corr** - resolution of temporal autocorrelation in noise (see :ref:`correlated-noise`)
    * **std_solar_noise** - standard deviation of the spatial and temporal correlated noise. It will be marked as :math:`\sigma`
    * **smooth_dist** - standard deviation of additional centered gaussian noise (will be normalized by Pmax). It will be marked as :math:`s`

For each solar generator located at x, y and with max power generation of :math:`P_\text{max}`

.. math:: prod_t(x,y) = P_\text{max} * smooth(pattern_t * (0.75+\sigma f_t^\text{solar}(x,y)) + n_s(x,y,t,P_\text{max}))

Where :

* :math:`f_t^\text{solar}(x,y)` is the solar correlated noise (see section :ref:`correlated-noise`)
* smooth is a smoothing function. We currently use :math:`smooth(x) = 1 - exp(-x)`
* :math:`n_s(x,y,t,P_\text{max})` is an independent additional noise following distribution :math:`N(0,s/P_\text{max})`
* 0.75 is the bias of the spatially and temporally correlated noise.

In other words, the yearly temporal pattern is multiplied by a biased noise function which defines the spatial and temporal correlation structures
of solar generators. It implies by the way that zero production timesteps remain zero.
Then a centered and independent gaussian noise is added to each generator. This quantity is smoothed and scaled in
interval :math:`[0,1]`. Finally, this normal production is rescaled to :math:`P_\text{max}`

.. figure:: ../pictures/solar_gen_10_5_chronic_example_year.png
   :scale: 50 %
   :alt: Solar year example

   Example of generated solar chronic across year 2012. Pmax of the solar farm is 37.3 MW. :math:`solar_\text{corr} = 20 minutes` - :math:`smooth_\text{dist} = 0.001` - :math:`\sigma = 0.4`

.. figure:: ../pictures/solar_gen_10_5_chronic_example_week.png
   :scale: 50 %
   :alt: Solar week example

   Focus on one week in summer

Wind generation
=================

The wind normal seasonal pattern relies on a simple cosine which oscillation period is one full year.
Its constant component has a part of 70% and the oscillating component accounts for 30%.
It is at its highest value during December and its lowest value during June. It is simulated as follows:

.. math:: pattern_t = 0.7 + 0.3 cos({2\pi(t-\delta t) \over 365*24*60})

* :math:`t` is the cumulated simulation time in minutes
* :math:`\delta t` is the time delta in minutes between our first simulation time step and the 02/12/2017

For wind generation, some additional parameters are provided.
Note that wind correlated noise structure is achieved with 3 components corresponding to short, medium and long time scales.
It is in *params.json*:

* **short_wind_corr**, **medium_wind_corr** and **long_wind_corr** - resolutions of temporal auto-correlations in noises (see :ref:`correlated-noise`). The higher the time scale the longest is the dependency in the auto-correlation structure
* **std_short_wind_noise**, **std_medium_wind_noise** and **std_long_wind_noise** - standard deviation of the spatial and temporal correlated noise. It will be marked as :math:`\sigma_\text{category}`
* **smooth_dist** - standard deviation of additional centered gaussian noise (will be normalized by Pmax). It will be marked as :math:`s`

Finally, for a given generator located at coordinates :math:`(x,y)`, the shape of the simulated chronics follow the equation:

.. math:: prod_t(x,y) = P_\text{max} smooth(0.1 * exp(4 * pattern_t * (0.3 + \sigma_\text{medium_wind} f_t^\text{medium_wind}(x,y)

    + \sigma_\text{long_wind} f_t^\text{long_wind}(x,y)

    )

    + \sigma_\text{short_wind} f_t^\text{short_wind}(x,y)

    )

    + n_s(x,y,t,P_\text{max}))

Where:

* :math:`f_t^\text{wind_category}(x,y)` are the wind correlated noises (see section :ref:`correlated-noise`)
* smooth is a smoothing function. We currently use :math:`smooth(x) = 1 - exp(-x)`
* :math:`n_s(x,y,t,P_\text{max})` is an independent additional noise following distribution :math:`N(0,s/P_\text{max})`

.. figure:: ../pictures/wind_gen_111_59_chronic_example_year.png
   :scale: 50 %
   :alt: Wind year example

   Example of generated wind chronic across year 2012. Pmax of the wind farm is 67.2 MW. :math:`wind_\text{corr} = 300 min, 1440 min, 20160 min` - :math:`smooth_\text{dist} = 0.001` - :math:`\sigma = 0.02, 0.15, 0.15`

.. figure:: ../pictures/wind_gen_111_59_chronic_example_week.png
   :scale: 50 %
   :alt: Wind week example

   Focus on one week in fall


Load generation
=================

For load generation, parameters are similar to solar generation

* A weekly consumption pattern file at .csv format. It will be marked as :math:`weeklypattern_t` and it doesn't depend on x and y
* In params.json:
    * **temperature_corr** - resolution of temporal auto-correlation in noise (see :ref:`correlated-noise`)
    * **std_temperature_noise** - standard deviation of the spatial and temporal correlated noise. It will be marked as :math:`\sigma`

Additionally to the weekly pattern, a seasonal pattern is modeled with a cosine which oscillation period is one full year.
Its constant component has a part of 5.5/7 and the oscillating component accounts for 1.5/7.
It is at its highest value during December and its lowest value during June. It is simulated as follows:

.. math:: seasonalpattern_t = {5.5 \over 7} + {1.5 \over 7} * cos({2\pi(t-\delta t) \over 365*24*60})

* :math:`t` is the cumulated simulation time in minutes
* :math:`\delta t` is the time delta in minutes between our first simulation time step and the 02/12 of the year before simulation

Finally, for each load site located at x, y and with max power consumption of :math:`P_\text{max}`

.. math:: load_t(x,y) = P_\text{max} * weeklypattern_t * (\sigma * f_t^\text{temperature}(x,y) + seasonalpattern_t)

Where :math:`f_t^\text{temperature}(x,y)` is the temperature correlated noise (see section :ref:`correlated-noise`)


.. figure:: ../pictures/load_87_70_chronic_example_year.png
   :scale: 50 %
   :alt: Load year example

   Example of generated load chronic across year 2012 in region R3. Pmax of the load is 77.1 MW. :math:`temperature_\text{corr} = 400 min` - :math:`\sigma = 0.06`

.. figure:: ../pictures/load_87_70_chronic_example_week.png
   :scale: 50 %
   :alt: Load week example

   Focus on one week in winter


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

