*********************************
KPI
*********************************

Introduction
============

In order to have an efficient overview on your synthetic chronics, a set of indicators (KPI) is available.
It also enable you to compare the synthetic chronics KPI with the same KPIs computed on reference (realistic) chronics.
These realistic chronics are to be given by the user. We provide french examples but also a *Jupyter notebook* to extract renewable chronics with *Renewable Ninja* API.
See section :ref:`reference-chronics`


Configuration and Inputs
=========================

The KPI module has a configuration by case. The input files has to be placed in the folder:

``[your_input_folder]/kpi/[your_case]``

The input files that have to be provided by the user are the following:

* *paramsKPI.json* - a configuration dictionary
* *France/* - A folder containing the reference chronics. It can be names however you want. We take the example of french chronics here

.. _reference-chronics:

Available reference chronics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Data is processed with chronix2grid.kpi.pivot_KPI::pivot_format. It currently works with ``comparison == "France"`` and ``comparison == "Texas"``.
You can update the function pivot_format to make it work with your own comparison folder, and apply your own preprocessing.
As a didactic example, this section is about the *France* reference chronics that are provided.

Two folders are located within *France*: if thermic chronics have been generated ("T" in mode argument), a full energy mix reference will be loaded (eco2mix data from RTE)
Else, if only wind and solar chronics have been generated, chronics from renewable_ninja (will be loaded)

* *eco2mix* data has been extracted in 3 french regions with different climates (Hauts de France, PACA, Nouvelle Aquitaine) in order to represent spatial correlation between wind and solar generators
* *renewable_ninja* data has been requested from `Renewable Ninja API <https://www.renewables.ninja/documentation/api/>`_ at different locations in France representing the same 3 regions (Hauts de France, PACA, Nouvelle Aquitaine). We provided a utilitary Notebook in order to extract modeled solar and wind production at the location you desire (by clicking on a Folium Map) and for the years you desire. Note that there is a limitation of requestsper your and per day on Renewable Ninja. See *getting_started/RenewableNinja_Solar_WInd_KPI_format.ipynb*. You can see the solar and wind locations in the pictures provided and the precise coordinates in json files


.. figure:: ../pictures/solar_locations.png
   :scale: 40 %
   :alt: Locations of solar generators requested in Renewable Ninja API

   Locations of solar generators requested in Renewable Ninja API

.. figure:: ../pictures/wind_locations.png
   :scale: 75 %
   :alt: Locations of wind generators requested in Renewable Ninja API

   Locations of wind generators requested in Renewable Ninja API


Configuration json
^^^^^^^^^^^^^^^^^^^
The configuration file *paramsKPI.json* has the following arguments:

* *comparison* - name of the folder in which your reference chronics are stored (here *France"). It has to be handled by a parser in kpi/preprocessing
* *timestep* - the timestep at which we will compute KPIs
* *cloudiness_quantile* and *cloudiness_factor* - parameters for the computation of cloudiness (see :ref:`solar-kpi`)
* *night_hours* - a dictionary that represents nights hours (ex: "summer":["06:00","23:00"]) per season
* *seasons* - a dictionary that represents the months belonging to each season (ex: "summer": [6,7,8])

Available KPIs
==================

Dispatch view
^^^^^^^^^^^^^^^^^

2 indicators concerning the overall economic dispatch are returned. It will only be returned if letter "T" is in launching mode, which means that an economic dispatch has been computed

* *energy_mix* - it compares the repartition of the energy production (MWh) in each carrier, in a pie chart
.. image:: ../pictures/energy_mix.png
* *reference_prod_per_carrier* and *synthetic_prod_per_carrier* are HTLM interactive file in which the energy productions (MWh) of each generator
are stacked and plotted in time.
.. image:: ../pictures/Dispatch_view_example.png

Load KPIs
^^^^^^^^^^

2 indicators concerning the energy consumption are returned.

* *load_by_day_of_week* - showing the average per day of week of the aggregate energy consumption (MWh)
.. image:: ../pictures/load_kpi/load_by_day_of_week.png
* *load_by_week_of_year* - showing the average per week of year of the aggregate energy consumption (MWh)
.. image:: ../pictures/load_kpi/load_by_week_of_year.png

Indicators showing the correlation between thermal production and energy consumption are returned, with one matrix for each region.
An example for region 2

.. image:: ../pictures/thermal_load_kpi/corr_thermal_load_R2.png

An equivalent is provided with wind production, which should be close to zero correlation in our reference scenario

.. image:: ../pictures/wind_load_kpi/corr_wind_load_R2.png


.. _solar-kpi:

Solar KPIs
^^^^^^^^^^
C

Wind KPIs
^^^^^^^^^^
C

Hydro KPIs
^^^^^^^^^^
C

Nuclear KPIs
^^^^^^^^^^^^^
C

Thermal KPIs
^^^^^^^^^^^^^
C