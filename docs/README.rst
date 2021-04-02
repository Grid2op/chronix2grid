Mentions
=========

Quick Overview
------------------

Chronix2Grid is a python package, providing a command-line application as well,
that allows to generate synthetic but realistic consumption, renewable production, electricity loss (dissipation)
and economic dispatched productions chronics given a power grid.
Reference data that you provide will serve to calibrate the parameters
so that the synthetic data reproduce some realistic criteria (KPIs) from the reference data.


Features
----------

* Specify a power grid model
* Generates many scenarios of realistic load, loss, solar, wind, hydro and thermic power chronics
* Choose time scale and temporal resolution
* Check if the synthetic chronics are realistic compared to reference chronics (TSO, renewable ninja) following several KPIs
* Easily interface your own generation models

4 generation submodules and a KPI module are available. These submodules are modular interface for your own generation model

* L - load generation
* R - wind and solar production generation
* D - loss generation a priori, that will be used for dispatch and potentially corrected afterwards
* T - thermic production generation thanks to a production dispatch (simplified optimal power flow computation)
* K - KPI generation in order to compare synthetic (generated) chronics to reference (real-life) chronics


Contribute
-------------

- Issue Tracker: https://github.com/mjothy/ChroniX2Grid/issues
- Source Code: https://github.com/mjothy/ChroniX2Grid/

Support
----------

If you are having issues, please let us know on github

License
---------
Copyright 2020-2021 RTE France

    RTE: http://www.rte-france.com

This Source Code is subject to the terms of the Mozilla Public License (MPL) v2.
