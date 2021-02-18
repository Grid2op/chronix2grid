***************
Getting Started
***************

Launch Chronix2Grid
====================

``chronix2grid [OPTIONS]``

--case string
                            Case folder to base generation on
--start-date string
                            Start date to generate chronics
--weeks int           Number of weeks to generate
--by-n-weeks int      Size of the output chunks in weeks
--n_scenarios int     Number of scenarios to generate
--mode string
                            Steps to execute : L(K) for loads only (and KPI);
                            R(K) for renewables (and KPI) only; LRT (K)
                            for load, renewable and thermic generation (and KPI);
                            LRDT(TK) for load, renewable, loss (dissipation) generation
                            (and thermic and KPI)

--input-folder string
                            Directory to read input files from.
--output-folder string
                            Directory to store output files.
--seed-for-loads string
                            Input seed to ensure reproducibility of load generation
--seed-for-res string
                            Input seed to ensure reproducibility of renewables generation
--seed-for-dispatch string
                            Input seed to ensure reproducibility of dispatch
--ignore-warnings
                            Ignore the warnings related to the existence of
                            data files in the chosen output directory.
--scenario_name string
                            Subname to add to the generated scenario output folder, as Scenario_subname_i
--nb_core int
                            Number of cores to parallelize the number of scenarios


Features
============

4 generation submodules and a KPI module are available

* L - load generation
* R - wind and solar production generation
* D - loss generation (depending on L and R)
* T - thermic production generation thanks to a production dispatch (simplified optimal power flow computation)
* K - KPI generation in order to compare synthetic (generated) chronics to reference (real-life) chronics

The figure below shows how these submodules can be launched together with --mode argument.
Note that D and T submodules can't be launched without previous L and R modules, and that KPIs can always been computed

.. image:: ../pictures/Launch_mode.png

Inputs and Configuration
========================

For synthetic chronic generation, one should provide

* Data that represents grid, demand and generators characteristics
* Pattern data (ex: weekly load profile)
* Json parameters for generation algorithms

One should also provide reference chronics data for comparison through a library of KPIs

.. image:: ../pictures/ChroniX2Grid_inputs.png

Outputs
========================

.. image:: ../pictures/ChroniX2Grid_outputs.png

Tests
=====

To launch the test suite:
``pipenv run python -m pytest --verbose --continue-on-collection-errors -p no:warnings``

