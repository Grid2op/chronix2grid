# ChroniX2Grid

Chronix2Grid is a python package, providing a command-line application as well, that allows to generate synthetic but realistic consumption, renewable production and economic dispatched productions chronics given a power grid. Reference data that you provide will serve to calibrate the parameters so that the synthetic data reproduce some realistic criteria (KPIs) from the reference data.

*   [1 Chronix2Grid at a glance](#glance)
*   [2 Installation](#installation)
*   [3 Getting Started](#getting-started)
*   [4 The command-line interface](#the-command-line-interface)
*   [5 Configuration](#configuration)
*   [6 Running the test suite](#running-the-test-suite)

## Chronix2Grid at a glance
![Chronix2grid_inputs](pictures/ChroniX2Grid_inputs.png "Chronix2Grid Inputs") 

![Chronix2grid_outputs](pictures/ChroniX2Grid_ouputs.png "Chronix2Grid Ouputs") 

## Installation

### Requirements:
*   Python >= 3.6

#### (Optional, recommended) Step 1: Create a virtual environment
```commandline
pip3 install -U virtualenv
python3 -m virtualenv venv_chronix2grid
```

#### Step 2 (first option): Install from pypi
```commandline
source venv_chronix2grid/bin/activate
pip install Chronix2Grid
```

#### Step 2 (second option): Install from source
```commandline
source venv_chronix2grid/bin/activate
git clone https://github.com/mjothy/ChroniX2Grid.git
cd ChroniX2Grid/
pip install -U .
```

## Getting Started
Four notebooks are provided to get you started with this package:

* **getting_started_cli.ipynb** guide you through the use of chronix2grid cli
* **getting_started_api.ipynb** is a more detailed notebook that presents chronix2grid api and showcases several ways to
 analyze the chronics produced
* **running_chronics_grid2op.ipynb** is a notebook giving an example of how generated chronics can further be used to create power grid scenarios and run them 
 analyze the chronics produced
* **RenewableNinja_Solar_WInd_KPI_format.ipynb** is an example on how to retrieve Solar and Wind reference chronics from the
  [renewable ninja](https://www.renewables.ninja/) api. There will be use to compare the KPI's with the generated chronics. 
 
Additionally, a data starting kit is provided to run an example in the folder 
  ```commandline
getting_started/example
```
 The output folder corresponds to the following run of chronix2grid : 
 ```commandline
chronix2grid --mode RLTK --output-folder ChroniX2Grid/getting_started/example/output --input-folder /ChroniX2Grid/getting_started/example/input --ignore-warnings --weeks 8 --case case118_l2rpn_wcci --n_scenarios 1 --start-date 2012-01-01 --by-n-weeks 4
```

### WARNING
In order to limit the size of the output and the running time for this example, chronics are only generated for 8 weeks.
This implicates that some kpis that defined on a whole year will not be exploitable for this example.
 
## The command-line interface
```commandline
Usage: chronix2grid [OPTIONS]

Options:
  --case TEXT               case folder to base generation on
  --start-date TEXT         Start date to generate chronics
  --weeks INTEGER           Number of weeks to generate
  --by-n-weeks INTEGER      Size of the output chunks in weeks
  --n_scenarios INTEGER     Number of scenarios to generate
  --mode TEXT               Steps to execute : L for loads only (and KPI);
                            R(K) for renewables (and KPI) only; LRTK for all
                            generation

  --input-folder TEXT       Directory to read input files from.
  --output-folder TEXT      Directory to store output files.
  --seed-for-loads TEXT     Input seed to ensure reproducibility of loads
                            generation

  --seed-for-res TEXT       Input seed to ensure reproducibility of renewables
                            generation

  --seed-for-dispatch TEXT  Input seed to ensure reproducibility of dispatch
  --ignore-warnings         Ignore the warnings related to the existence of
                            data files in the chosen output directory.

  --scenario_name TEXT      subname to add to the generated scenario output
                            folder, as Scenario_subname_i

  --nb_core INTEGER         number of cores to parallelize the number of
                            scenarios

  --help                    Show this message and exit.

```
## Configuration

### Chronic generation detailed configuration
More detailed configuration has to be set in *INPUT_FOLDER/generation/CASE/params.json* such as timestep, noise and
 correlation
 intensity...


### KPI configuration
Some general parameters have to be set in *INPUT_FOLDER/kpi/paramsKPI.json*
- **comparison**: name of benchmark folder to use for KPI reference chronics. For example, benchmark *France* has been implemented with eco2mix and renewable ninja data
- **timestep**: timestep for KPI computation. For example, renewable ninja data requires minimum timestep of 60min
- **night_hours**: dictionary to provide night hours for each season of year
- **seasons**: dictionary to provide months in each season of year

## Running the test suite
To run the tests, execute:
```commandline
python -m unittest discover
```