# EDispatch_L2RPN2020

This repo contains a simple Economic Dispatch using PyPSA for the IEEE 118 grid. <br>
Please take into consideration the following: <br>
 - The synthetic data generation (chronics) a full Grid2OP comptability is recommended when grid is created to avoid discordances with gen, load, line headers.
  - For now, Texas consumption, wind and solar are imposed to the grid.
  - Users might use ***one single optimization formulation*** for day, week, month and year. However, the last one is not recommended because of time-consuming.

## Requirements
- Pandas >= 0.25.3
- Numpy >= 1.18.1
- PyPSA == 0.16.1
- Python 3.x
- CBC Solver (recommended)
     - *Camilo's version (Mac) stable 2.10.3*
     - *Nicola's version (Windows) 1.7.4*

For installing PyPSA, please do the following steps: <br>
PypSA has been added as a "submodule" of this package, to install the last version on github do:

```
# git clone https://github.com/PyPSA/PyPSA.git
# cd /path/to/PyPSA/ 
# python -m pip install -e
git submodule init
git submodule update
pip install -U PyPsa
```
For installing CBC, please follow the instructions on https://github.com/coin-or/Cbc. In summary:
 - ***Linux***: On Debian/Ubuntu, Cbc is available in the package `coinor-cbc` and can be installed with apt. On Fedora, Cbc is available in the package `coin-or-Cbc`.
- ***Windows***: The easiest way to get Cbc on Windows is to download from [Bintray](https://bintray.com/coin-or/download/Cbc), although an old interactive installer for the [COIN-OR Optimization Suite](https://www.coin-or.org/download/binary/CoinAll/) is also still available.
- ***Mac OS X***: The easiest way to get Cbc on Mac OS X is through Homebrew.
    - `brew tap coin-or-tools/coinor`
    - `brew install cbc`

## Usage
Before running the code. Open the code `run_economic_dispatch.py` and modify as you want the next variables:
 - [DEPRECATED] `MODE_OPF` [period] (day, week, month, year) -> It will create one single formulation in PyPSA and then OPF will be running in a ***for loop*** for consecutives periods.
 - [DEPRECATED] `RESCALED_MIN` [min] (multiples of five) -> The time OPF runs inside a period (Can be 5, 10, 15, 20, 30, 60)
 - `MONTH_START` [int] -> Reference data is only is adapted for 2007 year. OPF will be initialized at the month you specify.
  - `MONTH_END` [int] -> Reference data is only is adapted for 2007 year. OPF will be end at the month you specify.


  To run the script, open the Terminal: <br>
  ```
  cd path/to/EDispatch_L2RPN2020
  python run_economic_dispatch.py --help
  ```
You will be able to modify, from the command line argument, the "mode_opf" and "rescaled_min". For example, if you want
to perform a daily optimization, at a five minute resolution you can:
```bash
python run_economic_dispatch.py --mode_opf="day" --rescaled_min=5
```
These are also valid values:
```bash
python run_economic_dispatch.py --mode_opf="month" --rescaled_min=15
```

## times taken
In the file "times.csv" (Benjamin) reported some times to perform the optimization on the month of January on an Ubuntu
desktop. Optimization is carried out using CBC solver for the 31 days of january.

This csv (separated with coma "," and with "." as decimal separator) counts 5 columns:
- `mode_opf`: the mode_opf used (among "day", "week" or "month")
- `rescaled_min`: the rescaled_min used (integer multiple of 5)
- `total_time`: total time taken (in second): what is output after the programm finished (*eg* 
  "Total time 8.97 min" -> it is then 538.2s -- which corresponds to 8.97 mins)
- `total_opt`: total number of optimization performed (for example, 
  if you ask a daily optimization, for the month of january, you get
  31, because there are 31 days in january)
- `avg_time`: average time (in second) for each optimization (computed by dividing total time with total opt)





