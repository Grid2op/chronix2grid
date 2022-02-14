#include <array>
#include <iostream>
using namespace std;

const int N = 62;
const int NB_TYPES = 5;
const array<string, NB_TYPES> power_plant_types = {"hydro", "nuclear", "solar", "thermal", "wind"};
const array<double, NB_TYPES> avg_pmaxs         = {250.0, 400.0, 46.650, 140.910, 48.0};
const array<double, NB_TYPES> capacity_factor   = {90.0, 95.0, 95.0, -1.0, 95.0};
const array<double, NB_TYPES> target_energy_mix = {9.0, 36.0, 17.0, 2.0, 36.0};
const int target_pmax = 1000;
const int average_load = 2800;