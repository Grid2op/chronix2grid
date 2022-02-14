#include "constants.hh"
#include <math.h>

double objective(array<double, NB_TYPES> energy_mix); 
array<double, NB_TYPES> get_pmaxs(array<int, N> x);
array<double, NB_TYPES> get_energy_mix(array<double, NB_TYPES> pmaxs);