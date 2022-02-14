#include "objective.hh"

double objective(array<double, NB_TYPES> energy_mix) {
  double res = 0;
  for (int i=0; i<NB_TYPES; i++) {
    res += pow(target_energy_mix[i] - energy_mix[i], 2.0);
  }
  return res;
}

array<double, NB_TYPES> get_pmaxs(array<int, N> x) {
  array<double, NB_TYPES> pmaxs;
  for (int i=0; i<NB_TYPES; i++) {
    int count = 0;
    for (int j=0; j<N; j++) {
      if (x[j] == i) count++;
    }

    pmaxs[i] = avg_pmaxs[i] * count;
  }

  return pmaxs;
}

array<double, NB_TYPES> get_energy_mix(array<double, NB_TYPES> pmaxs) {
  array<double, NB_TYPES> apriori_energy_mix;
  double sum = 0;
  for (int i=0; i<NB_TYPES; i++) {
    apriori_energy_mix[i] = capacity_factor[i] * pmaxs[i] / average_load;
    if (apriori_energy_mix[i] > 0)
      sum += apriori_energy_mix[i];
  }
  
  if (sum > 100) {
    apriori_energy_mix[0] -= (sum - 100);
    apriori_energy_mix[3] = 0;
  } else {
    apriori_energy_mix[3] = 100 - sum;
  }
  return apriori_energy_mix;
}