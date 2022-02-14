#include <iostream>
#include "objective.h"

int main(int argc, char** argv) {
  array<int, N> x;
  x.fill(0);

  for (int i = 0; i<5; i++)
    x[i] = 1;

  printf("Pmaxs:\n");
  array<double, NB_TYPES> pmaxs = get_pmaxs(x);
  for (double pmax : pmaxs) {
    printf("%f ", pmax);
  }
  printf("\n");

  printf("Energy mix:\n");
  array<double, NB_TYPES> em = get_energy_mix(pmaxs);
  for (double em : em) {
    printf("%f ", em);
  }
  printf("\n");

  double obj = objective(em);
  printf("Objective: %f\n", obj);

  return 0;
}