#include "algorithms.hh"

void print_infos(Infos infos) {
  cout << "Algorithm: " << infos.algorithm << '.' << endl;
  printf("%d iterations.\n", infos.nb_iterations);
  printf("Finished in %.2f seconds.\n", infos.elapsed_time);
}

void Solver::print_result(Result result) {
  print_infos(result.infos);

  array<int, NB_TYPES> total_types;
  total_types.fill(0);
  for (int i=0; i<N; i++) {
    total_types[result.x[i]]++;
  }

  for (int i=0; i<NB_TYPES; i++) {
    cout << power_plant_types[i] << ": " << total_types[i] << endl;
  }
  printf("\n");

  array<double, NB_TYPES> pmaxs      = get_pmaxs(result.x);
  array<double, NB_TYPES> energy_mix = get_energy_mix(pmaxs);


  printf("Objective: %f\n\n", result.obj);

  int total_pmax = 0;
  printf("Pmaxs:\n");
  for (double pmax : pmaxs) {
    printf("%.2f ", pmax);
    total_pmax += pmax;
  }
  printf("\nTotal pmax: %d\n\n", total_pmax);

  printf("Target energy mix:\n");
  for (double em : target_energy_mix) {
    printf("%.2f ", em);
  }
  printf("\n");

  printf("Energy mix:\n");
  for (double em : energy_mix) {
    printf("%.2f ", em);
  }
  printf("\n");

  double error = 0.0;
  for (int i=0; i<NB_TYPES; i++){
    error += abs(target_energy_mix[i] - energy_mix[i]);
  }
  printf("Difference between target and actual energy mix: %.2f%%\n", error);
}