#include "algorithms.hh"
#include <fstream>

void print_infos(Infos infos) {
  cout << "Algorithm: " << infos.algorithm << '.' << endl;
  printf("%d iterations.\n", infos.nb_iterations);
  printf("Finished in %.2f seconds.\n\n", infos.elapsed_time);
}

void Algorithm::print_result(Result result) {
  print_infos(result.infos);
  printf("--------------------------RESULT-----------------------------\n");

  vector<int> total_types(this->problem.NB_TYPES);
  for (int i=0; i<this->problem.N; i++) {
    total_types[result.x[i]]++;
  }

  for (int i=0; i<this->problem.NB_TYPES; i++) {
    cout << this->problem.power_plant_types[i] << ": " << total_types[i] << endl;
  }
  printf("\n");

  vector<double> pmaxs      = this->problem.get_pmaxs(result.x);
  vector<double> energy_mix = this->problem.get_energy_mix(pmaxs);

  printf("Objective: %f\n\n", result.obj);

  int total_pmax = 0;
  printf("Pmaxs:\n");
  for (double pmax : pmaxs) {
    printf("%.2f ", pmax);
    total_pmax += pmax;
  }
  printf("\nTotal pmax: %d\n\n", total_pmax);

  printf("Target energy mix:\n");
  for (double em : this->problem.target_energy_mix) {
    printf("%.2f ", em);
  }
  printf("\n");

  printf("Energy mix:\n");
  for (double em : energy_mix) {
    printf("%.2f ", em);
  }
  printf("\n");

  double error = 0.0;
  for (int i=0; i<this->problem.NB_TYPES; i++){
    error += abs(this->problem.target_energy_mix[i] - energy_mix[i]);
  }
  printf("Difference between target and actual energy mix: %.2f%%\n", error);

  printf("-------------------------------------------------------------\n");
}

void Algorithm::save_result(Result result, string path) {
  ofstream file;
  file.open(path);
  for (int i=0; i<this->problem.N; i++) {
    file << result.x[i];

    if (i != this->problem.N-1)
      file << '\n';
  }
   file.close();
}