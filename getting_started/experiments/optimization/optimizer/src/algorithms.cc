#include "algorithms.hh"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <climits>
#include "omp.h"

/*******************************/
/***********SOLVER**************/
/*******************************/

Solver::Solver() {
  srand(time(NULL));
}

void Solver::set_seed(int &seed) {
  if (seed != -1)
    srand(seed);
}


void Solver::print_result(Result result) {
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
    printf("%f ", pmax);
    total_pmax += pmax;
  }
  printf("\nTotal pmax: %d\n\n", total_pmax);

  printf("Target energy mix:\n");
  for (double em : target_energy_mix) {
    printf("%f ", em);
  }
  printf("\n");

  printf("Energy mix:\n");
  for (double em : energy_mix) {
    printf("%f ", em);
  }
  printf("\n");

  double error = 0.0;
  for (int i=0; i<NB_TYPES; i++){
    error += abs(target_energy_mix[i] - energy_mix[i]);
  }
  printf("Difference between target and actual energy mix: %f%%\n", error);
}

void* solve_thread(void* thread_args) {
  Thread_args *ta;
  ta = (struct Thread_args *) thread_args;
  ta->solver->solve(ta->result, ta->nb_iterations);
  return NULL;
}

void Solver::solve_parallel(Result* result, int* nb_iterations, const int &nb_threads) {
  pthread_t threads[nb_threads];
  Result results[nb_threads];
  Thread_args thread_args[nb_threads];
  int iterations[nb_threads];
  for (int i=0; i<nb_threads; i++) {
    iterations[i] = *nb_iterations / nb_threads;

    thread_args[i].nb_iterations = &iterations[i];
    thread_args[i].result = &results[i];
    thread_args[i].solver = this;
    int tmp = pthread_create(&threads[i], NULL, solve_thread, (void *)&thread_args[i]);
  }

  double min_obj = INT_MAX;
  int min_idx = -1;
  for (int i=0; i<nb_threads; i++) {
    int tmp = pthread_join(threads[i], NULL);
    if (results[i].obj < min_obj){
      min_obj = results[i].obj;
      min_idx = i;
    }
  }
  *result = results[min_idx];
}

/*******************************/
/********RANDOM SOLVER**********/
/*******************************/

void RandomSolver::solve(Result* result, int* nb_iterations) {
  array<int, N> best_x;
  double obj = INT_MAX;

  for(int i=0; i<*nb_iterations; i++){
    array<int, N> x;
    for (int i=0; i<N; i++) {
      x[i] = rand() % NB_TYPES;
    }
    array<double, NB_TYPES> pmaxs      = get_pmaxs(x);
    array<double, NB_TYPES> energy_mix = get_energy_mix(pmaxs);
    double tmp = objective(energy_mix);
    if (tmp < obj) {
      obj = tmp;
      best_x = x;
    }
  }
  result->obj = obj;
  result->x = best_x;
}