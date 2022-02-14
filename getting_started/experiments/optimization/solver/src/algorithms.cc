#include "algorithms.hh"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <climits>

double dRand(double min, double max) {
  double f = (double)rand() / RAND_MAX;
  return min + f * (max - min);
}

Infos mk_infos(string algorithm, int nb_iterations, double elapsed_time) {
  Infos infos = {algorithm, nb_iterations, elapsed_time};
  return infos;
}


/*******************************/
/***********SOLVER**************/
/*******************************/

Solver::Solver(string name) {
  this->name = name;
  srand(time(NULL));
}

void Solver::set_seed(int seed) {
  if (seed != -1)
    srand(seed);
}

void Solver::solve_parallel(Result &result, int &nb_iterations) {
  double start_time = time(NULL);
  printf("%f\n", start_time);
  
  Result result_final;

  // Get the number of processes
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  this->set_seed(time(NULL) + rank + 1);

  result.infos = mk_infos(this->name, nb_iterations, 0.0);
  this->solve(result, nb_iterations);

  // Find the minimum objective
  MPI_Allreduce(&result.obj, &result_final.obj, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  // Send the best solution to the main core
  MPI_Request r;  
  if (result.obj == result_final.obj && rank != 0) {
    MPI_Isend(&result.x, N, MPI_INT, 0, 99, MPI_COMM_WORLD, &r);
  }
  
  else if (rank == 0) {
    result.obj = result_final.obj;
    MPI_Irecv(&result.x, N, MPI_INT, MPI_ANY_SOURCE, 99, MPI_COMM_WORLD, &r);
  }
}

double Solver::compute_objective(array<int, N> x) {
  array<double, NB_TYPES> pmaxs      = get_pmaxs(x);
  array<double, NB_TYPES> energy_mix = get_energy_mix(pmaxs);
  return objective(energy_mix);
}

/**********************************/
/********RANDOM ALGORITHM**********/
/**********************************/

void RandomSolver::solve(Result &result, int &nb_iterations) {
  Result result2;

  array<int, N> best_x;
  double best_obj = INT_MAX;

  for(int i=0; i<nb_iterations; i++) {
    array<int, N> x;
    for (int i=0; i<N; i++) {
      x[i] = rand() % NB_TYPES;
    }
    double obj = this->compute_objective(x);
    if (best_obj > obj) {
      best_obj = obj;
      copy(x.begin(), x.end(), best_x.begin());
    }
  }
  result.obj = best_obj;
  result.x = best_x;
}

/***********************************************/
/********SIMULATED ANNEALING ALGORITHM**********/
/***********************************************/

double SimulatedAnnealing::acceptance_probability(double e, double e_new, double T) {
  if (e_new < e) return 1.0;
  return exp(-(e_new - e) / T);
}

double SimulatedAnnealing::compute_T(double Tmin, double Tmax, double decay, double nb_it) {
  return Tmin + (Tmax - Tmin) * exp(-decay * nb_it);
}

void SimulatedAnnealing::solve(Result &result, int &nb_iterations) {
  /*
   * Initialize x with a random solver
  */
  RandomSolver solver;
  int nb_iter = nb_iterations / 4;
  solver.solve(result, nb_iter);
  array<int, N> best_x = result.x;
  double best_obj = this->compute_objective(best_x);

  double Tmin = 0.001;
  double Tmax = 10000;
  double decay = 1.0 / nb_iterations * 10;
  double T = this->compute_T(Tmin, Tmax, decay, 0);

  for(int i=0; i<nb_iterations; i++) {
    int idx = rand() % N;
    int rand_type = rand() % NB_TYPES;

    array<int, N> tmp_x = best_x;
    tmp_x[idx] = rand_type;
    double obj = this->compute_objective(tmp_x);

    double acceptance_prob = this->acceptance_probability(best_obj, obj, T);

    if (dRand(0, 1) <= acceptance_prob) {
      best_obj = obj;
      best_x = tmp_x;
    }
    T = this->compute_T(Tmin, Tmax, decay, i+1);
  }
  result.obj = best_obj;
  result.x = best_x;
}