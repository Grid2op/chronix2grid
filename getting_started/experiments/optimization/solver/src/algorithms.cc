#include "algorithms.hh"
#include <cmath>
#include <stdlib.h>
#include <climits>
#include <chrono>
using namespace chrono;

double dRand(double min, double max) {
  double f = (double)rand() / RAND_MAX;
  return min + f * (max - min);
}

int get_time_sec() {
  return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

double get_time_milli() {
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}


/*******************************/
/***********SOLVER**************/
/*******************************/

Algorithm::Algorithm(string name, Problem problem) {
  this->name = name;
  this->problem = problem;
  srand(get_time_sec());
}

void Algorithm::set_seed(int seed) {
  if (seed != -1)
    srand(seed);
}

void Algorithm::run_parallel(Result &result) {
  double start_time = get_time_milli();
  
  Result result_final;

  // Get the number of processes
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Get the rank of the process
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  this->set_seed(get_time_sec() + rank + 1);

  result.infos = mk_infos(this->name, this->problem.nb_iterations, 0.0);
  this->run(result);

  // Find the minimum objective
  MPI_Allreduce(&result.obj, &result_final.obj, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  // Send the best solution to the main core
  MPI_Request r;  
  if (result.obj == result_final.obj && rank != 0) {
    MPI_Isend(&result.x, this->problem.N, MPI_INT, 0, 99, MPI_COMM_WORLD, &r);
  }
  
  else if (rank == 0) {
    result.obj = result_final.obj;
    MPI_Irecv(&result.x, this->problem.N, MPI_INT, MPI_ANY_SOURCE, 99, MPI_COMM_WORLD, &r);
    result.infos.elapsed_time = (get_time_milli() - start_time) / 1000;
  }
}

/**********************************/
/********RANDOM ALGORITHM**********/
/**********************************/

void RandomAlgorithm::run(Result &result) {
  vector<int> best_x(this->problem.N);
  double best_obj = this->problem.objective(best_x);

  for(int i=0; i<this->problem.nb_iterations; i++) {
    vector<int> x(this->problem.N);
    for (int i=0; i<this->problem.N; i++) {
      x[i] = rand() % this->problem.NB_TYPES;
    }
    double obj = this->problem.objective(x);
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

void SimulatedAnnealing::run(Result &result) {
  /*
   * Initialize x with a random solver
  */
  int nb_iter = this->problem.nb_iterations;
  this->problem.nb_iterations /= 4;
  RandomAlgorithm solver(this->problem);
  solver.run(result);
  this->problem.nb_iterations = nb_iter;
  vector<int> best_x = result.x;
  double best_obj = this->problem.objective(best_x);

  double Tmin = 0.001;
  double Tmax = 10000;
  double decay = 1.0 / this->problem.nb_iterations * 10;
  double T = this->compute_T(Tmin, Tmax, decay, 0);

  for(int i=0; i<this->problem.nb_iterations; i++) {
    int idx = rand() % this->problem.N;
    int rand_type = rand() % this->problem.NB_TYPES;

    vector<int> x = best_x;
    x[idx] = rand_type;
    double obj = this->problem.objective(x);

    double acceptance_prob = this->acceptance_probability(best_obj, obj, T);

    if (dRand(0, 1) <= acceptance_prob) {
      best_obj = obj;
      best_x = x;
    }
    T = this->compute_T(Tmin, Tmax, decay, i+1);
  }
  result.obj = best_obj;
  result.x = best_x;
}