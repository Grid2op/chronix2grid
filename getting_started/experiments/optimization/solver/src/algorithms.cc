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
  result.infos.elapsed_time = (get_time_milli() - start_time) / 1000.0;

  // Find the minimum objective and share it with every process
  MPI_Allreduce(&result.obj, &result_final.obj, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  /*
   * Now that each process knows the minimum objective, we still need to send
   * the optimal variables associated to this objective.
   * To avoid using non-blocking send/receive, every process except the main one
   * will send its variables to the main process BUT if its objective is not
   * the same than the minimum found earlier (i.e this process did not found the optimum),
   * it will set its first variable to -1 (an impossible value). This way, when the
   * main process receives a array of variables, it checks if the first is -1.
   * If yes, it discards the array, otherwise it keeps it.
   * We can synchronize the processes at the end.
  */
  int discriminant = -1;
  if (rank != 0) {
    if (result.obj != result_final.obj)
      result.x[0] = discriminant;

    MPI_Send(&result.x[0], this->problem.N, MPI_INT, 0, 99, MPI_COMM_WORLD);
  }
  
  else {
    result.obj = result_final.obj;
  
    MPI_Status s;
    for(int i = 1; i < size; i++) {
      vector<int> tmp_x(this->problem.N);
      MPI_Recv(&tmp_x[0], this->problem.N, MPI_INT, i, 99, MPI_COMM_WORLD, &s);
      if (tmp_x[0] != discriminant)
        result.x = tmp_x;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

/**********************************/
/********RANDOM ALGORITHM**********/
/**********************************/

void RandomAlgorithm::run(Result &result) {
  vector<int> best_x(this->problem.N);
  double best_obj = this->problem.objective(best_x);

  for(int i = 0; i < this->problem.nb_iterations; i++) {
    vector<int> x(this->problem.N);
    for (int i = 0; i < this->problem.N; i++) {
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

  for(int i = 0; i < this->problem.nb_iterations; i++) {
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