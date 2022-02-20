#include <stdexcept>
#include <memory>
#include "algorithms.hh"
#include "parse_args.hh"
#include "utils.hh"

int main(int argc, char** argv) {
  /* MPI initialisation */
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0)
    print_cpp_version();

  string save_path = getCmdOption(argv, argv + argc, "-p");
  string json_path = getCmdOption(argv, argv + argc, "-js");

  Problem problem(json_path);

  unique_ptr<Algorithm> algo;
  if (problem.algorithm == "random")
    algo = make_unique<RandomAlgorithm>(problem);
  else if (problem.algorithm == "simulated annealing")
    algo = make_unique<SimulatedAnnealing>(problem);
  else {
    char error[50];
    sprintf(error, "Unknown \"%s\" algorithm.", problem.algorithm.c_str());
    throw invalid_argument(error);
  }

  Result result;

  Result result_core;
  algo->run_parallel(result_core);

  if (rank == 0) {
    result.obj   = result_core.obj;
    result.x     = result_core.x;
    result.infos = result_core.infos;
    algo->print_result(result);
    algo->save_result(result, save_path);
  }
  MPI_Finalize();
  return 0;
}