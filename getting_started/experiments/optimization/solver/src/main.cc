#include <iostream>
#include "algorithms.hh"
#include "parse_args.hh"

int main(int argc, char** argv) {

  int nb_iterations = atoi(getCmdOption(argv, argv + argc, "-i"));

  SimulatedAnnealing solver;
  Result result;
  
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Result result_core;
  solver.solve_parallel(result_core, nb_iterations);

  if (rank == 0) {
    result.obj   = result_core.obj;
    result.x     = result_core.x;
    result.infos = result_core.infos;
    solver.print_result(result);
  }

  MPI_Finalize();
  return 0;
}