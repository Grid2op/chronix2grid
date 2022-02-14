#include <iostream>
#include "algorithms.hh"
#include "parse_args.hh"

int main(int argc, char** argv) {
  int nb_threads    = atoi(getCmdOption(argv, argv + argc, "-t"));
  int nb_iterations = atoi(getCmdOption(argv, argv + argc, "-i"));

  int seed = -1;
  RandomSolver solver;
  solver.set_seed(seed);
  Result result;
  solver.solve_parallel(&result, &nb_iterations, nb_threads);
  solver.print_result(result);
  return 0;
}