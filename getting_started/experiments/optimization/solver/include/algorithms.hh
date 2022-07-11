#pragma once

#include "problem.hh"

struct Infos {
  string algorithm;
  int nb_iterations;
  double elapsed_time;
};

static Infos mk_infos(string algorithm, int nb_iterations, double elapsed_time) {
  Infos infos = {algorithm, nb_iterations, elapsed_time};
  return infos;
};

struct Result {
  double obj;
  vector<int> x;
  Infos infos;
};

class Algorithm {
  public:
    Algorithm(string name, Problem problem);
    ~Algorithm() = default;

    /**
     *  @brief      Initialize the seed of the random number generator.
     *  @param seed the seed of the random number generator. Put -1 for default time seed.
    */
    void set_seed(int seed);

    /**
     *  @brief        Print the optimization result in a pretty way.
     *  @param result the optimization result you want to print.
    */
    void print_result(Result result);

    /**
     *  @brief        Save the optimization result in a text file.
     *  @param result the optimization result you want to save.
    */
    void save_result(Result result, string path);

    /**
     *  @brief               Call the solver on the optimization problem desribed in constants.hh using multiple cores.
     *  @param result        the address where the optimization result will be stored.
    */
    void run_parallel(Result &result);

    /**
     *  @brief   Compute the objective of optimization problem desribed in constants.hh.
     *  @param x The parameters of the optimization problem.
     *  @return  The value of the objective function.
    */
    double compute_objective(vector<int> x);

    virtual void run(Result &result) {printf("Virtual\n");};
  
  private:
    string name;
  protected:
    Problem problem;

};

/**
 *  @brief         Constructor for a random algorithm
 *  @param problem The optimization problem to solve
 *  @return        A RandomAlgorithm instance.
*/
class RandomAlgorithm: public Algorithm {
  public:
    RandomAlgorithm(Problem problem) : Algorithm("Random solver", problem) {};
    void run(Result &result);
};

/**
 *  @brief         Constructor for a simulated annealing algorithm
 *  @param problem The optimization problem to solve
 *  @return        A SimulatedAnnealing instance.
*/
class SimulatedAnnealing: public Algorithm {
  public:
    SimulatedAnnealing(Problem problem) : Algorithm("Simulated annealing", problem) {};
    void run(Result &result);
  private:
    double acceptance_probability(double e, double e_new, double T);
    double compute_T(double Tmin, double Tmax, double decay, double nb_it);
};