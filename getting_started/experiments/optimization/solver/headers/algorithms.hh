#include "objective.hh"

struct Infos {
  string algorithm;
  int nb_iterations;
  double elapsed_time;
};

struct Result {
  double obj;
  array<int, N> x;
  Infos infos;
};

class Solver {
  public:
    Solver(string name);
    ~Solver() = default;

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
     *  @brief               Call the solver on the optimization problem desribed in constants.hh using multiple cores.
     *  @param result        the address where the optimization result will be stored.
     *  @param nb_iterations the number of iterations of the solver
    */
    void solve_parallel(Result &result, int &nb_iterations);

    /**
     *  @brief   Compute the objective of optimization problem desribed in constants.hh.
     *  @param x The parameters of the optimization problem.
     *  @return  The value of the objective function.
    */
    double compute_objective(array<int, N> x);

    virtual void solve(Result &result, int &nb_iterations) {};
  
  private:
    string name;

};

/**
 *  @brief  Constructor for a random solver
 *  @return A RandomSolver instance.
*/
class RandomSolver: public Solver {
  public:
    RandomSolver() : Solver("Random solver") {};
    void solve(Result &result, int &nb_iterations);
};

/**
 *  @brief  Constructor for a simulated annealing solver
 *  @return A RandomSolver instance.
*/
class SimulatedAnnealing: public Solver {
  public:
    SimulatedAnnealing() : Solver("Simulated annealing") {};
    void solve(Result &result, int &nb_iterations);
  private:
    double acceptance_probability(double e, double e_new, double T);
    double compute_T(double Tmin, double Tmax, double decay, double nb_it);
};