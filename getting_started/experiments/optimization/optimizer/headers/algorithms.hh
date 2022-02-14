#include "objective.hh"

struct Result {
  double obj;
  array<int, N> x;
};

class Solver {
  public:
    Solver();
    ~Solver() = default;

    /**
     *  @brief Initialize the seed of the random number generator.
     *  @param seed the seed of the random number generator. Put -1 for default time seed.
    */
    void set_seed(int &seed);

    /**
     *  @brief Dispatch the number of iterations on multiple threads.
     *  @param result the optimization result you want to print.
    */
    void solve_parallel(Result* result, int* nb_iterations, const int &nb_threads);

    /**
     *  @brief Print the optimization result in a pretty way.
     *  @param result the optimization result you want to print.
    */
    void print_result(Result result);

    /**
     *  @brief Call the solver on the optimization problem desribed in constants.hh.
     *  @param result the address where the optimization result will be stored.
     *  @param nb_iterations the number of iterations of the solver
    */
    virtual void solve(Result* result, int* nb_iterations) {};

};

struct Thread_args {
  Result* result;
  int* nb_iterations;
  Solver* solver;
};

/**
 *  @brief  Constructor for a random solver
 *  @return A RandomSolver instance.
*/
class RandomSolver: public Solver {
  public:
    RandomSolver() : Solver() {};
    void solve(Result* result, int* nb_iterations);
    RandomSolver* clone() { return new RandomSolver(*this); }
};