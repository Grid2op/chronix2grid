#include <iostream>
#include <vector>
#include "mpi.h"
using namespace std;

class Problem {
  public:
    Problem() = default;
    Problem(string name);
    void init(string name);

    /**
     *  @brief   Compute the objective of optimization problem desribed in constants.hh.
     *  @param x The parameters of the optimization problem.
     *  @return  The value of the objective function.
    */
    double objective(vector<int> x);

    vector<double> get_pmaxs(vector<int> x);
    vector<double> get_energy_mix(vector<double> pmaxs);

    /* Attributes */
    string algorithm;
    int nb_iterations;

    int N;
    int NB_TYPES;
    int average_load;
    vector<string> power_plant_types;
    vector<double> avg_pmaxs;
    vector<double> capacity_factor;
    vector<double> target_energy_mix;

};