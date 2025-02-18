#include <vector>

const int epoch = 5000;
const double Ub = 10;
const double Lb = -10;
const double pulse_rate = 0.95;
const double pfMin = 0.0;
const double pfMax = 10.0;
const double phi = 3.14159265358979323846;

const std::vector<int> dimensions = {10, 30, 50};
const std::vector<int> popSizes = {30, 50, 70};
const std::vector<int> objectiveFunctions = {1, 2, 3, 4};

// calcul fitness
double host_init_target(double *bat, int fit_function, int dimension);


// Fonctions random
double getRandom(double low, double high);
double getRandomClamped();

// Return best target index
int index_best_target(double *target, int pop_size);

// Fonction externe qui va tourner sur le GPU
void cuda_pso(double* positions, double* velocities, double* targets, double* g_best,double best_target, int popSize, int dimention, int fit_function);