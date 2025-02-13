#include <vector>

const int epoch = 5000;
const float Ub = 10;
const float Lb = -10;
const float pulse_rate = 0.95;
const float pfMin = 0.0;
const float pfMax = 10.0;
const float phi = 3.1415;

const std::vector<int> dimensions = {10};//{10, 30, 50};
const std::vector<int> popSizes = {30};//{30, 50, 70};
const std::vector<int> objectiveFunctions = {1};// {1, 2, 3, 4};

// calcul fitness
float host_init_target(float *bat, int fit_function, int dimension);


// Fonctions random
float getRandom(float low, float high);
float getRandomClamped();

// Return best target index
int index_best_target(float *target, int pop_size);

// Fonction externe qui va tourner sur le GPU
extern "C" void cuda_pso(float* positions, float* velocities, float* targets, float* g_best,float best_target, int popSize, int dimention, int fit_function);