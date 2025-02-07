
const float phi = 3.1415;

// Fonctions random
float getRandom(float low, float high);
float getRandomClamped();

// Fonction externe qui va tourner sur le GPU
extern "C" void cuda_pso(int epoch = 1000, int popSize = 50, double Ub = 10, double Lb = -10, double pulse_rate = 0.95, double pfMin = 0., double pfMax = 10., int dimention = 30, int fit_function = 1);