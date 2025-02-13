#include "kernel.h"
#include <random>

// Calcul de la fonction objective
float host_init_target(float *bat, int fit_function, int dimension)
{
    float sum = 0;

    for (int i = 0; i < dimension; ++i) {
        // Sphère
        if (fit_function == 1) {
            sum += powf((bat[i]), 2); // (bat->solution[i] - 2) * (bat->solution[i] - 2);
        }

            // Rastrigin
        else if (fit_function == 2) {
            sum += bat[i] * bat[i] - 10 * cosf(2 * phi * bat[i]) + 10;
        }

            // Rosenbrock
        else if (fit_function == 3) {
            if (i < dimension - 1) {
                sum += 100 * powf((bat[i + 1] - powf(bat[i], 2)), 2) +
                       powf((1 - bat[i]), 2);
            }
        }
    }

    return sum;
}

// Obtenir un random entre low et high
float getRandom(float low, float high) {
    return low + (high - low) * (rand() / float(RAND_MAX));
}
// Obtenir un random entre 0.0f and 1.0f inclusif
float getRandomClamped() {
    float new_rand = rand() / float(RAND_MAX);
    return new_rand;
}

// Obtenir la meilleur solution
int index_best_target(float *target, int pop_size)
{
    int best = target[0];
    int best_index = 0;
    for (int i = 1; i < pop_size; ++i) {
        if (target[i] < best){
            best = target[i];
            best_index = i;
        }
    }

    return best_index;
}