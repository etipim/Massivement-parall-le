#include "kernel.h"
#include <random>

// Calcul de la fonction objective
double host_init_target(double *bat, int fit_function, int dimension)
{
    double sum = 0;

    double sum1 = 0;
    double sum2 = 0;

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

            // Ackley
        else if (fit_function == 4) {
            sum1 += std::pow(bat[i], 2);
            sum2 += std::cos(2.0 * phi * bat[i]);
        }
    }

    if (fit_function == 4) {
        double term1 = -20.0 * std::exp(-0.2 * std::sqrt(sum1 / dimension));
        double term2 = -std::exp(sum2 / dimension);

        sum = term1 + term2 + 20.0 + std::exp(1.0);
    }

    return sum;
}

// Obtenir un random entre low et high
double getRandom(double low, double high) {
    return low + (high - low) * (rand() / double(RAND_MAX));
}
// Obtenir un random entre 0.0f and 1.0f inclusif
double getRandomClamped() {
    double new_rand = rand() / double(RAND_MAX);
    return new_rand;
}

// Obtenir la meilleur solution
int index_best_target(double *target, int pop_size)
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