//#include "WOA.h"
//#include "BA.h"
#include "kernel.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include "iomanip"

void BA_CUDAS() {
    // Ouverture du fichier CSV unique
    std::ofstream resultFile("results.csv");

    // Écriture des en-têtes avec chaque itération en colonne
    resultFile << "ObjectiveFunction,Dimension,PopulationSize";
    for (int iter = 0; iter < 10; ++iter) {
        resultFile << ",Iteration_" << iter + 1 << ",GPU_Time";
    }
    resultFile << "\n";

    for (int objFunc : objectiveFunctions) {
        for (int dim : dimensions) {
            for (int popSize : popSizes) {
                resultFile << objFunc << "," << dim << "," << popSize;

                for (int iter = 0; iter < 10; ++iter) {
                    //double positions[dim * popSize];
                    //double velocities[dim * popSize];
                    //double targets[popSize];
                    //double g_best[dim];
                    double best_target;

                    double* positions = new double[dim * popSize];
                    double* velocities = new double[dim * popSize];
                    double* targets = new double[popSize];
                    double* g_best = new double[dim];

                    for (int i = 0; i < dim * popSize; ++i) {
                        // Init positions pop
                        positions[i] = getRandom(Lb, Ub);
                        // Init velocities
                        velocities[i] = 0;
                    }

                    // Init targets
                    for (int i = 0; i < popSize; ++i) {
                        double bat[dim];
                        for (int j = 0; j < dim; ++j) {
                            bat[j] = positions[i * dim + j];
                        }
                        targets[i] = host_init_target(bat, objFunc, dim);
                    }

                    // Trouver la meilleure solution
                    int best_index = index_best_target(targets, popSize);
                    for (int j = 0; j < dim; ++j) {
                        g_best[j] = positions[best_index * dim + j];
                    }
                    best_target = host_init_target(g_best, objFunc, dim);

                    // Exécution GPU
                    clock_t begin = clock();
                    cuda_pso(positions, velocities, targets, g_best, best_target, popSize, dim, objFunc);
                    clock_t end = clock();

                    double gpu_time = (double)(end - begin) / CLOCKS_PER_SEC;

                    // Affichage console
                    printf("GPU Time: %10.3lf s\n", gpu_time);
                    printf("Print final target : %f \n", host_init_target(g_best, objFunc, dim));

                    // Ajout du résultat en colonne et ajout du temps GPU total
                    resultFile << "," << host_init_target(g_best, objFunc, dim) << "," << gpu_time;

                    delete[] positions;
                    delete[] velocities;
                    delete[] targets;
                    delete[] g_best;
                }

                resultFile << "\n";
            }
        }
    }
    resultFile.close();
}


int main()
{
    BA_CUDAS();

    return 0;
}

