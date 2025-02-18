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

//void BA() {
//
//    std::vector<int> dimensions = {10};//{10, 30, 50};
//    std::vector<int> popSizes = {30};//{30, 50, 70};
//    std::vector<int> objectiveFunctions = {1};// {1, 2, 3, 4};
//
//    for (int objFunc : objectiveFunctions) {
//        for (int dim : dimensions) {
//            for (int popSize : popSizes) {
//
//                // Save the best fitness for each test to a file
//                std::ofstream resultFile("test_result_obj_" + std::to_string(objFunc) +
//                                         "_dim_" + std::to_string(dim) +
//                                         "_pop_" + std::to_string(popSize) + ".txt");
//                for (int testNum = 1; testNum <= 10; ++testNum) {
//                    OriginalBA* originalBa = new OriginalBA(5000, popSize, 10.0, -10.0, 0.95, 0.0, 10.0, dim, objFunc);
//                    originalBa->generate_population();
//                    originalBa->search_best_sol();
//                    originalBa->solve();
//                    double result = originalBa->result();
//
//
//                    resultFile << "Best Fitness: " << std::setprecision(15) << result << std::endl;
//
//
////                    std::cout << "Objective Function: " << objFunc
////                              << ", Dimension: " << dim
////                              << ", Pop Size: " << popSize
////                              << ", Test: " << testNum
////                              << ", Best Fitness: " << result << std::endl;
//                }
//
//                resultFile.close();
//            }
//        }
//    }
//
//
//}

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
                    double positions[dim * popSize];
                    double velocities[dim * popSize];
                    double targets[popSize];
                    double g_best[dim];
                    double best_target;

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
                    printf("Print final target : %f \n", best_target);

                    // Ajout du résultat en colonne et ajout du temps GPU total
                    resultFile << "," << host_init_target(g_best, objFunc, dim) << "," << gpu_time / 10;
                }

                resultFile << "\n";
            }
        }
    }
    resultFile.close();/*

    const int test_dimension = 10;
    const int test_pop_size = 30;
    const int test_objectiveFunctions = 1;

    double positions[test_dimension*test_pop_size];
    double velocities[test_dimension*test_pop_size];
    double targets[test_pop_size];
    double g_best[test_dimension];
    double best_target;

    for (int i = 0; i < test_dimension*test_pop_size; ++i) {
        // init positions pop
        positions[i] = getRandom(Lb, Ub);

        // init vélocité
        velocities[i] = 0;
    }

    for (int i = 0; i < test_pop_size; ++i) {

        // découpage par bat
        double bat[test_dimension];
        for (int j = 0; j < test_dimension; ++j) {
            bat[j] = positions[i * test_dimension + j];
        }

        // init taget
        targets[i] = host_init_target(bat, test_objectiveFunctions, test_dimension);


    }

    // Trouver l'index de la meilleure chauve-souris
    int best_index = index_best_target(targets, test_pop_size);

    // Récupérer les coordonnées de la meilleure chauve-souris
    for (int j = 0; j < test_dimension; ++j) {
        g_best[j] = positions[best_index * test_dimension + j];
    }

    // Meilleur précision
    best_target = host_init_target(g_best, test_objectiveFunctions, test_dimension);

    // Appel de la fonction CUDA
    clock_t begin = clock();
    cuda_pso(positions, velocities, targets, g_best, best_target, test_pop_size, test_dimension, test_objectiveFunctions);
    clock_t end = clock();

    printf("GPU \t ");
    printf("%10.3lf \t", (double)(end - begin)/CLOCKS_PER_SEC);*/

}


int main()
{
    //BA();
    BA_CUDAS();

    return 0;
}

