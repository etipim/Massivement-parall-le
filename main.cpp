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

    const int test_dimension = 10;
    const int test_pop_size = 30;
    const int test_objectiveFunctions = 1;

    float positions[test_dimension*test_pop_size];
    float velocities[test_pop_size];
    float targets[test_pop_size];
    float g_best[test_dimension];
    float best_target;

    for (int i = 0; i < test_dimension*test_pop_size; ++i) {
        positions[i] = getRandom(Lb, Ub);
    }

    for (int i = 0; i < test_pop_size; ++i) {

        // découpage par bat
        float bat[test_dimension];
        for (int j = 0; j < test_dimension; ++j) {
            bat[j] = positions[i * test_dimension + j];
        }

        // init taget
        targets[i] = host_init_target(bat, test_objectiveFunctions, test_dimension);

        // init vélocité
        velocities[i] = 0;
    }

    // Trouver l'index de la meilleure chauve-souris
    int best_index = index_best_target(targets, test_pop_size);

    // Récupérer les coordonnées de la meilleure chauve-souris
    for (int j = 0; j < test_dimension; ++j) {
        g_best[j] = positions[best_index * test_dimension + j];
    }

    // Vérification
    printf("Initial g_best values:\n");
    for (int j = 0; j < test_dimension; ++j) {
        printf("g_best[%d] = %f\n", j, g_best[j]);
    }

    // Meilleur précision
    best_target = targets[0];

    // Appel de la fonction CUDA
    cuda_pso(positions, velocities, targets, g_best, best_target, test_pop_size, test_dimension, test_objectiveFunctions);

    // Affichage des premières valeurs
    printf("First value of positions: %f\n", positions[0]);
    printf("First value of velocities: %f\n", velocities[0]);
    printf("First value of targets: %f\n", targets[0]);

}


int main()
{
        //BA();
        BA_CUDAS();

        return 0;
}

