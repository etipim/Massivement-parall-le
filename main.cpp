#include "WOA.h"
#include "BA.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include "iomanip"

double calcul(const std::vector<double>& solution) {
    double s = 0;
    for (int i = 0; i < solution.size(); i++) {
        s += pow(solution[i], 2);
    }
    return s;
}

void BA() {

    std::vector<int> dimensions = {10};//{10, 30, 50};
    std::vector<int> popSizes = {30};//{30, 50, 70};
    std::vector<int> objectiveFunctions = {1};// {1, 2, 3, 4};

    for (int objFunc : objectiveFunctions) {
        for (int dim : dimensions) {
            for (int popSize : popSizes) {

                // Save the best fitness for each test to a file
                std::ofstream resultFile("test_result_obj_" + std::to_string(objFunc) +
                                         "_dim_" + std::to_string(dim) +
                                         "_pop_" + std::to_string(popSize) + ".txt");
                for (int testNum = 1; testNum <= 10; ++testNum) {
                    OriginalBA* originalBa = new OriginalBA(5000, popSize, 10.0, -10.0, 0.95, 0.0, 10.0, dim, objFunc);
                    originalBa->generate_population();
                    originalBa->search_best_sol();
                    originalBa->solve();
                    double result = originalBa->result();


                    resultFile << "Best Fitness: " << std::setprecision(15) << result << std::endl;


//                    std::cout << "Objective Function: " << objFunc
//                              << ", Dimension: " << dim
//                              << ", Pop Size: " << popSize
//                              << ", Test: " << testNum
//                              << ", Best Fitness: " << result << std::endl;
                }

                resultFile.close();
            }
        }
    }


}


int main()
{
        BA();

        return 0;
}

