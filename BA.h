//
// Created by etipi on 09/01/2024.
//

#ifndef BAT_ALGORITHM_BA_H
#define BAT_ALGORITHM_BA_H

#include <string>
#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>
#include <random>
#include <stdlib.h>
#include <time.h>
#include <chrono>

#include "Agent.h"


class OriginalBA {
public:
    OriginalBA(int epoch = 1000, int popSize = 50, double Ub = 10, double Lb = -10, double pulse_rate = 0.95, double pfMin = 0., double pfMax = 10., int dimention = 30, int fit_function = 1);
    void generate_population();
    void search_best_sol();
    void solve();
    double result();
private:
    Agent* init_bat(); //std::vector<int> init_bat();
    double init_target(Agent* bat);
    int index_best_target();
    double random(double min, double max);
    std::vector<double> init_velo();


    Agent generateEmptyAgent(int solution);
    void evolve();



    std::vector<Agent*> pop;  //std::vector<std::vector<int>> pop;
    std::vector<double> target;
    std::vector<std::vector<double>> velocity;

    Agent* g_best;
    double best_target;
    int fit_function;

    int epoch;
    int popSize;
    double loudness;
    double pulse_rate;
    double pfMin;
    double pfMax;
    int dimention;

    double Lb; //lower bounds
    double Ub; //Upper bounds

    // Déclaration et initialisation du générateur de nombres aléatoires
    std::default_random_engine generator;
};

#endif //BAT_ALGORITHM_BA_H
