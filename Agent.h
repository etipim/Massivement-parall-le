//
// Created by etipi on 15/01/2024.
//

#ifndef BAT_ALGORITHM_AGENT_H
#define BAT_ALGORITHM_AGENT_H


#include <string>
#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>
#include <random>
#include <stdlib.h>
#include <time.h>

class Agent {
public:
    std::vector<double> solution;
    std::vector<double> velocity;
    double fitness;
    double pulse_frequency;

    // Déclaration et initialisation du générateur de nombres aléatoires
    std::default_random_engine generator;


    //Agent(double loudness_min=0.7, double loudness_max=0.9, double pr_min=0.0, double pr_max=0.85, double pf_min=0.1, double pf_max=10);
    Agent(double pf_min=0.1, double pf_max=10);

    void get_fitness(int fit_function);
    void setSolution(double m);
    void setVelocity(double m);
};

#endif //BAT_ALGORITHM_AGENT_H
