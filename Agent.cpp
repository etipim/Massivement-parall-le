//
// Created by etipi on 15/01/2024.
//

#include "Agent.h"

//Agent::Agent(double loudness_min, double loudness_max, double pr_min, double pr_max, double pf_min, double pf_max)
//{
//    // Génération d'un nombre aléatoire selon une distribution uniforme
//    std::uniform_real_distribution<double> distributionLoudness(loudness_min, loudness_max);
//    std::uniform_real_distribution<double> distributionPr(pr_min, pr_max);
//    std::uniform_real_distribution<double> distributionPf(pf_min, pf_max);
//
//    this->loudness = distributionLoudness(this->generator);
//    this->pulse_rate = distributionPr(this->generator);
//    this->pulse_frequency = distributionPf(this->generator);
//}

Agent::Agent(double pf_min, double pf_max)
{
    // Génération d'un nombre aléatoire selon une distribution uniforme

    std::uniform_real_distribution<double> distributionPf(pf_min, pf_max);


    this->pulse_frequency = distributionPf(this->generator);
}

void Agent::get_fitness(int fit_function) {
    double sum = 0.0;

    for (int i = 0; i < this->solution.size(); ++i) {

        // Sphère
        if (fit_function == 1) {
            sum += this->solution[i]*this->solution[i]; // (bat->solution[i] - 2) * (bat->solution[i] - 2);
        }

            // Rastrigin
        else if (fit_function == 2) {
            sum += this->solution[i] * this->solution[i] - 10.0 * std::cos(2.0 * M_PI * this->solution[i]) + 10.0;
        }

            // Rosenbrock
        else if (fit_function == 3) {
            if (i < this->solution.size() - 1) {
                sum += 100.0 * std::pow((this->solution[i + 1] - std::pow(this->solution[i], 2)), 2) +
                       std::pow((this->solution[i] - 1.0), 2);
            }
        }

            // Ackley
        else if (fit_function == 4) {
            double term1 = -20.0 * std::exp(-0.2 * std::sqrt(0.5 * (std::pow(this->solution[i], 2) + std::pow(this->solution[i + 1], 2))));
            double term2 = -std::exp(0.5 * (std::cos(2.0 * M_PI * this->solution[i]) + std::cos(2.0 * M_PI * this->solution[i + 1])));
            sum += term1 + term2 + 20.0 + std::exp(1.0);
            i++; // Because Ackley function is typically defined for two dimensions, we increment i again.
        }
    }

    this->fitness = sum;
}

//void Agent::setSolution(double m) { solution = m; }
//void Agent::setVelocity(double m) { velocity = m; }