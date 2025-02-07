//
// Created by etipi on 09/01/2024.
//

#include "BA.h"
#include "Agent.h"
#include <random>

OriginalBA::OriginalBA(int epoch, int popSize, double Ub, double Lb, double pulse_rate, double pfMin, double pfMax, int dimention, int fit_function)
{
    this->epoch = epoch;
    this->popSize = popSize;
    this->loudness = loudness;
    this->pulse_rate = pulse_rate;
    this->pfMin = pfMin;
    this->pfMax = pfMax;

    this->fit_function = fit_function;

    this->Lb = Lb;
    this->Ub = Ub;

    // dimention
    this->dimention = dimention;
}

// générer la population
void OriginalBA::generate_population()
{
    for (int i = 0; i < this->popSize; ++i) {
        this->pop.push_back(init_bat());
        this->target.push_back(init_target(pop[i]));

        this->velocity.push_back(init_velo());
    }

    this->best_target = this->target[0];
}

std::vector<double> OriginalBA::init_velo()
{
    std::vector<double> velo;
    for (int j = 0; j < dimention; ++j) {
        velo.push_back(0);
    }
    return velo;
}

// avoir la meilleur solution
void OriginalBA::search_best_sol()
{
    this->g_best = this->pop[index_best_target()];
}

// Algo global
void OriginalBA::solve()
{
    int t = 0;
    while (t < this->epoch)
    {

        std::vector<Agent*> pop_new;
        for (int idx = 0; idx < this->popSize; ++idx) {
            // init agent
            Agent* agent = new Agent();
            agent->solution = this->pop[idx]->solution;

            // change pulse frequency
            double pulse_frequency = this->pfMin + (this->pfMax - this->pfMin) * random(0., 1.); // Eq. 2


            Agent* x_new = new Agent();
            x_new->solution = this->pop[idx]->solution;
            for (int i = 0; i < this->dimention; ++i) {
                // change velocity
                this->velocity[idx][i] = random(0., 1) * this->velocity[idx][i] + (this->g_best->solution[i] - this->pop[idx]->solution[i]) * pulse_frequency; // Eq. 3

                // Update the solution based on the updated velocity
                double new_solution = x_new->solution[i] + this->velocity[idx][i];

                // change solution and insert born
                if (new_solution > this->Ub){
                    x_new->solution[i] = this->Ub;
                }
                else if (new_solution < this->Lb){
                    x_new->solution[i] = this->Lb;
                }
                else {
                    x_new->solution[i] = new_solution;  // Eq. 4
                }
            }

            agent->solution = x_new->solution;

            pop_new.push_back(agent);
        }


        std::vector<int> pop_child_index;
        std::vector<Agent*> pop_child;
        for (int idx = 0; idx < this->popSize; ++idx) {
            if (this->target[idx] > init_target(pop_new[idx])) {
                this->pop[idx] = pop_new[idx];
                this->target[idx] = init_target(pop_new[idx]);
            } else {
                if (random(0, 1) > this->pulse_rate) {
                    Agent* agent = new Agent();
                    agent->solution = this->pop[idx]->solution;
                    //agent->velocity = this->pop[idx]->velocity;
                    for (int i = 0; i < this->dimention; ++i) {
                        // Update the solution based on the updated velocity
                        double new_solution = this->g_best->solution[i] + 0.01 * random(this->Lb, this->Ub);

                        if (new_solution > this->Ub){
                            agent->solution[i] = this->Ub;
                        }
                        else if (new_solution < this->Lb){
                            agent->solution[i] = this->Lb;
                        }
                        else {
                            agent->solution[i] = new_solution;
                        }
                    }
                    pop_child_index.push_back(idx);
                    pop_child.push_back(agent);
                }
            }
        }

        for (int idx = 0; idx < pop_child_index.size(); ++idx) {
            if (init_target(pop_new[pop_child_index[idx]]) > init_target(pop_child[idx])) {
                pop_new[pop_child_index[idx]]->solution = pop_child[idx]->solution;
            }
        }

        this->pop = pop_new;
        for (int idx = 0; idx < this->popSize; ++idx) {
            this->target[idx] = init_target(this->pop[idx]);
            if (this->target[idx] < this->best_target){
                this->best_target = this->target[idx];
            }
        }

        if (init_target(this->g_best) > init_target(this->pop[index_best_target()])){
            this->g_best = this->pop[index_best_target()];
        }

        t +=1;
    }
}

// Créer la position de la chauve souris
Agent* OriginalBA::init_bat()
{
    // Génération d'un nombre aléatoire selon une distribution uniforme
    std::uniform_real_distribution<double> distribution(this->Lb, this->Ub);

    Agent* bats = new Agent();

    for (int i = 0; i < this->dimention; ++i) {
        bats->solution.push_back(distribution(this->generator));
        bats->velocity.push_back(0);
    }

    return bats;
}

// Calcul de la fonction objective
double OriginalBA::init_target(Agent* bat)
{
    double sum = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < bat->solution.size(); ++i) {

        // Sphère
        if (fit_function == 1) {
            sum += std::pow((bat->solution[i]), 2); // (bat->solution[i] - 2) * (bat->solution[i] - 2);
        }

            // Rastrigin
        else if (fit_function == 2) {
            sum += bat->solution[i] * bat->solution[i] - 10.0 * std::cos(2.0 * M_PI * bat->solution[i]) + 10.0;
        }

            // Rosenbrock
        else if (fit_function == 3) {
            if (i < bat->solution.size() - 1) {
                sum += 100.0 * std::pow((bat->solution[i + 1] - std::pow(bat->solution[i], 2)), 2) +
                       std::pow((1.0 - bat->solution[i]), 2);
                       //std::pow((bat->solution[i] - 1.0), 2);
            }
        }

            // Ackley
        else if (fit_function == 4) {
            sum1 += std::pow(bat->solution[i], 2);
            sum2 += std::cos(2.0 * M_PI * bat->solution[i]);

//            double term1 = -20.0 * std::exp(-0.2 * std::sqrt(0.5 * (std::pow(bat->solution[i], 2) + std::pow(bat->solution[i + 1], 2))));
//            double term2 = -std::exp(0.5 * (std::cos(2.0 * M_PI * bat->solution[i]) + std::cos(2.0 * M_PI * bat->solution[i + 1])));
//            sum += term1 + term2 + 20.0 + std::exp(1.0);
        }
    }

    if (fit_function == 4) {
        double term1 = -20.0 * std::exp(-0.2 * std::sqrt(sum1 / bat->solution.size()));
        double term2 = -std::exp(sum2 / bat->solution.size());

        sum = term1 + term2 + 20.0 + std::exp(1.0);
    }

    return sum;
}

int OriginalBA::index_best_target()
{
    int min = this->target[0];
    int best_index = 0;
    for (int i = 1; i < this->target.size(); ++i) {
        if (this->target[i] < min){
            min = this->target[i];
            best_index = i;
        }
    }

    return best_index;
}

// Fonction pour réaliser des randoms entre deux borne
double OriginalBA::random(double min, double max)
{
    std::random_device rd;  // Utilisé pour obtenir une seed pour le générateur
    std::mt19937 gen(rd() + static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())); // Générateur Mersenne Twister

    std::uniform_real_distribution<double> distribution(min, max);
    return distribution(gen);
}

// Récupère le meilleur résultat
double OriginalBA::result()
{
    return this->init_target(this->g_best);
}