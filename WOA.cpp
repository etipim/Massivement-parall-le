#include "WOA.h"


OriginalWOA::OriginalWOA(int epoch1 = 10000, int popSize1 = 100)
{

}

void OriginalWOA::trierPopulationEnFonctionDeFitness(std::vector<double>& fitn, std::vector<std::vector<double>>& population) {

}

//corriger la posisition si hors des limites
double OriginalWOA::amendPosition(double pos, double lb, double ub) {

}

//algo du  WOA
std::vector<double> OriginalWOA::solve(std::vector<double> &lb,std::vector<double> &ub, std::function<double(std::vector<double>)> func){



}
