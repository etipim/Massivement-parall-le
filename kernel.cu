#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>
#include <curand_kernel.h>

#include <cstdio>
#include <iostream>


#include "kernel.h"

// Calcul de la fonction objective
__device__ double init_target(double *bat, int fit_function, int dimension)
{
    double sum = 0;

    double sum1 = 0;
    double sum2 = 0;

    for (int i = 0; i < dimension; ++i) {
        // Sphère
        if (fit_function == 1) {
            //printf("bat %i : %f \n", i, bat[i]);
            sum += powf((bat[i]), 2); // (bat->solution[i] - 2) * (bat->solution[i] - 2);
        }

            // Rastrigin
        else if (fit_function == 2) {
            sum += bat[i] * bat[i] - 10 * cosf(2 * phi * bat[i]) + 10;
        }

            // Rosenbrock
        else if (fit_function == 3) {
            if (i < dimension - 1) {
                sum += 100 * pow((bat[i + 1] - powf(bat[i], 2)), 2) +
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

__global__ void init_curand(curandState *state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(clock() + i, i, 0, &state[i]);
}

// Réalisation du bat algo
__global__ void update_position(double* pop, double* new_pop, double* velocity, double* g_best, int pop_size, int dimension, int fit_function, curandState *state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= pop_size) return;

    //curandState state;
    //curand_init(clock64() + i, 0, 0, &state);

    // change pulse frequency
    double rand_frequence = curand_uniform(&state[i]);
    double pulse_frequence = pfMin + (pfMax + pfMin) * rand_frequence;

    for (int j = 0; j < dimension; j++){
        // change velocity
        double rand_velocity = curand_uniform(&state[i]);
        velocity[i * dimension + j] = rand_velocity * velocity[i * dimension + j] + (g_best[j] - pop[i * dimension + j]) * pulse_frequence; // Eq.3
        //printf("velo : %f \n", velocity[i * dimension + j]);
        // Update the solution based on the updated velocity
        double new_solution = pop[i * dimension + j] + velocity[i * dimension + j];

        // change solution and insert born
        if (new_solution > Ub){
            new_pop[i * dimension + j] = Ub;
        }
        else if (new_solution < Lb){
            new_pop[i * dimension + j] = Lb;
        }
        else {
            new_pop[i * dimension + j] = new_solution;  // Eq. 4
        }
    }
}

// Vérification des performances
__global__ void check_new_pos(double *pop, double *new_pop, double *pop_child, bool *pop_child_idx, double *targets, double *g_best, int fit_function, int dimension, int pop_size, curandState *state){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= pop_size) return;

    // Déclarer un tableau local pour éviter les conflits entre threads
    double* bat = new double[dimension];

    // découpage par bat
    for (int j = 0; j < dimension; ++j) {
        bat[j] = new_pop[i * dimension + j];
    }

    // Calcul de la performance de la chauve souris
    double new_target = init_target(bat, fit_function, dimension);
    //printf("new_target : %f \n", new_target);
    //printf("targets : %f \n", targets[i]);

    // Si la chauve souris est mieux positioné que avant on change ça position
    if (targets[i] > new_target){
        //printf("update pop %i \n", i);
        for(int j = 0; j < dimension; j++){
            pop[i * dimension + j] = new_pop[i * dimension + j];
            //printf("pop %i : %f \n", j, pop[i * dimension + j]);
        }

        targets[i] = new_target;
    } else {
        // éxection d'une pulsation aléatoire
        double rand_pulse = curand_uniform(&state[i]);
        if (rand_pulse > pulse_rate) {
            //printf("create pop child %i \n", i);
            for (int j = 0; j < dimension; j++){
                // add solution close to best
                double rand_best = Lb + (Ub - Lb) * curand_uniform(&state[i]);
                double new_solution = g_best[j] + 0.01 * rand_best;

                // change solution and insert born
                if (new_solution > Ub){
                    pop_child[i * dimension + j] = Ub;
                }
                else if (new_solution < Lb){
                    pop_child[i * dimension + j] = Lb;
                }
                else {
                    pop_child[i * dimension + j] = new_solution;
                    //printf("pop %i : %f \n", j, pop_child[i * dimension + j]);
                }
            }

            // Garde en mémoire les chauves souris avec une pulsation aléatoire
            pop_child_idx[i] = true;
        }
    }

    delete[] bat;
}

// Edit new_pop with pop_child
__global__ void update_new_pop(double* new_pop, double* pop_child, bool* pop_child_idx, int pop_size, int dimension, int fit_function){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= pop_size) return;

    if (pop_child_idx[i]){

        double* bat = new double[dimension];

        // découpage et clacul taget pour la bat de new_pop
        for (int j = 0; j < dimension; ++j) {
            bat[j] = new_pop[i * dimension + j];
        }
        double new_target = init_target(bat, fit_function, dimension);

        // découpage et clacul taget pour la bat de pop_child
        for (int j = 0; j < dimension; ++j) {
            bat[j] = pop_child[i * dimension + j];
        }
        double target_child = init_target(bat, fit_function, dimension);

        if (new_target > target_child){
            for (int j = 0; j < dimension; j++){
                new_pop[i * dimension + j] = pop_child[i * dimension + j];
            }
        }

        delete[] bat;
    }
}

/**
 * Runs on the GPU, called from the CPU or the GPU
*/
__global__ void Update_target_and_pop(double *positions, double *new_pop, double *target, int pop_size, int dimension, int fit_function){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= pop_size)
        return;

    double* bat = new double[dimension];

    // Mise à jour de positions par new_pop
    for (int j = 0; j < dimension; ++j) {
        positions[i * dimension + j] = new_pop[i * dimension + j];
        bat[j] = new_pop[i * dimension + j];
    }

    // mise à jour de la target
    target[i] = init_target(bat, fit_function, dimension);


    delete[] bat;

}

/**
 * Runs on the GPU, called from the CPU or the GPU
*/
__global__ void kernel_update_g_Best(double *positions, double *g_best, int pop_size, int dimension, int fit_function)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= pop_size)
        return;

    double* bat = new double[dimension];

    // récupère une chauves-souris
    for (int j = 0; j < dimension; ++j) {
        bat[j] = positions[i * dimension + j];
    }

    // si la chauves souris est meilleur que g_best
    if (init_target(bat, fit_function, dimension) < init_target(g_best, fit_function, dimension))
    {
        for (int k = 0; k < dimension; k++)
            g_best[k] = positions[i * dimension + k];
    }

    delete[] bat;
}

// cuda function
void cuda_pso(double* positions, double* velocities, double* targets, double* g_best, double best_target, int pop_size, int dimension, int fit_function)
{
    int size = pop_size * dimension;

    // declare all the arrays on the device
    double *devPos;
    double *devNewPos;
    double *devVel;
    double *devTar;
    double *devBst;
    double *devPosChl;
    bool *devPosChlIdx;
    curandState *devStates;

    // Memory allocation
    cudaMalloc((void**)&devPos, sizeof(double) * size);
    cudaMalloc((void**)&devNewPos, sizeof(double) * size);
    cudaMalloc((void**)&devVel, sizeof(double) * size);
    cudaMalloc((void**)&devTar, sizeof(double) * pop_size);
    cudaMalloc((void**)&devBst, sizeof(double) * dimension);
    cudaMalloc((void**)&devPosChl, sizeof(double) * size);
    cudaMalloc((void**)&devPosChlIdx, sizeof(bool) * pop_size);
    cudaMalloc((void **)&devStates, pop_size * sizeof(curandState));

    // Thread & Block number
    int threadsNum = 64;
    int blocksNum = (pop_size + threadsNum - 1) / threadsNum;

    // Initialisation des générateurs de nombres aléatoires
    init_curand<<<blocksNum, threadsNum>>>(devStates);

    cudaDeviceSynchronize();

    // Copy particle datas from host to device
    /**
     * Copy in GPU memory the data from the host
     * */
    cudaMemcpy(devPos, positions, sizeof(double) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devVel, velocities, sizeof(double) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devTar, targets, sizeof(double) * pop_size, cudaMemcpyHostToDevice);
    cudaMemcpy(devBst, g_best, sizeof(double) * dimension, cudaMemcpyHostToDevice);

    // boucle principale
    for (int iter = 0; iter < epoch; iter++){

        // init new pop
        double* new_pop = new double[size];

        cudaMemcpy(devNewPos, new_pop, sizeof(double) * size, cudaMemcpyHostToDevice);

        // Réalisation du bat algo
        update_position<<<blocksNum, threadsNum>>>(devPos, devNewPos, devVel, devBst, pop_size, dimension, fit_function, devStates);

        cudaDeviceSynchronize();

        cudaError_t e_err = cudaGetLastError();
        if (e_err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(e_err));
        }

        // init pop_child
        double* pop_child = new double[size];
        bool* pop_child_idx = new bool[pop_size];
        for (int i = 0; i < size; i++){
            pop_child[i] = 0;//positions[i];
        }
        for (int i = 0; i < pop_size; i++){
            pop_child_idx[i] = false;
        }

        cudaMemcpy(devPosChl, pop_child, sizeof(double) * size, cudaMemcpyHostToDevice);
        cudaMemcpy(devPosChlIdx, pop_child_idx, sizeof(bool) * pop_size, cudaMemcpyHostToDevice);

        // Vérification des performances
        check_new_pos<<<blocksNum, threadsNum>>>(devPos, devNewPos, devPosChl, devPosChlIdx, devTar, devBst, fit_function, dimension, pop_size, devStates);

        cudaDeviceSynchronize();

        cudaError_t a_err = cudaGetLastError();
        if (a_err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(a_err));
        }

        // Edit new_pop with pop_child
        update_new_pop<<<blocksNum, threadsNum>>>(devNewPos, devPosChl, devPosChlIdx, pop_size, dimension, fit_function);

        cudaDeviceSynchronize();

        cudaError_t b_err = cudaGetLastError();
        if (b_err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(b_err));
        }

        // Edit population
        Update_target_and_pop<<<blocksNum, threadsNum>>>(devPos, devNewPos, devTar, pop_size, dimension, fit_function);

        cudaDeviceSynchronize();

        cudaError_t c_err = cudaGetLastError();
        if (c_err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(c_err));
        }

        // Update g_best
        kernel_update_g_Best<<<blocksNum, threadsNum>>>(devPos, devBst, pop_size, dimension, fit_function);

        cudaDeviceSynchronize();

        cudaError_t d_err = cudaGetLastError();
        if (d_err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(d_err));
        }

        //cudaMemcpy(new_pop, devNewPos, sizeof(double) * size, cudaMemcpyDeviceToHost);

        //cudaMemcpy(pop_child, devPosChl, sizeof(double) * size, cudaMemcpyDeviceToHost);
        //cudaMemcpy(pop_child_idx, devPosChlIdx, sizeof(bool) * pop_size, cudaMemcpyDeviceToHost);

        //printf("best target : %f \n", host_init_target(g_best, fit_function, dimension));

        // Sup created var
        delete[] new_pop;
        delete[] pop_child;
        delete[] pop_child_idx;

        new_pop = nullptr;
        pop_child = nullptr;
        pop_child_idx = nullptr;
    }


    cudaMemcpy(positions, devPos, sizeof(double) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities, devVel, sizeof(double) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(targets, devTar, sizeof(double) * pop_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(g_best, devBst, sizeof(double) * dimension, cudaMemcpyDeviceToHost);





    // cleanup
    cudaFree(devPos);
    cudaFree(devNewPos);
    cudaFree(devVel);
    cudaFree(devTar);
    cudaFree(devBst);
    cudaFree(devPosChl);
    cudaFree(devPosChlIdx);

    printf("Best population :\n");
    for (int j = 0; j < dimension; j++) {
        printf("%f", g_best[j]);
        if (j < dimension - 1) printf(", ");
    }
    printf("]\n");

    printf("final target : %f \n", host_init_target(g_best, fit_function, dimension));
}