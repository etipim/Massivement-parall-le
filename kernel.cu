#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>
#include <curand_kernel.h>

#include <cstdio>
#include <iostream>


#include "kernel.h"

// Calcul de la fonction objective
__device__ float init_target(float *bat, int fit_function, int dimension)
{
    float sum = 0;

    for (int i = 0; i < dimension; ++i) {
        // Sphère
        if (fit_function == 1) {
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
    }

    return sum;
}

// Réalisation du bat algo
__global__ void update_position(float* pop, float* new_pop, float* velocity, float* g_best, int popSize, int dimension, int fit_function)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(clock64() + i, 0, 0, &state);

    // change pulse frequency
    float rand_frequence = curand_uniform(&state);
    float pulse_frequence = pfMin + (pfMax + pfMin) * rand_frequence;

    for (int j = 0; j < dimension; j++){
        // change velocity
        float rand_velocity = curand_uniform(&state);
        velocity[i * dimension + j] = rand_velocity * velocity[i * dimension + j] + (g_best[j] - pop[i * dimension + j]) * pulse_frequence; // Eq.3

        // Update the solution based on the updated velocity
        float new_solution = pop[i * dimension + j] + velocity[i * dimension + j];

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

//    // Génère aléatoirement la population de chauves souris
//    curandState state;
//    curand_init(clock64() + idx, 0, 0, &state);
//    double rand_val = curand_uniform(&state);
//
//    pop[idx] = Lb + (Ub - Lb) * rand_val;
//
//    // Calcul la position des chauves souris
//    target[idx] = init_target(pop, fit_function, dimension);
//
//    // initialisation la vélocité des chauves souris à 0
//    velocity[idx] = 0;
//
//    //__syncthreads();
//
//    // Initialisation de la meilleure cible
//    if (idx == 0) {
//        best_target = target[0];
//    }

    //__syncthreads();  // Synchroniser avant l'affichage final

    // Debug: Afficher les valeurs après mise à jour
    //printf("Thread %d: Après maj | new_pop[0] = %f, velocity[0] = %f\n", i, new_pop[i * dimension], velocity[i]);
}

// Vérification des performances
__global__ void check_new_pos(float *pop, float *new_pop, float *pop_child, bool *pop_child_idx, float *targets, float *g_best, float *bat, int fit_function, int dimension){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(clock64() + i, 0, 0, &state);

    // découpage par bat
    for (int j = 0; j < dimension; ++j) {
        bat[j] = new_pop[i * dimension + j];
    }

    // Calcul de la performance de la chauve souris
    float new_target = init_target(bat, fit_function, dimension);

    // Si la chauve souris est mieux positioné que avant on change ça position
    if (targets[i] > new_target){
        for(int j = 0; j < dimension; j++){
            pop[i * dimension + j] = new_pop[i * dimension + j];
        }

        targets[i] = new_target;
    } else {
        // éxection d'une pulsation aléatoire
        float rand_pulse = curand_uniform(&state);
        if (rand_pulse > pulse_rate) {
            for (int j = 0; j < dimension; j++){
                // add solution close to best
                float rand_best = curand_uniform(&state);
                float new_solution = g_best[j] + 0.01 * rand_best;

                // change solution and insert born
                if (new_solution > Ub){
                    pop_child[i * dimension + j] = Ub;
                }
                else if (new_solution < Lb){
                    pop_child[i * dimension + j] = Lb;
                }
                else {
                    pop_child[i * dimension + j] = new_solution;
                }
            }

            // Garde en mémoire les chauves souris avec une pulsation aléatoire
            pop_child_idx[i] = true;
        }
    }
}

//
extern "C" void cuda_pso(float* positions, float* velocities, float* targets, float* g_best, float best_target, int pop_size, int dimension, int fit_function)
{
    int size = pop_size * dimension;

    // declare all the arrays on the device
    float *devPos;
    float *devNewPos;
    float *devVel;
    float *devTar;
    float *devBst;
    float *devPosChl;
    float *devBat;
    bool *devPosChlIdx;

    // Memory allocation
    cudaMalloc((void**)&devPos, sizeof(float) * size);
    cudaMalloc((void**)&devNewPos, sizeof(float) * size);
    cudaMalloc((void**)&devVel, sizeof(float) * pop_size);
    cudaMalloc((void**)&devTar, sizeof(float) * pop_size);
    cudaMalloc((void**)&devBst, sizeof(float) * dimension);
    cudaMalloc((void**)&devPosChl, sizeof(float) * size);
    cudaMalloc((void**)&devBat, sizeof(float) * dimension);
    cudaMalloc((void**)&devPosChlIdx, sizeof(bool) * pop_size);

    // Thread & Block number
    int threadsNum = 256;
    //int blocksNum = ceil(size / threadsNum);
    int blocksNum = (pop_size + threadsNum - 1) / threadsNum;

    // Copy particle datas from host to device
    /**
     * Copy in GPU memory the data from the host
     * */
    cudaMemcpy(devPos, positions, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devVel, velocities, sizeof(float) * pop_size, cudaMemcpyHostToDevice);
    cudaMemcpy(devTar, targets, sizeof(float) * pop_size, cudaMemcpyHostToDevice);
    cudaMemcpy(devBst, g_best, sizeof(float) * dimension, cudaMemcpyHostToDevice);

    for (int iter = 0; iter < 1; iter++){//epoch; iter++){


        // init new pop
        float* new_pop = new float[size];
        //for (int i = 0; i < size; i++) {
        //    new_pop[i] = positions[i];
        //}
        cudaMemcpy(devNewPos, new_pop, sizeof(float) * size, cudaMemcpyHostToDevice);

        // Réalisation du bat algo
        update_position<<<blocksNum, threadsNum>>>(devPos, devNewPos, devVel, devBst, pop_size, dimension, fit_function);

        cudaDeviceSynchronize();

        cudaError_t a_err = cudaGetLastError();
        if (a_err != cudaSuccess) {
            printf("Erreur CUDA après update_position: %s\n", cudaGetErrorString(a_err));
            fflush(stdout);
        }

        cudaMemcpy(new_pop, devNewPos, sizeof(float) * size, cudaMemcpyDeviceToHost);
        cudaMemcpy(g_best, devBst, sizeof(float) * dimension, cudaMemcpyDeviceToHost);

        // Vérification
        printf("Initial g_best values:\n");
        for (int j = 0; j < dimension; ++j) {
            printf("g_best[%d] = %f\n", j, g_best[j]);
        }

        // init pop_child
        float* pop_child = new float[size];
        float* bat = new float[dimension];
        bool* pop_child_idx = new bool[pop_size];
        for (int i = 0; i < size; i++){
            pop_child[i] = 0;//positions[i];
        }
        for (int i = 0; i < pop_size; i++){
            pop_child_idx[i] = false;
        }

        cudaMemcpy(devPosChl, pop_child, sizeof(float) * size, cudaMemcpyHostToDevice);
        cudaMemcpy(devPosChlIdx, pop_child_idx, sizeof(bool) * pop_size, cudaMemcpyHostToDevice);
        cudaMemcpy(devBat, bat, sizeof(float) * dimension, cudaMemcpyHostToDevice);

        // Vérification des performances
        check_new_pos<<<blocksNum, threadsNum>>>(devPos, devNewPos, devPosChl, devPosChlIdx, devTar, devBst, devBat, fit_function, dimension);

        cudaDeviceSynchronize();

        cudaError_t b_err = cudaGetLastError();
        if (b_err != cudaSuccess) {
            printf("Erreur CUDA après update_position: %s\n", cudaGetErrorString(b_err));
            fflush(stdout);
        }

        // Edit new_pop with pop_child
        cudaMemcpy(new_pop, devNewPos, sizeof(float) * size, cudaMemcpyDeviceToHost);
        cudaMemcpy(pop_child, devPosChl, sizeof(float) * size, cudaMemcpyDeviceToHost);
        cudaMemcpy(pop_child_idx, devPosChlIdx, sizeof(bool) * pop_size, cudaMemcpyDeviceToHost);

//        printf("Random : %f \n", getRandomClamped());
//        printf("Nouvelle population époque %d :\n", iter + 1);
//        for (int i = 0; i < 1; i++) {
//            printf("Chauve-souris %d : [", i);
//            for (int j = 0; j < dimension; j++) {
//                printf("%f", new_pop[i * dimension + j]);
//                if (j < dimension - 1) printf(", ");
//            }
//            printf("]\n");
//        }

        for (int i = 0; i < pop_size; i++) {
            printf("Chauve-souris %d (actif : %s) : [", i, pop_child_idx[i] ? "true" : "false");
        }

        // Edit population


        // Calcul final target


        // Find best bat


        // Sup created var
        delete[] new_pop;
        delete[] pop_child;
        delete[] pop_child_idx;

        new_pop = nullptr;
        pop_child = nullptr;
        pop_child_idx = nullptr;
    }

    cudaMemcpy(positions, devPos, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities, devVel, sizeof(float) * pop_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(targets, devTar, sizeof(float) * pop_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(g_best, devBst, sizeof(float) * dimension, cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(devPos);
    cudaFree(devNewPos);
    cudaFree(devVel);
    cudaFree(devTar);
    cudaFree(devBst);
    cudaFree(devPosChl);
    cudaFree(devPosChlIdx);

    printf("Population époque :\n");
    for (int i = 0; i < 1; i++) {
        printf("Chauve-souris %d : [", i);
        for (int j = 0; j < dimension; j++) {
            printf("%f", positions[i * dimension + j]);
            if (j < dimension - 1) printf(", ");
        }
        printf("]\n");
    }
}