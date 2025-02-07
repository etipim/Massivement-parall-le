#include <cuda_runtime.h>
#include <cuda.h>
#include <math_functions.h>


#include "kernel.h"

// Calcul de la fonction objective
__device__ float init_target(float *bat, int fit_function, int dimension)
{
    float sum = 0.0;
    float sum1 = 0.0;
    float sum2 = 0.0;
    for (int i = 0; i < dimension; ++i) {
        // Sphère
        if (fit_function == 1) {
            sum += pow((bat[i]), 2); // (bat->solution[i] - 2) * (bat->solution[i] - 2);
        }

        // Rastrigin
        else if (fit_function == 2) {
            sum += bat[i] * bat[i] - 10.0 * cos(2.0 * phi * bat[i]) + 10.0;
        }

        // Rosenbrock
        else if (fit_function == 3) {
            if (i < dimension - 1) {
                sum += 100.0 * pow((bat[i + 1] - pow(bat[i], 2)), 2) +
                       pow((1.0 - bat[i]), 2);
            }
        }
    }

    return sum;
}

// générer la population
__global__ void generate_population(float *pop, float *velocity, float *target, float best_target, int popSize, int dimension, int Ub, int Lb, int fit_function)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;


    // Génère aléatoirement la population de chauves souris
    curandState state;
    curand_init(clock64() + idx, 0, 0, &state);
    double rand_val = curand_uniform(&state);

    pop[idx] = Lb + (Ub - Lb) * rand_val;

    // Calcul la position des chauves souris
    target[idx] = init_target(pop[idx], fit_function, dimension);

    // initialisation la vélocité des chauves souris à 0
    velocity[idx] = 0;

    //__syncthreads();

    // Initialisation de la meilleure cible
    if (idx == 0) {
        best_target = target[0];
    }
}

//
extern "C" void cuda_pso(int epoch, int pop_size, int dimension, )
{
    int size = pop_size * dimension;

    // declare all the arrays on the device
    float *devPos;
    float *devVel;
    float *devTar;

    // Memory allocation
    cudaMalloc((void**)&devPos, sizeof(float) * size);
    cudaMalloc((void**)&devVel, sizeof(float) * size);
    cudaMalloc((void**)&devTar, sizeof(float) * size);

    // Thread & Block number
    int threadsNum = 32;
    int blocksNum = ceil(size / threadsNum);

    // Copy particle datas from host to device
    /**
     * Copy in GPU memory the data from the host
     * */
    cudaMemcpy(devPos, positions, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devVel, velocities, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(devTar, targets, sizeof(float) * size, cudaMemcpyHostToDevice);

    for (){
        generate_population<<<blocksNum, threadsNum>>>()
    }
}