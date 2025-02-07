#include "kernel.h"
#include <random>

// Obtenir un random entre low et high
float getRandom(float low, float high) {
    return low + float(((high - low) + 1)*rand()/(RAND_MAX + 1.0));
}
// Obtenir un random entre 0.0f and 1.0f inclusif
float getRandomClamped() {
    return (float) rand()/(float) RAND_MAX;
}