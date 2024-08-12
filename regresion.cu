#include <cuda.h>
#include <regresion.cuh>

template<unsigned int N, unsigned int samples>
Regression<N, samples>::Regression (float* x[N*samples], float* y[samples]) : x(x), y(y) {};

template<unsigned int N, unsigned int samples>
void Regression<N, samples>::runtime (void) {
    firstKernel<<<samples, N>>>(void);
    secondKernel<<<samples, N>>>(void);
};

template<unsigned int N, unsigned int samples>
__device__ void Regression<N, samples>::prefetch_data (void) {
};

template<unsigned int N, unsigned int samples>
__device__ void Regression<N, samples>::grad_step (void) const {
};

template<unsigned int N, unsigned int samples>
__device__ void Regression<N, samples>::forward (void) const {
    // mult -> atomicSum -> fetch after prefetch of the next kernel -> grad_step

};

// Make the forward step
template<unsigned int N, unsigned int samples>
__global__ void Regression<N, samples>::firstKernel (void) {
};

// prefetch data and bring final data to device
template<unsigned int N, unsigned int samples>
__global__ void Regression<N, samples>::secondKernel (void) {
    prefetch_data(void);
    // put the semafor
    // fetch the atomic data
    // make the grad_step
};
