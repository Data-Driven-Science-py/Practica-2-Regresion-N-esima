#ifndef REGRESION_CUH
#define REGRESION_CUH

template <unsigned int N, unsigned int samples>
class Regression {
public:
    explicit Regression (float* x[N*samples], float* y[samples]);
private:
    const float* x[N*samples];
    const float* y[samples];
    mutable float* cum_grad[N + 1];
    mutable float* weight[N];
    mutable float* bias;
    __device__ void prefetch_cum_grad ();
    __device__ void grad_step ();
    __device__ void dJ_dwn () const;
    __device__ void dJ_db () const;
}
#endif // REGRESION_CUH

