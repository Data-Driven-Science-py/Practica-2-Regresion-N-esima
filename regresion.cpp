#include <cstdio>
#include <cstdlib>
#include <regresion.h>
#include <iostream>

template<int N>
RegressionModel<N>::RegressionModel (void) {
        init_weights();
};

template<int N>
void RegressionModel<N>::init_weights (void) {
    for (int i=0; i < N; ++i) {
        W_[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    B_ = static_cast<float>(rand()) / RAND_MAX;
};

template<int N>
void RegressionModel<N>::grad_step (float&& alpha, float**base, float* y, float** x) {
    for (int j = 0; j < N; ++j) {
        W_[j] -= alpha*dJ_dwn(j, base, y, x);
    }
    B_ -= alpha* dJ_db(base, y);
};

template<int N>
float RegressionModel<N>::dJ_dwn (unsigned int n, float** base, float* y, float** x) const {
    float out = 0;
    float* out_array = new float[length(base)]{0};

    #pragma unroll
    for (int i = 0; i < length(base); ++i) {
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            out_array[i] += base[i][j];
        }
        out_array[i] += B_ - y[i];
        out_array[i] *= x[i][n];
    }

    for (auto value: out_array) {
        out += value;
    }

    return 0.5 * out/length(out_array);
};

template<int N>
float** RegressionModel<N>::dJ_db (float** base, float* y) const {
    float* out_array = new float[length(base)]{0};
    float out = 0;
    #pragma unroll
    for (int i = 0; i < length(base); ++i) {
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            out_array[i] += base[i][j];
        }
        out_array[i] += B_ - y[i];
    }

    #pragma unroll
    for (auto value: out_array) {
        out += value;
    }
    delete[] out_array;

    return 0.5 * out/length(out_array);
}

template<int N>
float** RegressionModel<N>::base_dJ (float x[][N]) const {
    // x*w + b
    float** out = new float*[length(x)];
    #pragma unroll
    for (int j = 0; j < length(x); ++j) {
        out[j] = new float[N];
        #pragma unroll
        for (int i = 0; i < length(x[j]); ++i) {
            out[j][i] = x[j][i] * W_[i];
        }
    }
    return out;
};

template<int N>
float RegressionModel<N>::compute_loss (float** base, float* y) const {
    float out_array[length(base)] = {0};
    float out = 0;
    #pragma unroll
    for (int i = 0; i < length(base); ++i) {
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            out_array[i] += base[i][j];
        }
        out_array[i] += B_ - y[i];
    }

    for (int k=0; k< length(out_array); ++k) {
        out += std::pow(out_array[k], 2);
    }
    delete [] out_array;

    return out/length(out_array);
}

template <int N>
void RegressionModel<N>::fit (unsigned int&& epochs, float&& lr, float** x, float* y) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float** base = base_dJ(x);
        grad_step(lr, base, y, x);
        float loss = compute_loss(base, y);

        std::cout << "Epoch:" << epoch << ", Loss:" << loss << std::endl;

        for (int i; i < length(x) ; ++i) {
            delete[] base[i];
        }
        delete[] base;
    }
};

