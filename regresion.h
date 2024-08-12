#ifndef REGRESION_H
#define REGRESION_H

template <int N>
class RegressionModel {
public:
    explicit RegressionModel (void);
    void fit (unsigned int&& epochs, float&& lr);
private:
    void init_weights (void);
    float compute_loss (float** base, float* y) const;
    float** base_dJ (float** x) const;
    float** dJ_dwn (unsigned int n, float** base, float* y, float** x) const;
    void grad_step (float alpha,float** base, float* y, float** x) const;
    float** dJ_db (float** base, float* y) const;
    float W_[N];
    float B_[N];
};

#endif //REGRESION_H
