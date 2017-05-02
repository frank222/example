//
//  main.cpp
//  Example
//
//  Created by Shiqian Mao on 16/10/22.
//  Copyright © 2016年 shiqianmao. All rights reserved.
//

#include <cmath>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

typedef std::vector<std::vector<float>> Matrix;

struct Model {
    Matrix w;
    std::vector<float> b;
};

void MatMulAddB(const Matrix& data, const Matrix& w, const std::vector<float>& b, Matrix* pred) {
    unsigned long num_samples = data.size();
    unsigned long num_features = data[0].size();
    unsigned long num_class = w.size();
    *pred = Matrix(num_samples, std::vector<float>(num_class, 0));
    for (int n = 0; n < num_samples; n++) {
        for (int j = 0; j < num_features; j++) {
            for (int k = 0; k < num_class; k++) {
                (*pred)[n][k] += data[n][j] * w[k][j] + b[k];
            }
        }
    }
}

void Transform(Matrix* pred, const std::string& transform) {
    unsigned long num_samples = pred->size();
    unsigned long num_class = (*pred)[0].size();
    if (transform == "log") {
        for (int n = 0; n < num_samples; n++) {
            for (int k = 0; k < num_class; k++) {
                (*pred)[n][k] = log((*pred)[n][k]);
            }
        }
    }
}

void Softmax(Matrix* pred) {
    Matrix expp = *pred;
    for (int i = 0; i < pred->size(); i++) {
        float sum = 0;
        for (int c = 0; c < (*pred)[i].size(); c++) {
            sum += exp((*pred)[i][c]);
            (*pred)[i][c] = exp((*pred)[i][c]);
        }
        for (int c = 0; c < (*pred)[i].size(); c++) {
            (*pred)[i][c] /= sum;
        }
    }
}

void Feedforward(const Matrix& data, Model* model, Matrix* pred) {
    MatMulAddB(data, model->w, model->b, pred);
    Transform(pred, "log");
    //Softmax(pred);
}

void MatAdd(Matrix* a, const Matrix& b) {
    for (int i = 0; i < a->size(); i++) {
        for (int j = 0; j < (*a)[i].size(); j++) {
            (*a)[i][j] += b[i][j];
        }
    }
}

void VecAdd(std::vector<float>* a, std::vector<float>& b) {
    for (int i = 0; i < a->size(); i++) {
        (*a)[i] += b[i];
    }
}

std::vector<float> MeanCol(const Matrix& m) {
    unsigned long num_samples = m.size();
    unsigned long num_class = m[0].size();
    std::vector<float> mean(num_class, 0);
    for (int n = 0; n < num_samples; n++) {
        for (int c = 0; c < num_class; c++) {
            mean[c] += m[n][c] / num_samples;
        }
    }
    return mean;
}

float Mean(const Matrix& M) {
    unsigned long n = M.size();
    unsigned long m = M[0].size();
    float mean = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            mean += M[i][j];
        }
    }
    return mean;
}


void Backprop(const Matrix& pred, const Matrix& x, const Matrix& y, Model* model) {
    unsigned long num_samples = pred.size();
    unsigned long num_class = pred[0].size();
    unsigned long num_features = x[0].size();
    Matrix delta = y;
    Matrix delta_raw = y;
    Matrix loss = y;
    for (int n = 0; n < num_samples; n++) {
        for (int c = 0; c < num_class; c++) {
            loss[n][c] = -log(pred[n][c])*y[n][c] - log(1-pred[n][c])*(1-y[n][c]);
            delta_raw[n][c] = pred[n][c] - y[n][c];
            delta[n][c] = delta_raw[n][c] * (-(1 - pred[n][c]) * pred[n][c]);
        }
    }
    std::cout << "loss:" << Mean(loss) << std::endl;
    std::cout << "loss_derivative:" << Mean(delta_raw) << std::endl;
    
    float lr = .1;
    unsigned long num_out = model->w.size();
    unsigned long num_in = model->w[0].size();
    Matrix grad(num_out, std::vector<float>(num_in, 0));
    std::vector<float> grad_b(num_class, 0);
    for (int n = 0; n < num_samples; n++) {
        for (int c = 0; c < num_class; c++) {
            for (int j = 0; j < num_features; j++) {
                grad[c][j] += - delta[n][c] * x[n][j] / num_samples * lr;
            }
            grad_b[c] += -delta_raw[n][c] / num_samples * lr;
        }
    }
    MatAdd(&(model->w), grad);
    VecAdd(&(model->b), grad_b);
}

void Train(const Matrix& x, const Matrix& y, Model* model) {
    Matrix pred;
    int num_iters = 100;
    for (int iter = 0; iter < num_iters; iter++) {
        std::cout << "==========iter:" << iter << std::endl;
        Feedforward(x, model, &pred);
        Backprop(pred, x, y, model);
    }
}

int main(int argc, const char * argv[]) {
    Matrix x = {{1, 2, 3}};
    Matrix y = {{0, 1}};
    Model model;
    model.w = {{.1, .1, .1}, {.1, .1, .1}};
    model.b = {0, 0};
    Train(x, y, &model);
    return 0;
}
