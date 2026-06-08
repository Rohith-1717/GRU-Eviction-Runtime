#include "runtime_gru.hpp"
#include "runtime_gru_params.hpp"

#include <cmath>

static inline float sigmoid(float x){
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float relu(float x){
    return x > 0.0f ? x : 0.0f;
}

static inline float dotProduct(const float* a, const float* b, size_t length){
    float result = 0.0f;

    for(size_t i = 0; i < length; ++i){
        result += a[i] * b[i];
    }

    return result;
}

static inline void matVec(const float* matrix, const float* vector, float* output, size_t rows, size_t cols){
    for(size_t r = 0; r < rows; ++r){
        output[r] = dotProduct(matrix + r * cols, vector, cols);
    }
}

RuntimeGRU::RuntimeGRU(){
    resetState();
}

void RuntimeGRU::resetState(){
    for(size_t i = 0; i < HIDDEN_SIZE; ++i){
        hidden_[i] = 0.0f;
    }
}

void RuntimeGRU::step(const float* input){
    float hiddenPrev[HIDDEN_SIZE];
    for(size_t i = 0; i < HIDDEN_SIZE; ++i){
        hiddenPrev[i] = hidden_[i];
    }

    float inputReset[HIDDEN_SIZE];
    float inputUpdate[HIDDEN_SIZE];
    float inputNew[HIDDEN_SIZE];
    float recurrentReset[HIDDEN_SIZE];
    float recurrentUpdate[HIDDEN_SIZE];
    matVec(GRU_WIR, input, inputReset, HIDDEN_SIZE, ACCESS_FEATURE_SIZE);
    matVec(GRU_WIZ, input, inputUpdate, HIDDEN_SIZE, ACCESS_FEATURE_SIZE);
    matVec(GRU_WIN, input, inputNew, HIDDEN_SIZE, ACCESS_FEATURE_SIZE);
    matVec(GRU_WHR, hiddenPrev, recurrentReset, HIDDEN_SIZE, HIDDEN_SIZE);
    matVec(GRU_WHZ, hiddenPrev, recurrentUpdate, HIDDEN_SIZE, HIDDEN_SIZE);

    float resetGate[HIDDEN_SIZE];
    float updateGate[HIDDEN_SIZE];
    float resetState[HIDDEN_SIZE];

    for(size_t i = 0; i < HIDDEN_SIZE; ++i){
        resetGate[i] = sigmoid(
            inputReset[i]
            + recurrentReset[i]
            + GRU_BIR[i]
            + GRU_BIR_HIDDEN[i]
        );

        updateGate[i] = sigmoid(
            inputUpdate[i]
            + recurrentUpdate[i]
            + GRU_BIZ[i]
            + GRU_BIZ_HIDDEN[i]
        );

        resetState[i] = resetGate[i] * hiddenPrev[i];
    }

    float recurrentNew[HIDDEN_SIZE];

    matVec(GRU_WHN, resetState, recurrentNew, HIDDEN_SIZE, HIDDEN_SIZE);

    for(size_t i = 0; i < HIDDEN_SIZE; ++i){
        float candidateInput =
            inputNew[i]
            + recurrentNew[i]
            + GRU_BIN[i]
            + GRU_BHN[i];

        float candidate = std::tanh(candidateInput);

        hidden_[i] =
            (1.0f - updateGate[i]) * candidate
            + updateGate[i] * hiddenPrev[i];
    }
}

float RuntimeGRU::predictReuseProbability(const float* pageFeatures) const{
    float scorerInput[HIDDEN_SIZE + PAGE_FEATURE_SIZE];

    for(size_t i = 0; i < HIDDEN_SIZE; ++i){
        scorerInput[i] = hidden_[i];
    }

    for(size_t i = 0; i < PAGE_FEATURE_SIZE; ++i){
        scorerInput[HIDDEN_SIZE + i] = pageFeatures[i];
    }

    float fc1[FC1_SIZE];

    matVec(
        FC1_W,
        scorerInput, fc1,
        FC1_SIZE, HIDDEN_SIZE + PAGE_FEATURE_SIZE
    );

    for(size_t i = 0; i < FC1_SIZE; ++i){
        fc1[i] = relu(fc1[i] + FC1_B[i]);
    }

    float fc2[FC2_SIZE];

    matVec(
        FC2_W,fc1,fc2,FC2_SIZE,
        FC1_SIZE
    );

    for(size_t i = 0; i < FC2_SIZE; ++i){
        fc2[i] = relu(fc2[i] + FC2_B[i]);
    }

    float logit =
        dotProduct(OUT_W, fc2, FC2_SIZE)
        + OUT_B[0];

    return sigmoid(logit);
}
