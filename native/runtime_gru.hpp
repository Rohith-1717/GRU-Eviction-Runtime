#pragma once

#include <cstddef>

class RuntimeGRU{
public:
    static constexpr size_t ACCESS_FEATURE_SIZE = 5;
    static constexpr size_t PAGE_FEATURE_SIZE = 4;
    static constexpr size_t HIDDEN_SIZE = 64;
    static constexpr size_t FC1_SIZE = 64;
    static constexpr size_t FC2_SIZE = 32;
    RuntimeGRU();
    void resetState();
    void step(const float* accessFeatures);
    float predictReuseProbability(const float* pageFeatures) const;

private:

    float hidden_[HIDDEN_SIZE];
};
