#pragma once

#include "quantized_tensor.h"

namespace marian {

class QuantizedTensor32BitCompressed : public QuantizedTensorBase {
// 30 + 2 bits
private:
  unsigned int* data_;

public:
  QuantizedTensor32BitCompressed(size_t device, unsigned int* data, float* centers, int size, int bits = 2) {
    bits_ = bits;
    device_ = device;
    data_ = data;
    centers_ = centers;
    size_ = size;
  }

  QuantizedTensor32BitCompressed(int size, size_t device, int bits = 2) {
    device_ = device;
    size_ = size;
    bits_ = bits;
    cudaSetDevice(device_);
    CUDA_CHECK(cudaMalloc(&data_, sizeof(unsigned int*) * size));
    CUDA_CHECK(cudaMalloc(&centers_, sizeof(float) * (1 << (bits_ - 1))));
    cudaMemset(data_, 0, sizeof(unsigned int*) * (1 << (bits_ - 1)));
    cudaMemset(centers_, 0, sizeof(unsigned int*) * (1 << (bits_ - 1)));
  }

  ~QuantizedTensor32BitCompressed() {}

  unsigned int* data() { return data_; }

  QuantizedTensor subtensor(int pos, int size);

  void copyFrom(QuantizedTensor t);

  void scatterAdd(Tensor t, int offset);

  void encode(Tensor t);

  void decode(Tensor t, int offset);
};

}