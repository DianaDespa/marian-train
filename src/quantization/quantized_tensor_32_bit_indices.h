#pragma once

#include "common/logging.h"
#include "quantization/quantized_tensor.h"

namespace marian {

class QuantizedTensor32BitIndices : public QuantizedTensorBase, public std::enable_shared_from_this<QuantizedTensor32BitIndices> {
// sorted in buckets
private:
  int* indices_;
  int* sizes_;
  size_t sizeIndices1_{655000};
  size_t sizeIndices2_{230000};
  int* indexOffsets_;

  void findSubtensorBucket(int idx, int* indexOffsets, int* sizes, int pos, int size);
  
  template<typename UnaryPredicate>
  thrust::device_ptr<int> fillIndicesFromIndex(int idx, Tensor t, UnaryPredicate predicate);

public:
  QuantizedTensor32BitIndices(size_t device, int* indices, float* centers, int bits = 2) {
    bits_ = bits;
    device_ = device;
    cudaSetDevice(device_);

    sizes_ = (int *)malloc((1 << bits_) * sizeof(int));
    indexOffsets_ = (int *)malloc((1 << bits_) * sizeof(int));
    memset(sizes_, 0, (1 << bits_) * sizeof(int));
    memset(indexOffsets_, 0, (1 << bits_) * sizeof(int));

    indices_ = indices;
    centers_ = centers;
  }

  QuantizedTensor32BitIndices(size_t device,
                              int bits = 2,
                              size_t sizeIndices1 = 655000,
                              size_t sizeIndices2 = 230000) {
    bits_ = bits;
    sizeIndices1_ = sizeIndices1;
    sizeIndices2_ = sizeIndices2;
    size_ = 2 * (sizeIndices1_ + sizeIndices2_);
    device_ = device;
    cudaSetDevice(device_);
    CUDA_CHECK(cudaMalloc(&centers_, sizeof(float) * (1 << (bits_ - 1))));
    CUDA_CHECK(cudaMalloc(&indices_, sizeof(int) * size_));
    cudaMemset(centers_, 0, sizeof(float) * (1 << (bits_ - 1)));
    cudaMemset(indices_, -1, sizeof(float) * size_);

    sizes_ = (int *)malloc((1 << bits_) * sizeof(int));
    indexOffsets_ = (int *)malloc((1 << bits_) * sizeof(int));
    memset(sizes_, 0, (1 << bits_) * sizeof(int));
    memset(indexOffsets_, 0, (1 << bits_) * sizeof(int));
  }

  ~QuantizedTensor32BitIndices() {}

  int* indices() { return indices_; }

  void setIndices(int* indices) { indices_ = indices; }

  int* sizes() { return sizes_; }

  void setSize(int size) { size_ = size; }

  int* indexOffsets() { return indexOffsets_; }

  QuantizedTensor subtensor(int pos, int size);

  void copyFrom(QuantizedTensor t);

  void scatterAdd(Tensor t, int offset);

  void encode(Tensor t);

  void decode(Tensor t, int offset);
};

}