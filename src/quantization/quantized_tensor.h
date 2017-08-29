#pragma once

#include "tensors/tensor.h"

namespace marian {

class QuantizedTensorBase : public std::enable_shared_from_this<QuantizedTensorBase> {
protected:
  float* centers_;
  size_t device_;
  double dropRate_{0.99};
  int bits_{2};
  int size_;

  static const int THREAD_COUNT{512};

  // A helper, returns i-th element from a GPU stored array.
  template <class T>
  T get(T* data, int i) {
    T res;
    cudaMemcpy(&res, data + i, sizeof(T), cudaMemcpyDeviceToHost);
    return res;
  }

  template <class T>
  void set(T* data, int i, T val) {
    cudaMemcpy(data + i, &val, sizeof(T), cudaMemcpyHostToDevice);
  }

public:
  QuantizedTensorBase() {}

  ~QuantizedTensorBase() {}

  float* centers() { return centers_; }

  size_t getDevice() { return device_; }

  size_t size() { return size_; }

  void setSize(size_t size) { size_ = size; }
 
  virtual std::shared_ptr<QuantizedTensorBase> subtensor(int pos, int size) {
    return std::shared_ptr<QuantizedTensorBase>(new QuantizedTensorBase());
  }

  virtual void copyFrom(std::shared_ptr<QuantizedTensorBase> t) {}

  virtual void scatterAdd(Tensor t, int offset = 0) {}

  virtual void encode(Tensor t) {}

  virtual void decode(Tensor t, int offset) {}

  void computeCenters(Tensor t);

};

typedef std::shared_ptr<QuantizedTensorBase> QuantizedTensor;
}