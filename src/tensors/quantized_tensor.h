#pragma once

#include <memory>

#include "common/definitions.h"
#include "common/shape.h"
#include "tensors/memory_piece.h"

namespace marian {

class QuantizedTensorBase : public std::enable_shared_from_this<QuantizedTensorBase>,
    public TensorBase {
private:
  float* centers_;
  // int* indices_;

public:
  QuantizedTensorBase(Ptr<MemoryPiece> memory, Shape shape, size_t device, bits)
      : memory_(memory), shape_(shape) device_(device) {
    cudaSetDevice(device_);
    CUDA_CHECK(cudaMalloc(&data_, sizeof(float) * (1<<bits)));
  }

  ~QuantizedTensorBase() {}

  virtual float* centers() { return centers_; }
  // virtual int* indices() { return indices_; }
};

typedef std::shared_ptr<QuantizedTensorBase> QuantizedTensor;
}