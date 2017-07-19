#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <memory>

#include "kernels/cuda_helpers.h"
#include "kernels/tensor_operators.h"

namespace marian {

__global__ void gFindSubtensorQuantized(int* indices0,
																				int* indices1,
																				int size0,
																				int size1,
																				int targetStart,
																				int targetEnd,
																				int* resultStart0,
																				int* resultStart1,
																				int* resultEnd0,
																				int* resultEnd1) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size0 || idx >= size1)
    return;

  if(indices0[idx] >= targetStart
     && (idx == 0 || indices0[idx - 1] < targetStart)) {
    resultStart0[0] = idx;
  }

  if(indices0[idx] < targetEnd
     && (idx == size0 - 1 || indices0[idx + 1] >= targetEnd))
    resultEnd0[0] = idx;

	if(indices1[idx] >= targetStart
     && (idx == 0 || indices1[idx - 1] < targetStart)) {
    resultStart1[0] = idx;
  }

  if(indices1[idx] < targetEnd
     && (idx == size1 - 1 || indices1[idx + 1] >= targetEnd))
    resultEnd1[0] = idx;
}

class QuantizedTensorBase : public std::enable_shared_from_this<QuantizedTensorBase> {
	int* indices0_; // one big table
	int* indices1_;
	float center0_;
	float center1_;
	int encodingSize_;
	int size0_;
	int size1_;
	size_t device_;

public:
	QuantizedTensorBase(int encodingSize, int capacity, size_t device) {
		if (encodingSize != 2)
			return;

		device_ = device;
		encodingSize_ = encodingSize;
		cudaSetDevice(device_);

		cudaMalloc(&indices0_, sizeof(int) * capacity);
		cudaMalloc(&indices1_, sizeof(int) * capacity);
	}

	QuantizedTensorBase(int* indices0, int* indices1, int size0, int size1, size_t device) {
    indices0_ = indices0;
    indices1_ = indices1;
    size0_ = size0;
    size1_ = size1;
    device_ = device;
  }

	~QuantizedTensorBase() {}

	int encodingSize() { return encodingSize_; }

	int size0() { return size0_; }

	int size1() { return size1_; }

	int* indices0() { return indices0_; }

	int* indices1() { return indices1_; }

	void setSizes(int size0, int size1) { 
		size0_ = size0; 
		size1_ = size1; 
	}

	std::shared_ptr<QuantizedTensorBase> subtensor(int pos, int size, int idx) {
    cudaSetDevice(device_);
    cudaStreamSynchronize(0);
    int* start0;
		int* start1;
		int* end0;
		int* end1;

		cudaMalloc(&start0, sizeof(int));
		cudaMalloc(&start1, sizeof(int));
    cudaMalloc(&end0, sizeof(int));
    cudaMalloc(&end1, sizeof(int));

    int threads = 512;
    int blocks = 1 + size_ / threads;
    cudaMemset(start0, -1, sizeof(int));
    cudaMemset(start1, -1, sizeof(int));
    cudaMemset(end0, 0, sizeof(int));
    cudaMemset(end1, 0, sizeof(int));

    gFindSubtensorQuantized<<<blocks, threads>>>(
        indices0_, indices1_, size0_, size1_, pos, pos + size, start0, start1, end0, end1);

    int startOffset0;
		int startOffset1;
		int endOffset0;
		int endOffset1;
    // int tmp_dt; 
    cudaMemcpy(&startOffset0, start0, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&startOffset1, start1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&endOffset0, end0, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&endOffset1, end1, sizeof(int), cudaMemcpyDeviceToHost);

    // if(startOffset != -1 && startOffset < size_)
    //   cudaMemcpy(
    //       &tmp_dt, indices_ + startOffset, sizeof(int), cudaMemcpyDeviceToHost);

    int subtensorSize0 = max(0, endOffset0 - startOffset0 + 1);
    int subtensorSize1 = max(0, endOffset1 - startOffset1 + 1);
    cudaStreamSynchronize(0);
    return std::shared_ptr<QuantizedTensorBase>(new QuantizedTensorBase(
        indices0_ + startOffset0, indices1_ + startOffset1, subtensorSize0, subtensorSize1, device_));
  }

	void copyFrom(std::shared_ptr<QuantizedTensorBase> t, bool data_only = false) {
		// TODO
  }

	void toDense(Tensor t, int offset) {
		// TODO
  }

};

typedef std::shared_ptr<QuantizedTensorBase> QuantizedTensor;
}
