#include <cuda.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <memory>

#include "kernels/cuda_helpers.h"
#include "tensors/tensor.h"
#include "quantization/quantized_tensor_32_bit_compressed.h"

namespace marian {

__global__ void gFindSubtensor32BitCompressed(unsigned int* data,
                                              int size,
                                              int targetStart,
                                              int targetEnd,
                                              int* resultStart,
                                              int* resultEnd) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;

  unsigned int mask = (unsigned int)~0 >> 2;

  if((data[idx] & mask) >= targetStart
     && (idx == 0 || (data[idx - 1] & mask) < targetStart)) {
    resultStart[0] = idx;
  }

  if((data[idx] & mask) < targetEnd
     && (idx == size - 1 || (data[idx + 1] & mask) >= targetEnd))
    resultEnd[0] = idx;
}

__global__ void gScatterAdd32BitCompressed(float* denseData,
                                           float* sparseCenters,
                                           unsigned int* sparseEncoding,
                                           int denseSize,
                                           int sparseSize,
                                           int offset) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= sparseSize)
    return;

  unsigned int mask = (unsigned int)~0 >> 2;
  int index = sparseEncoding[idx] & mask;
  unsigned int dataEncoding = sparseEncoding[idx] >> 30;
  float data = ((dataEncoding / 2 > 0) ? -1.0 : 1.0) * sparseCenters[dataEncoding % 2];

  if(index + offset >= 0
     && index + offset < denseSize)
    denseData[index + offset] += data;
}

__global__ void gAddEncoding(float* denseData,
                            float* centers,
                            unsigned int* destinationData,
                            int sparseSize) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= sparseSize)
    return;

  // Get encoding
  float absData = std::abs(denseData[destinationData[idx]]);
  unsigned int encoding;

  if (absData < centers[1]) {
    if (denseData[destinationData[idx]] >= 0)
      encoding = 0;
    else
      encoding = (unsigned int)2 << 30;
  } else {
    if (denseData[destinationData[idx]] >= 0)
      encoding = (unsigned int)1 << 30;
    else
      encoding = (unsigned int)3 << 30;
  }

  // Append to data
  destinationData[idx] = destinationData[idx] | encoding;

}

struct is_bigger_abs_than
{
  float val_;

  is_bigger_abs_than(float val) : val_(val) {}

  __host__ __device__
  bool operator()(const float x)
  {
    return std::abs(x) > val_;
  }
};

QuantizedTensor QuantizedTensor32BitCompressed::subtensor(int pos, int size) {
  cudaSetDevice(device_);
  cudaStreamSynchronize(0);
  
  int* start;
  int* end;
  CUDA_CHECK(cudaMalloc(&start, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&end, sizeof(int)));
  cudaMemset(start, -1, sizeof(int));
  cudaMemset(end, 0, sizeof(int));

  int blocks = 1 + size_ / THREAD_COUNT;
  gFindSubtensor32BitCompressed<<<blocks, THREAD_COUNT>>>(
        data_, size_, pos, pos + size, start, end);

  int startOffset;
  int endOffset;
  CUDA_CHECK(cudaMemcpy(&startOffset, start, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&endOffset, end, sizeof(int), cudaMemcpyDeviceToHost));
  int subtensorSize = max(0, endOffset - startOffset + 1);

  cudaFree(start);
  cudaFree(end);

  cudaStreamSynchronize(0);
  return QuantizedTensor(new QuantizedTensor32BitCompressed(
      device_, data_ + startOffset, centers_, subtensorSize));
}

void QuantizedTensor32BitCompressed::copyFrom(QuantizedTensor t) {
  size_t size = t->size();

  if(size_ < size) {
    return;
  }

  size_ = size;
  if(size == 0)
    return;
  cudaSetDevice(device_);
  cudaStreamSynchronize(0);

  float* centers = t->centers();
  unsigned int* data = std::dynamic_pointer_cast<QuantizedTensor32BitCompressed>(t)->data();
  CUDA_CHECK(cudaMemcpy(data_, data, size * sizeof(float), cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(centers_, centers, (1 << (bits_ - 1)) * sizeof(float), cudaMemcpyDefault));
  cudaStreamSynchronize(0);
}

void QuantizedTensor32BitCompressed::scatterAdd(Tensor t, int offset) {
  cudaSetDevice(device_);
  cudaStreamSynchronize(0);
  int threads = 512;
  int blocks = 1 + size_ / threads;
  gScatterAdd32BitCompressed<<<blocks, threads>>>(
      t->data(), centers_, data_, t->size(), size_, offset);
  cudaStreamSynchronize(0);
}

void QuantizedTensor32BitCompressed::encode(Tensor t) {
  computeCenters(t);

  thrust::device_ptr<float> sourseData(t->data());
  thrust::device_ptr<unsigned int> destinationData(data_);

  try {
    thrust::counting_iterator<unsigned int> it1(0);
    thrust::counting_iterator<unsigned int> it2 = it1 + t->size();
    auto it = thrust::copy_if(
        it1,
        it2,
        sourseData,
        destinationData,
        is_bigger_abs_than(get(centers_, 0)));
    
    size_ = it - &destinationData[0];
  } catch(thrust::system_error e) {
    std::cerr << "Error inside copy_if: " << e.what() << std::endl;
  }

  int blocks = 1 + size_ / THREAD_COUNT;
  gAddEncoding<<<blocks, THREAD_COUNT>>>(t->data(), centers_, data_, size_);

  cudaStreamSynchronize(0);
}

void QuantizedTensor32BitCompressed::decode(Tensor t, int offset) {
  cudaSetDevice(device_);
  int blocks = 1 + size_ / THREAD_COUNT;
  t->set(0);
  gScatterAdd32BitCompressed<<<blocks, THREAD_COUNT>>>(
      t->data(), centers_, data_, t->size(), size_, offset);
  cudaStreamSynchronize(0);
}

}