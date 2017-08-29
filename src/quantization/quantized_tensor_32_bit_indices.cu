#include <cuda.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <memory>

#include "kernels/cuda_helpers.h"
#include "tensors/tensor.h"
#include "quantization/quantized_tensor_32_bit_indices.h"

namespace marian {

__global__ void gFindSubtensor32BitIndices(int* indices,
                                           int size,
                                           int targetStart,
                                           int targetEnd,
                                           int* resultStart,
                                           int* resultEnd) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;

  if(indices[idx] >= targetStart
     && (idx == 0 || indices[idx - 1] < targetStart)) {
    resultStart[0] = idx;
  }

  if(indices[idx] < targetEnd
     && (idx == size - 1 || indices[idx + 1] >= targetEnd))
    resultEnd[0] = idx;
}

__global__ void gScatterAdd32BitIndices(float* denseData,
                                        float* centers,
                                        int* indices,
                                        int denseSize,
                                        int* sparseSizes,
                                        int* indexOffsets,
                                        int offset) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= indexOffsets[3] + sparseSizes[3])
    return;
  if(indices[idx] + offset >= 0
     && indices[idx] + offset < denseSize) {
    if (idx >= indexOffsets[0] && idx < indexOffsets[0] + sparseSizes[0])
      denseData[indices[idx] + offset] += centers[0];
    else if (idx >= indexOffsets[1] && idx < indexOffsets[1] + sparseSizes[1])
      denseData[indices[idx] + offset] -= centers[0];
    else if (idx >= indexOffsets[2] && idx < indexOffsets[2] + sparseSizes[2])
      denseData[indices[idx] + offset] += centers[1];
    else if (idx >= indexOffsets[3] && idx < indexOffsets[3] + sparseSizes[3])
      denseData[indices[idx] + offset] -= centers[1];
  }
}

struct is_between
{
  float val1_;
  float val2_;
  float sign_;

  is_between(float val1, float val2, float sign) : val1_(val1), val2_(val2), sign_(sign) {}

  __host__ __device__
  bool operator()(const float x)
  {
    return x * sign_ > val1_ && x * sign_ < val2_;
  }
};

struct is_bigger_equals_than
{
  float val_;
  float sign_;

  is_bigger_equals_than(float val, float sign) : val_(val), sign_(sign) {}

  __host__ __device__
  bool operator()(const float x)
  {
    return x * sign_ >= val_;
  }
};

void QuantizedTensor32BitIndices::findSubtensorBucket(
      int idx, int* indexOffsets, int* sizes, int pos, int size) {
  int* start;
  int* end;
  CUDA_CHECK(cudaMalloc(&start, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&end, sizeof(int)));
  cudaMemset(start, -1, sizeof(int));
  cudaMemset(end, 0, sizeof(int));

  int blocks = 1 + size_ / THREAD_COUNT;
  gFindSubtensor32BitIndices<<<blocks, THREAD_COUNT>>>(
      indices_ + indexOffsets_[idx], sizes_[idx], pos, pos + size, start, end);

  int startOffset;
  int endOffset;
  CUDA_CHECK(cudaMemcpy(&startOffset, start, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&endOffset, end, sizeof(int), cudaMemcpyDeviceToHost));
  int subtensorSize = max(0, endOffset - startOffset + 1);

  if (startOffset > -1) {
    indexOffsets[idx] = indexOffsets_[idx] + startOffset;
    sizes[idx] = subtensorSize;
  }
  cudaFree(start);
  cudaFree(end);

  cudaStreamSynchronize(0);  
}

QuantizedTensor QuantizedTensor32BitIndices::subtensor(int pos, int size) {
  cudaSetDevice(device_);
  cudaStreamSynchronize(0);

  QuantizedTensor t = QuantizedTensor(new QuantizedTensor32BitIndices(
      device_, indices_, centers_));
  int* sizes = std::dynamic_pointer_cast<QuantizedTensor32BitIndices>(t)->sizes();
  int* indexOffsets = std::dynamic_pointer_cast<QuantizedTensor32BitIndices>(t)->indexOffsets();

  findSubtensorBucket(0, indexOffsets, sizes, pos, size);
  findSubtensorBucket(1, indexOffsets, sizes, pos, size);
  findSubtensorBucket(2, indexOffsets, sizes, pos, size);
  findSubtensorBucket(3, indexOffsets, sizes, pos, size);

  t->setSize(indexOffsets[3] + sizes[3]);
  
  return t;
}

void QuantizedTensor32BitIndices::copyFrom(QuantizedTensor t) {
  size_t size = t->size();
  
  if (size_ < size) {
    return;
  }

  if(size == 0)
    return;

  size_ = size;  

  cudaSetDevice(device_);
  cudaStreamSynchronize(0);

  float* centers = t->centers();
  int* indices = std::dynamic_pointer_cast<QuantizedTensor32BitIndices>(t)->indices();
  int* sizes = std::dynamic_pointer_cast<QuantizedTensor32BitIndices>(t)->sizes();  
  int* indexOffsets = std::dynamic_pointer_cast<QuantizedTensor32BitIndices>(t)->indexOffsets();

  CUDA_CHECK(cudaMemcpy(centers_, centers, (1 << (bits_ - 1)) * sizeof(float), cudaMemcpyDefault));
  CUDA_CHECK(cudaMemcpy(indices_, indices, size * sizeof(int), cudaMemcpyDefault));
  memcpy(sizes_, sizes, (1 << bits_) * sizeof(int));
  memcpy(indexOffsets_, indexOffsets, (1 << bits_) * sizeof(int));
  cudaStreamSynchronize(0);
}

void QuantizedTensor32BitIndices::scatterAdd(Tensor t, int offset = 0) {
  cudaSetDevice(device_);
  cudaStreamSynchronize(0);

  int* sizes;
  int* indexOffsets;
  int bucketsSize = sizeof(int) * (1 << bits_);
  CUDA_CHECK(cudaMalloc(&sizes, bucketsSize));
  CUDA_CHECK(cudaMalloc(&indexOffsets, bucketsSize));
  CUDA_CHECK(cudaMemcpy(sizes, sizes_, bucketsSize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(indexOffsets, indexOffsets_, bucketsSize, cudaMemcpyHostToDevice));

  int blocks = 1 + size_ / THREAD_COUNT;
  gScatterAdd32BitIndices<<<blocks, THREAD_COUNT>>>(
      t->data(), centers_, indices_, t->size(), sizes, indexOffsets, offset);
  cudaFree(sizes);
  cudaFree(indexOffsets);
  cudaStreamSynchronize(0);
}

template<typename UnaryPredicate>
thrust::device_ptr<int> QuantizedTensor32BitIndices::fillIndicesFromIndex(int idx, Tensor t, UnaryPredicate predicate) {
  cudaSetDevice(device_);

  thrust::device_ptr<float> data(t->data());
  thrust::device_ptr<int> indices(indices_);

  thrust::device_ptr<int> it(indices_);
  try {
    thrust::counting_iterator<int> it1(0);
    thrust::counting_iterator<int> it2 = it1 + t->size();
    it = thrust::copy_if(
        it1,
        it2,
        data,
        indices + indexOffsets_[idx],
        predicate);
    sizes_[idx] = it - &indices[0] - indexOffsets_[idx];
  } catch(thrust::system_error e) {
    std::cerr << "Error inside copy_if " << idx << ": " << e.what() << std::endl;
  }

  return it;
}

void QuantizedTensor32BitIndices::encode(Tensor t) {
  computeCenters(t);

  indexOffsets_[0] = 0;
  thrust::device_ptr<int> indices(indices_);

  auto it = fillIndicesFromIndex(0, t, is_between(get(centers_, 0), get(centers_, 1), 1));
  indexOffsets_[1] = it - &indices[0];
  it = fillIndicesFromIndex(1, t, is_between(get(centers_, 0), get(centers_, 1), -1));
  indexOffsets_[2] = it - &indices[0];
  it = fillIndicesFromIndex(2, t, is_bigger_equals_than(get(centers_, 1), 1));
  indexOffsets_[3] = it - &indices[0];
  it = fillIndicesFromIndex(3, t, is_bigger_equals_than(get(centers_, 1), -1));

  size_ = indexOffsets_[3] + sizes_[3];

  cudaStreamSynchronize(0);
}

void QuantizedTensor32BitIndices::decode(Tensor t, int offset) {
  cudaSetDevice(device_);

  int* sizes;
  int* indexOffsets;
  int bucketsSize = sizeof(int) * (1 << bits_);
  CUDA_CHECK(cudaMalloc(&sizes, bucketsSize));
  CUDA_CHECK(cudaMalloc(&indexOffsets, bucketsSize));
  CUDA_CHECK(cudaMemcpy(sizes, sizes_, bucketsSize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(indexOffsets, indexOffsets_, bucketsSize, cudaMemcpyHostToDevice));

  int blocks = 1 + size_ / THREAD_COUNT;
  t->set(0);
  gScatterAdd32BitIndices<<<blocks, THREAD_COUNT>>>(
      t->data(), centers_, indices_, t->size(), sizes, indexOffsets, offset);
  cudaFree(sizes);
  cudaFree(indexOffsets);
  cudaStreamSynchronize(0);
}

}