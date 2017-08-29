#include <cuda.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "kernels/cuda_helpers.h"
#include "quantization/quantized_tensor.h"

namespace marian {

__global__ void sample(
    float* originalData, float* data, int size, int scale) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;
  data[idx] = abs(originalData[idx * scale]);
}

void QuantizedTensorBase::computeCenters(Tensor t) {
  int totalSize = t->size();
  int sortSize = min(100000, totalSize);
  int blocksSample = 1 + sortSize / THREAD_COUNT;

  cudaSetDevice(device_);
  cudaStreamSynchronize(0);

  float* tmpData;
  CUDA_CHECK(cudaMalloc(&tmpData, sizeof(float) * sortSize));
  cudaMemset(tmpData, 0, sizeof(float) * sortSize);

  sample<<<blocksSample, THREAD_COUNT>>>(
      t->data(), tmpData, sortSize, totalSize / sortSize);
  thrust::device_ptr<float> tmpDataPtr(tmpData);
  try {
    thrust::sort(tmpDataPtr, tmpDataPtr + sortSize);
  } catch(thrust::system_error &e) {
    LOG(warn)->warn("Sort error {}", e.what());
  }

  int cutOffIndex = std::max(0, (int)(sortSize * dropRate_) - 1);
  float minVal = get(tmpData, cutOffIndex);

  set(centers_, 0, minVal);

  long long bucketCount = (1 << (bits_ - 1));
  if (bucketCount > 1) {
    float maxVal = get(tmpData, sortSize - 1);
    float nextMinVal = (maxVal + minVal) / bucketCount;

    set(centers_, 1, nextMinVal);
  }

  cudaFree(tmpData);
  cudaStreamSynchronize(0);
}

}