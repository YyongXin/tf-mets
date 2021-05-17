/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif  // GOOGLE_CUDA

#include "tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/platform/stream_executor.h"
#define TF_GPU_MEM_TRACE
namespace tensorflow {

GPUcudaMallocAllocator::GPUcudaMallocAllocator(Allocator* allocator,
                                               PlatformGpuId platform_gpu_id)
    : base_allocator_(allocator) {
  stream_exec_ =
      GpuIdUtil::ExecutorForPlatformGpuId(platform_gpu_id).ValueOrDie();
}

GPUcudaMallocAllocator::~GPUcudaMallocAllocator() { delete base_allocator_; }

void* GPUcudaMallocAllocator::AllocateRaw(size_t alignment, size_t num_bytes) {
#ifdef GOOGLE_CUDA
  // allocate with cudaMalloc
  se::cuda::ScopedActivateExecutorContext scoped_activation{stream_exec_};
  CUdeviceptr rv = 0;
  CUresult res = cuMemAlloc(&rv, num_bytes);
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "cuMemAlloc failed to allocate " << num_bytes;
    return nullptr;
  }
#ifdef TF_GPU_MEM_TRACE
  std::fstream mem_info_log("mem-info.log", std::ios::in| std::ios::out| std::ios::app);
  int64_t time_stamp = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  mem_info_log << "MALLOC: " << rv << ' ' << num_bytes << ' ' << time_stamp << '\n';
  size_t available_memory = 0, total_memory = 0, used_memory = 0;
  cudaMemGetInfo(&available_memory, &total_memory);
  used_memory = total_memory - available_memory;
  std::fstream cuda_mem_log("cuda-memory.log", std::ios::in| std::ios::out| std::ios::app);
  cuda_mem_log << (double)(used_memory) / 1024.0 / 1024.0 << ' '
	  << (double)(available_memory) / 1024.0 / 1024.0 << ' '
	  << (double)(total_memory) / 1024.0 / 1024.0 << '\n';
  std::cout << (double)(used_memory) / 1024.0 / 1024.0 << ' '
	  << (double)(available_memory) / 1024.0 / 1024.0 << ' '
	  << (double)(total_memory) / 1024.0 / 1024.0 << '\n';
#endif // TF_GPU_MEM_TRACE
  return reinterpret_cast<void*>(rv);
#else
  return nullptr;
#endif  // GOOGLE_CUDA
}
void GPUcudaMallocAllocator::DeallocateRaw(void* ptr) {
#ifdef GOOGLE_CUDA
  // free with cudaFree
  CUresult res = cuMemFree(reinterpret_cast<CUdeviceptr>(ptr));
  if (res != CUDA_SUCCESS) {
    LOG(ERROR) << "cuMemFree failed to free " << ptr;
  }
#ifdef TF_GPU_MEM_TRACE
  std::fstream mem_info_log("mem-info.log", std::ios::in| std::ios::out| std::ios::app);
  int64_t time_stamp = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  mem_info_log << "FREE: " << ptr << ' ' << time_stamp << '\n';
  std::cout << "FREE: " << ptr << ' ' << time_stamp << '\n';
  size_t available_memory = 0, total_memory = 0, used_memory = 0;
  cudaMemGetInfo(&available_memory, &total_memory);
  used_memory = total_memory - available_memory;
  std::fstream cuda_mem_log("cuda-memory.log", std::ios::in| std::ios::out| std::ios::app);
  cuda_mem_log << (double)(used_memory) / 1024.0 / 1024.0 << ' '
	  << (double)(available_memory) / 1024.0 / 1024.0 << ' '
	  << (double)(total_memory) / 1024.0 / 1024.0 << '\n';
#endif // TF_GPU_MEM_TRACE
#endif  // GOOGLE_CUDA
}

absl::optional<AllocatorStats> GPUcudaMallocAllocator::GetStats() {
  return base_allocator_->GetStats();
}

bool GPUcudaMallocAllocator::TracksAllocationSizes() const { return false; }

}  // namespace tensorflow
