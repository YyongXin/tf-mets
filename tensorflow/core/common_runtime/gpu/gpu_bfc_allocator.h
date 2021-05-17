/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Copyright 2019, 2020. IBM All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_mem_allocator.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// A GPU memory allocator that implements a 'best-fit with coalescing'
// algorithm.
class GPUBFCAllocator : public BFCAllocator {
 public:
  GPUBFCAllocator(GPUMemAllocator* sub_allocator, size_t total_memory,
                  const string& name);
  GPUBFCAllocator(GPUMemAllocator* sub_allocator, size_t total_memory,
                  const GPUOptions& gpu_options, const string& name);
  ~GPUBFCAllocator() override {}

  void SetStreams(se::Stream* compute) override;
  void* Pagein(const LMSTensorBuffer *buf) override;
  void* PageinAsync(const LMSTensorBuffer *buf, const std::function<void()>& done) override;
  void* Pageout(const LMSTensorBuffer *buf) override;
  void* PageoutAsync(const LMSTensorBuffer *buf, const std::function<void()>& done) override;
  void HostMemoryDeallocate(void *host_ptr) override;

  TF_DISALLOW_COPY_AND_ASSIGN(GPUBFCAllocator);

#ifdef TENSORFLOW_MEM_DEBUG
  bool ShouldRecordOpName() const override { return true; }
#endif

 private:
  static bool GetAllowGrowthValue(const GPUOptions& gpu_options);
  static bool GetGarbageCollectionValue();

  // Large Model Support
  se::StreamExecutor* stream_exec_;  // not owned, non-null
  se::Stream* H2D_stream_ = nullptr;
  se::Stream* D2H_stream_ = nullptr;
  se::Stream* compute_stream_ = nullptr;
  EventMgr* event_mgr_ = nullptr;
  Allocator* host_allocator_ = nullptr;

  void EnsureHostAllocator();
  std::once_flag host_allocator_init_;
  inline Allocator* host_allocator() {
    if (host_allocator_ == nullptr) EnsureHostAllocator();
    return host_allocator_;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
