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

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

bool GPUBFCAllocator::GetAllowGrowthValue(const GPUOptions& gpu_options) {
  const char* force_allow_growth_string =
      std::getenv("TF_FORCE_GPU_ALLOW_GROWTH");
  if (force_allow_growth_string == nullptr) {
    return gpu_options.allow_growth();
  }

  if (strcmp("false", force_allow_growth_string) == 0) {
    if (gpu_options.allow_growth()) {
      LOG(WARNING)
          << "Overriding allow_growth setting because the"
          << " TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original"
          << " config value was " << gpu_options.allow_growth() << ".";
    }
    return false;
  } else if (strcmp("true", force_allow_growth_string) == 0) {
    if (!gpu_options.allow_growth()) {
      LOG(WARNING)
          << "Overriding allow_growth setting because the"
          << " TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original"
          << " config value was " << gpu_options.allow_growth() << ".";
    }
    return true;
  }

  LOG(ERROR)
      << "The TF_FORCE_GPU_ALLOW_GROWTH environment variable is set but could"
      << " not be parsed: \"" << force_allow_growth_string << "\". Valid"
      << " values are \"true\" or \"false\". Using original config value"
      << " of " << gpu_options.allow_growth() << ".";
  return gpu_options.allow_growth();
}

bool GPUBFCAllocator::GetGarbageCollectionValue() {
  const char* enable_gpu_garbage_collection =
      std::getenv("TF_ENABLE_GPU_GARBAGE_COLLECTION");
  if (enable_gpu_garbage_collection == nullptr) {
    // By default, turn on the memory garbage collection.
    return true;
  }
  if (strcmp("false", enable_gpu_garbage_collection) == 0) {
    return false;
  } else if (strcmp("true", enable_gpu_garbage_collection) == 0) {
    return true;
  }

  LOG(ERROR)
      << "The TF_ENABLE_GPU_GARBAGE_COLLECTION environment variable is set but"
      << " could not be parsed: \"" << enable_gpu_garbage_collection << "\"."
      << " Valid values are \"true\" or \"false\"."
      << " Using the default value \"true\".";
  return true;
}

GPUBFCAllocator::GPUBFCAllocator(GPUMemAllocator* sub_allocator,
                                 size_t total_memory, const string& name)
    : GPUBFCAllocator(sub_allocator, total_memory, GPUOptions(), name) {}

GPUBFCAllocator::GPUBFCAllocator(GPUMemAllocator* sub_allocator,
                                 size_t total_memory,
                                 const GPUOptions& gpu_options,
                                 const string& name)
    : BFCAllocator(sub_allocator, total_memory,
                   GPUBFCAllocator::GetAllowGrowthValue(gpu_options), name,
                   GPUBFCAllocator::GetGarbageCollectionValue()),
      stream_exec_(sub_allocator->stream_executor()) {
  if (gpu_options.experimental().lms_enabled()) {
    SetLMSConfig(true, gpu_options.experimental().lms_defrag_enabled());
    H2D_stream_ = new se::Stream(stream_exec_);
    H2D_stream_->Init();
    D2H_stream_ = new se::Stream(stream_exec_);
    D2H_stream_->Init();
    event_mgr_ = EventMgrFactory::Singleton()->GetEventMgr(stream_exec_, gpu_options);
  }
}

void GPUBFCAllocator::SetStreams(se::Stream* compute) {
  compute_stream_ = compute;
}

void* GPUBFCAllocator::Pagein(const LMSTensorBuffer *buf) {
  size_t nbytes = buf->size();
  void *host_ptr = buf->GetHostPtr();
  void *device_ptr = AllocateRaw(Allocator::kAllocatorAlignment, nbytes);

  VLOG(2) << "PAGEIN  <- " << (void*)buf << " (" << nbytes << ")";
  se::DeviceMemoryBase dst(device_ptr, nbytes);
  auto result = stream_exec_->SynchronousMemcpyH2D(host_ptr, nbytes, &dst);
  CHECK(result.ok());
  return device_ptr;
}

void* GPUBFCAllocator::PageinAsync(const LMSTensorBuffer *buf,
                                   const std::function<void()>& done) {
  size_t nbytes = buf->size();
  void *host_ptr = buf->GetHostPtr();
  void *device_ptr = buf->GetDevicePtr();

  if (device_ptr == nullptr) {
    device_ptr = AllocateRaw(Allocator::kAllocatorAlignment, nbytes);
  }

  VLOG(2) << "PAGEIN  <- " << (void*)buf << " (" << nbytes << ") ASYNC";
  se::DeviceMemoryBase dst(device_ptr, nbytes);

  // Wait for the compute stream to make sure the device buffer is truly available.
  H2D_stream_->ThenWaitFor(compute_stream_);

  H2D_stream_->ThenMemcpy(&dst, host_ptr, nbytes);
  event_mgr_->ThenExecute(H2D_stream_,
                          [this, done]() {
                            CHECK(this->H2D_stream_->ok());
                            done();
                          });
  return device_ptr;
}

void* GPUBFCAllocator::Pageout(const LMSTensorBuffer *buf) {
  size_t nbytes = buf->size();
  void *device_ptr = buf->GetDevicePtr();
  void *host_ptr = buf->GetHostPtr();
  if (host_ptr == nullptr) {
    host_ptr = host_allocator()->AllocateRaw(Allocator::kAllocatorAlignment, nbytes);
  }

  VLOG(2) << "-> PAGEOUT " << (void*)buf << " (" << nbytes << ")";
  const se::DeviceMemoryBase src(device_ptr, nbytes);
  auto result = stream_exec_->SynchronousMemcpyD2H(src, nbytes, host_ptr);
  CHECK(result.ok());
  return host_ptr;
}

void* GPUBFCAllocator::PageoutAsync(const LMSTensorBuffer *buf,
                                    const std::function<void()>& done) {
  size_t nbytes = buf->size();
  void *device_ptr = buf->GetDevicePtr();
  void *host_ptr = buf->GetHostPtr();
  if (host_ptr == nullptr) {
    host_ptr = host_allocator()->AllocateRaw(Allocator::kAllocatorAlignment, nbytes);
  }

  VLOG(2) << "-> PAGEOUT " << (void*)buf << " (" << nbytes << ") ASYNC";
  const se::DeviceMemoryBase src(device_ptr, nbytes);

  // Wait for the compute stream to make sure the data is available.
  D2H_stream_->ThenWaitFor(compute_stream_);

  D2H_stream_->ThenMemcpy(host_ptr, src, nbytes);
  event_mgr_->ThenExecute(D2H_stream_,
                          [this, done]() {
                            CHECK(this->D2H_stream_->ok());
                            done();
                          });
  return host_ptr;
}

void GPUBFCAllocator::HostMemoryDeallocate(void *host_ptr) {
  host_allocator()->DeallocateRaw(host_ptr);
}

void GPUBFCAllocator::EnsureHostAllocator() {
  std::call_once(host_allocator_init_,
                 [&] { host_allocator_ = GPUProcessState::singleton()->GetGpuHostAllocator(0); });
}

}  // namespace tensorflow
