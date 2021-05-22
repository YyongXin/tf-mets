/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SINGA_CORE_COMMON_H_
#define SINGA_CORE_COMMON_H_
#include <atomic>
#include <chrono>
#include <memory>
#include <random>
#include <string>

#include "singa/singa_config.h"
#include "singa/utils/logging.h"

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#ifdef USE_CUDNN
#include <cudnn.h>
#endif
#endif  // USE_CUDA

#ifdef USE_DNNL
#include <dnnl.hpp>
#endif  // USE_DNNL

#ifdef USE_OPENCL
#include "singa/utils/opencl_utils.h"
#endif  // USE_OPENCL

using std::atomic;

namespace singa {

enum OpType {
    kUndefined,
    kCopyH2H,
    kCopyH2D,
    kCopyD2H,
    kCopyD2D,
    kSync,
    kFwdPool,
    kBwdPool,
    kFwdBN,
    kBwdBN,
    kFwdRNN,
    kBwdRNN,
    kFwdActivation,
    kBwdActivation,
    kFwdDropout,
    kBwdDropout,
    kFwdConv,
    kBwdConvBias,
    kBwdConvWeight,
    kBwdConvNeuron,
    kFwdSoftmax,
    kBwdSoftmax,
    kFwdLrn,
    kBwdLrn,
    kCastType,
    kL1,
    kL2,
    kAbs,
    kCeil,
    kExp,
    kLog,
    kReLU,
    kSigmoid,
    kSoftPlus,
    kSoftSign,
    kSign,
    kSqrt,
    kSquare,
    kTransform,
    kCos,
    kCosh,
    kAcos,
    kAcosh,
    kSin,
    kSinh,
    kAsin,
    kAsinh,
    kTan,
    kTanh,
    kAtan,
    kAtanh,
    kSoftMax,
    kBiasAdd,
    kAdd,
    kSub,
    kEltwiseMult,
    kDiv,
    kdwPow,
    kPow,
    kLT,
    kLE,
    kGT,
    kGE,
    kReLUBackward,
    kDot,
    kRowMax,
    kGEMM,
    kGEMV,
    kRand,
    kAxpy,
    kCrossEntropy,
    kSoftmaxCrossEntropy,
    kMultColumn,
    kMultRow,
    kMult,
};

std::string op_type_to_string(OpType type);

namespace lang {
/// To implemente functions using cpp libraries
typedef struct _Cpp {
} Cpp;
/// To implemente functions using cuda libraries
typedef struct _Cuda {
} Cuda;
/// To implement function using opencl libraries
typedef struct _Opencl { } Opencl; }  // namespace lang

class Device;
struct DeviceOptInfoToAppend;
/// Block represent a chunk of memory (on device or host).
class Block {
 public:
  Block(void* ptr, size_t size, Device* device = nullptr, size_t offset = 0)
      : data_(ptr), size_(size), offset_(offset), device_(device) {
    ref_count_ = 1;  // std::make_shared<std::atomic<int>>(1);
  }
  // Disabled as it is not used currently.
  // Block(void* ptr, size_t size, size_t offset, std::shared_ptr<atomic<int>>
  //  ref) : data_(ptr), size_(size), offset_(offset), ref_count_(ref) {}
  /// Memory block write.
  void* mutable_data();
  /// Memory block read.
  const void* data() const;
  void free_data();
  void* get_data();
  void update_data(void* new_data);

  size_t size() const { return size_; }
  size_t offset() const { return offset_; }
  int IncRefCount() {
    return ++ref_count_;  // Note do not use ref_count_++;
  }
  int DecRefCount() { return --ref_count_; }
  int ref_count() const { return ref_count_.load(); }

  bool initialized() const { return initialized_; }

  void SetEstSwapOutTime(double time) { est_swap_out_time_ = time; }
  void SetEstSwapInTime(double time) { est_swap_in_time_ = time; }
  double GetEstSwapOutTime() { return est_swap_out_time_; }
  double GetEstSwapInTime() { return est_swap_in_time_; }

 private:
  Block() {}
  void* data_ = nullptr;
  size_t size_ = 0;
  size_t offset_ = 0;
  bool initialized_ = false;
  Device* device_ = nullptr;
  // Disabled as it is not used currently.
  // std::shared_ptr<std::atomic<int>> ref_count_ = nullptr;
  std::atomic<int> ref_count_;
  double est_swap_out_time_ = 0.;  // us
  double est_swap_in_time_ = 0.;  // us
};

/// For append purpose in the device class.
struct DeviceOptInfoToAppend {
  std::string mem_op_type;
  std::string block_ptr;
  int size;
  long time_stamp =
      (std::chrono::system_clock::now()).time_since_epoch().count();

  DeviceOptInfoToAppend(std::string op_type, std::string ptr, int s)
      : mem_op_type(op_type), block_ptr(ptr), size(s) {}
};

typedef struct _Context {
  std::mt19937 random_generator;
#ifdef USE_CUDA
  cublasHandle_t cublas_handle;
  cudaStream_t stream;
  curandGenerator_t curand_generator;
#ifdef USE_CUDNN
  cudnnHandle_t cudnn_handle;
#endif
#endif  // USE_CUDA

#ifdef USE_DNNL
  dnnl::engine dnnl_engine;
  dnnl::stream dnnl_stream;
#endif  // USE_DNNL

#ifdef USE_OPENCL
  // This stores the context ID of the OpenCL context controlled by ViennaCL.
  long vcl_ctx_id;
#endif

} Context;

}  // namespace singa
#endif  // SINGA_CORE_COMMON_H_
