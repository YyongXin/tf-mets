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

#include <cassert>
#include <fstream>
#include "singa/core/common.h"

#include "singa/core/device.h"

#define MEM_LOG_DBG

namespace singa {

std::string op_type_to_string(OpType type) {
#define AddOPType(tp) case OpType::k##tp: return #tp;
    switch (type) {
        AddOPType(Undefined)
        AddOPType(CopyH2H)
        AddOPType(CopyH2D)
        AddOPType(CopyD2H)
        AddOPType(CopyD2D)
        AddOPType(Sync)
        AddOPType(FwdPool)
        AddOPType(BwdPool)
        AddOPType(FwdBN)
        AddOPType(BwdBN)
        AddOPType(FwdRNN)
        AddOPType(BwdRNN)
        AddOPType(FwdActivation)
        AddOPType(BwdActivation)
        AddOPType(FwdDropout)
        AddOPType(BwdDropout)
        AddOPType(FwdConv)
        AddOPType(BwdConvBias)
        AddOPType(BwdConvWeight)
        AddOPType(BwdConvNeuron)
        AddOPType(FwdSoftmax)
        AddOPType(BwdSoftmax)
        AddOPType(FwdLrn)
        AddOPType(BwdLrn)
        AddOPType(CastType)
        AddOPType(L1)
        AddOPType(L2)
        AddOPType(Abs)
        AddOPType(Ceil)
        AddOPType(Exp)
        AddOPType(Log)
        AddOPType(ReLU)
        AddOPType(Sigmoid)
        AddOPType(SoftPlus)
        AddOPType(SoftSign)
        AddOPType(Sign)
        AddOPType(Sqrt)
        AddOPType(Square)
        AddOPType(Transform)
        AddOPType(Cos)
        AddOPType(Cosh)
        AddOPType(Acos)
        AddOPType(Acosh)
        AddOPType(Sin)
        AddOPType(Sinh)
        AddOPType(Asin)
        AddOPType(Asinh)
        AddOPType(Tan)
        AddOPType(Tanh)
        AddOPType(Atan)
        AddOPType(Atanh)
        AddOPType(SoftMax)
        AddOPType(BiasAdd)
        AddOPType(Add)
        AddOPType(Sub)
        AddOPType(EltwiseMult)
        AddOPType(Div)
        AddOPType(dwPow)
        AddOPType(Pow)
        AddOPType(LT)
        AddOPType(LE)
        AddOPType(GT)
        AddOPType(GE)
        AddOPType(ReLUBackward)
        AddOPType(Dot)
        AddOPType(RowMax)
        AddOPType(GEMM)
        AddOPType(GEMV)
        AddOPType(Rand)
        AddOPType(Axpy)
        AddOPType(CrossEntropy)
        AddOPType(SoftmaxCrossEntropy)
        AddOPType(MultColumn)
        AddOPType(MultRow)
        AddOPType(Mult)
        default: assert(0);
    }
#undef AddOPType
    return "";
}

void* Block::mutable_data() {
  initialized_ = true;

  // Instrument block info: opt_type, ptr and time_stamp.
  if (device_ != nullptr && device_->device_type_ == DT_SwapCudaGPU) {
    std::stringstream ss;
    ss << this;
    std::string tmp_str = ss.str();
    DeviceOptInfoToAppend info("Mutable", tmp_str, size());
    auto t = (std::chrono::system_clock::now()).time_since_epoch().count();
    info.time_stamp = t;
    // std::cout << "==> " << __func__ << ": " << __FILE__ << ":" << __LINE__
    //          << std::endl;
    device_->Append(info);
  }

  // We need to update ptr after swap-in is done, if variable is not swapped
  // back yet as expected.
  if (data_ == nullptr && size_ > 0) {
    if (device_->device_type_ != DT_SwapCudaGPU) {
      data_ = device_->Malloc((int)size_);
    }
    if (device_->device_type_ == DT_SwapCudaGPU) {
      // std::cout << "==> " << __func__ << ": " << __FILE__ << ":" << __LINE__
      //          << std::endl;
      auto tmp_data_ = device_->UpdateGpuPtrInfo(this);
      return static_cast<char*>(tmp_data_) + offset_;
    }
  }

#ifdef MEM_LOG_DBG
  std::fstream mem_info_log("mem-info.log", std::ios::in| std::ios::out| std::ios::app);
  int64_t time_stamp = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  mem_info_log << "WRITE: " << data_ << ' ' << size() << ' ' << time_stamp << '\n';
#endif
  // Original case.
  return static_cast<char*>(data_) + offset_;
}

const void* Block::data() const {
  CHECK(initialized_) << "Must initialize data before reading it";

  // Instrument block info: opt_type, ptr, time_stamp.
  if (device_ != nullptr) {
    std::stringstream ss;
    ss << this;
    std::string tmp_str = ss.str();
    DeviceOptInfoToAppend info("Read", tmp_str, size());
    auto t = (std::chrono::system_clock::now()).time_since_epoch().count();
    info.time_stamp = t;
    device_->Append(info);
  }

  // We need to update ptr after swap-in is done, if variable is not swapped
  // back yet as expected.
  if (data_ == nullptr && device_->device_type_ == DT_SwapCudaGPU) {
    auto tmp_data_ = device_->UpdateGpuPtrInfo(this);
    return static_cast<char*>(tmp_data_) + offset_;
  }

#ifdef MEM_LOG_DBG
  std::fstream mem_info_log("mem-info.log", std::ios::in| std::ios::out| std::ios::app);
  int64_t time_stamp = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  mem_info_log << "READ: " << data_ << ' ' << size() << ' ' << time_stamp << '\n';
#endif
  return static_cast<char*>(data_) + offset_;
}

/// Get data without calling the original data() to avoid appending block info.
void* Block::get_data() { return data_; }

// Update data_, after the swap-in completes.
void Block::update_data(void* new_data) { data_ = new_data; }

void Block::free_data() {
  if (data_) {
    device_->Free(data_);
    data_ = nullptr;
    initialized_ = false;
  }
}

}  // namespace singa
