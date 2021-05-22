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
#include "singa/core/device.h"

namespace singa {

const int WARMUP_TIMES = 8;
const int REPEAT_TIMES = 32;
//bool Device::lazy_alloc_ = true;
bool Device::lazy_alloc_ = false;

Device::Device(int id, int num_executors)
    : id_(id), num_executors_(num_executors) {
  // TODO(wangwei) create scheduler and vm.
  host_ = defaultDevice;
  graph_ = new Graph(this);
}

Device::Device(DeviceType dt, int id, int num_executors)
    : device_type_(dt), id_(id), num_executors_(num_executors) {
  // TODO(wangwei) create scheduler and vm.
  host_ = defaultDevice;
  graph_ = new Graph(this);
}

Device::~Device() {
  if (graph_) {
    delete graph_;
  }
}

void Device::EstimateGraphNodeTime() {
   for (auto &&node : graph_->nodes()) {
       double time = 0;
       clock_t start = 0, end = 0;
       for (int i=0; i<REPEAT_TIMES+WARMUP_TIMES; ++i) {
           if (i == WARMUP_TIMES) { start = clock(); }
           DoExec(std::move(node->op()), 0);
       }
       end = clock();
       time = (double)(end-start)*1e6/REPEAT_TIMES/CLOCKS_PER_SEC;
       node->SetEstimeTime(time);
   }
}

void Device::EstimateBlockSwapTime() {
    for (auto &&blk_info : graph_->blocks()) {
        auto blk = blk_info.first;
        double time = 0;
        clock_t start = 0, end = 0;
        for (int i=0; i<REPEAT_TIMES+WARMUP_TIMES; ++i) {
            if (i == WARMUP_TIMES) { start = clock(); }
            // TODO:
        }
        end = clock();
        // unit: us
        time = (double)(end-start)*1e6/REPEAT_TIMES/CLOCKS_PER_SEC;
        blk->SetEstSwapInTime(time);
    }
    for (auto &&blk_info : graph_->blocks()) {
        auto blk = blk_info.first;
        double time = 0;
        clock_t start = 0, end = 0;
        for (int i=0; i<REPEAT_TIMES+WARMUP_TIMES; ++i) {
            if (i == WARMUP_TIMES) { start = clock(); }
            // TODO:
        }
        end = clock();
        // unit: us
        time = (double)(end-start)*1e6/REPEAT_TIMES/CLOCKS_PER_SEC;
        blk->SetEstSwapOutTime(time);
    }
}

void Device::Exec(function<void(Context*)>&& fn,
                  OpType type,
                  const vector<Block*> read_blocks,
                  const vector<Block*> write_blocks, bool use_rand_generator) {
  if (graph_enabled_ == true) {
    graph_->AddOperation(std::move(fn), type, read_blocks, write_blocks);
  } else {
    // printf("immediately ops\n");
    DoExec(std::move(fn), 0);
  }
}

void Device::RunGraph(bool serial) {
  bool previous_state = graph_enabled_;
  graph_enabled_ = false;

  if (serial) {
    // sequential execution
    graph_->RunInSerial();
  } else {
    // execute according to dependencies
    graph_->RunGraph();
  }

  //graph_->Debug();

  graph_enabled_ = previous_state;
}

// Todo(Wangwei) Get Block From The Memory manager
Block* Device::NewBlock(int size) {
  CHECK_GE(size, 0)
      << "size is negative, could be caused by the type cast "
      << "from size_t to int. In that case, the size is too large.";
  if (size > 0) {
    void* ptr = nullptr;
    // NOTICE: Singa supports lazy allocation. That means when tensor or blocks
    // are created, devices do not allocate memory for them immediately.
    // Instead, when the block is accessed for the first time, the memory is
    // allocated. More details, see <http://singa.apache.org/docs/graph/>.
    // If we want to trace the malloc/free sequence, we MUST disable lazy
    // allocation.
    if (device_type_ == DT_SwapCudaGPU) {
      lazy_alloc_ = false;
    }
    if (!lazy_alloc_) {
      ptr = Malloc(size);
    }
    Block* blk = new Block(ptr, size, this);
    // Make table and append vec_block.
    AppendAfterMalloc(blk, ptr, size);
    return blk;
  } else {
    return nullptr;
  }
}

// TODO(wangwei) return Block to the memory manager
void Device::FreeBlock(Block* block) {
  if (block != nullptr) {
    // Free(block->mutable_data());
    auto tmp_ptr = block->mutable_data();
    Free(tmp_ptr);
    // Instrument block info for free operation.
    std::stringstream ss;
    ss << block;
    std::string tmp_str = ss.str();
    DeviceOptInfoToAppend info("Free", tmp_str, block->size());
    auto t = (std::chrono::system_clock::now()).time_since_epoch().count();
    info.time_stamp = t;
    Append(info);
    delete block;
  }
}

void* Device::UpdateGpuPtrInfo(const Block* blk) { return UpdateGpuPtr(blk); }

OpType CopyDirectionToOpType(CopyDirection direct) {
    OpType type;
    switch (direct) {
        case CopyDirection::kHostToHost: type = OpType::kCopyH2H; break;
        case CopyDirection::kHostToDevice: type = OpType::kCopyH2D; break;
        case CopyDirection::kDeviceToHost: type = OpType::kCopyD2H; break;
        case CopyDirection::kDeviceToDevice: type = OpType::kCopyD2D; break;
        default: assert(0);
    }
    return type;
}

void Device::CopyDataToFrom(Block* dst, Block* src, size_t nBytes,
                            CopyDirection direct, int dst_offset,
                            int src_offset) {
  OpType type = CopyDirectionToOpType(direct);
  this->Exec(
      [this, dst, src, nBytes, direct, dst_offset, src_offset](Context* ctx) {
        this->CopyToFrom(
            reinterpret_cast<char*>(dst->mutable_data()) + dst_offset,
            reinterpret_cast<const char*>(src->data()) + src_offset, nBytes,
            direct, ctx);
      }, type,
      {src}, {dst});
}

void Device::CopyDataFromHostPtr(Block* dst, const void* src, size_t nBytes,
                                 size_t dst_offset) {
  auto direct = lang_ == kCpp ? kHostToHost : kHostToDevice;
  OpType type = CopyDirectionToOpType(direct);
  void* dstptr = reinterpret_cast<char*>(dst->mutable_data()) + dst_offset;
  Exec([this, dstptr, src, nBytes,
        direct](Context* ctx) { CopyToFrom(dstptr, src, nBytes, direct, ctx); },
       type, {}, {dst});
}
void Device::Sync() {}
}  // namespace singa
