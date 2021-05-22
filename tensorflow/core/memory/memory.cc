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
#ifndef DISABLE_WARNINGS

#include "singa/core/memory.h"

#include <cuda.h>
#include <stdint.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#include "singa/proto/core.pb.h"
#include "singa/utils/logging.h"

#define MEM_LOG_DBG

#ifdef USE_CUDA

namespace singa {
extern std::map<std::string, std::string> BlkTypeMap;

std::pair<size_t, size_t> CnMemPool::GetMemUsage() {
  size_t free, total;
  auto status = cnmemMemGetInfo(&free, &total, NULL);
  CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
      << cnmemGetErrorString(status);
  return std::make_pair(free, total);
}
std::pair<size_t, size_t> CnMemPool::GetMemUsage(int id) {
  CHECK_EQ(cudaSetDevice(id), cudaError_t::cudaSuccess);
  size_t free, total;
  auto status = cnmemMemGetInfo(&free, &total, NULL);
  CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
      << cnmemGetErrorString(status);
  return std::make_pair(free, total);
}

CnMemPool::CnMemPool(int numDevices, size_t init_size, size_t max_size) {
  for (int i = 0; i < numDevices; i++) conf_.add_device(i);
  conf_.set_init_size(init_size);
  conf_.set_max_size(max_size);
}

CnMemPool::CnMemPool(const MemPoolConf &conf) { conf_ = conf; }

void CnMemPool::Init() {
  mtx_.lock();
  if (!initialized_) {
    const size_t kNBytesPerMB = (1u << 20);
    CHECK_GE(conf_.device_size(), 1);
    cnmemDevice_t *settingPtr = new cnmemDevice_t[conf_.device_size()];
    CHECK_GT(conf_.init_size(), 0u);
    int i = 0;
    for (auto device : conf_.device()) {
      settingPtr[i].device = device;
      settingPtr[i].size = conf_.init_size() * kNBytesPerMB;
      settingPtr[i].numStreams = 0;
      settingPtr[i].streams = NULL;
      settingPtr[i].streamSizes = 0;
      i++;
    }
    auto status = cnmemInit(conf_.device_size(), settingPtr, conf_.flag());
    CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
        << " " << cnmemGetErrorString(status);
    delete[] settingPtr;
    initialized_ = true;
  }
  mtx_.unlock();
}

CnMemPool::~CnMemPool() {
  mtx_.lock();
  if (initialized_) {
    cnmemStatus_t status = cnmemFinalize();
    CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
        << " " << cnmemGetErrorString(status);
    initialized_ = false;
  }
  mtx_.unlock();
}

void CnMemPool::Malloc(void **ptr, const size_t size) {
  if (!initialized_) Init();
  cnmemStatus_t status = cnmemMalloc(ptr, size, NULL);
  CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
      << " " << cnmemGetErrorString(status);
#ifdef MEM_LOG_DBG
  std::fstream mem_info_log("mem-info.log",
                            std::ios::in | std::ios::out | std::ios::app);
  int64_t time_stamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
  std::stringstream tmp_ss;
  tmp_ss << *ptr;
  auto it = BlkTypeMap.find(tmp_ss.str());
  if (it != BlkTypeMap.end()) {
    mem_info_log << "MALLOC: Type: " << it->second << ' ' << *ptr << ' ' << size
                 << ' ' << time_stamp << '\n';
  } else {
    mem_info_log << "MALLOC: " << *ptr << ' ' << size << ' ' << time_stamp
                 << '\n';
  }
  size_t available_memory = 0, total_memory = 0, used_memory = 0;
  cudaMemGetInfo(&available_memory, &total_memory);
  used_memory = total_memory - available_memory;
  std::fstream cuda_mem_log("cuda-memory.log",
                            std::ios::in | std::ios::out | std::ios::app);
  cuda_mem_log << (double)(used_memory) / 1024.0 / 1024.0 << ' '
               << (double)(available_memory) / 1024.0 / 1024.0 << ' '
               << (double)(total_memory) / 1024.0 / 1024.0 << '\n';
#endif
}

void CnMemPool::Free(void *ptr) {
  CHECK(initialized_)
      << "Cannot free the memory as the pool is not initialzied";
  cnmemStatus_t status = cnmemFree(ptr, NULL);
  CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
      << " " << cnmemGetErrorString(status);
#ifdef MEM_LOG_DBG
  std::fstream mem_info_log("mem-info.log",
                            std::ios::in | std::ios::out | std::ios::app);
  int64_t time_stamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
  mem_info_log << "FREE: " << ptr << ' ' << time_stamp << '\n';
  size_t available_memory = 0, total_memory = 0, used_memory = 0;
  cudaMemGetInfo(&available_memory, &total_memory);
  used_memory = total_memory - available_memory;
  std::fstream cuda_mem_log("cuda-memory.log",
                            std::ios::in | std::ios::out | std::ios::app);
  cuda_mem_log << (double)(used_memory) / 1024.0 / 1024.0 << ' '
               << (double)(available_memory) / 1024.0 / 1024.0 << ' '
               << (double)(total_memory) / 1024.0 / 1024.0 << '\n';
#endif
}

// ===========================================================================
void CudaMemPool::Malloc(void **ptr, const size_t size) {
  cudaError_t status = cudaMalloc(ptr, size);
  CHECK_EQ(status, cudaError_t::cudaSuccess);
}

void CudaMemPool::Free(void *ptr) {
  cudaError_t status = cudaFree(ptr);
  CHECK_EQ(status, cudaError_t::cudaSuccess);
}

// ===========================================================================
// Helper utilities.
// TODO: Move these helper functions into common.h or anywhere else.

/// TODO: Simplify struct PoolOptInfo?
/// [ptr, size, mem_op_type, idx]
struct PoolOptInfo {
  std::string ptr;
  size_t size;
  int mem_op_type;
  int idx;
  PoolOptInfo(std::string ptr_, size_t size_, int op_type, int idx_)
      : ptr(ptr_), size(size_), mem_op_type(op_type), idx(idx_) {}
};
/// [idx, mem_op_type, size_delta]
struct PoolOptSimplifiedInfo {
  int idx;
  int mem_op_type;
  size_t size_delta;
  PoolOptSimplifiedInfo(size_t delta, int op_type, int idx_)
      : size_delta(delta), mem_op_type(op_type), idx(idx_) {}
};
/// Sort PoolOptInfo by ptr then idx.
struct sort_by_ptr_idx_ascending {
  inline bool operator()(const PoolOptInfo &info1, const PoolOptInfo &info2) {
    return ((info1.ptr < info2.ptr) ||
            ((info1.ptr == info2.ptr) && (info1.idx < info2.idx)));
  }
};
/// Sort PoolOptSimplifiedInfo by idx.
struct sort_by_itr_idx_ascending {
  inline bool operator()(const PoolOptSimplifiedInfo &info1,
                         const PoolOptSimplifiedInfo &info2) {
    return (info1.idx < info2.idx);
  }
};

struct PoolBlockLifeTime {
  int name;
  size_t size;
  int r_idx;
  int d_idx;
  PoolBlockLifeTime(int name_, size_t size_, int r, int d)
      : name(name_), size(size_), r_idx(r), d_idx(d) {}
};

/// Sort PoolBlockLifeTime by descending size and r_idx.
struct sort_by_size_r_idx_descending {
  inline bool operator()(const PoolBlockLifeTime &life_time1,
                         const PoolBlockLifeTime &life_time2) {
    return ((life_time1.size > life_time2.size) ||
            ((life_time1.size == life_time2.size) &&
             (life_time1.r_idx < life_time2.r_idx)));
  }
};

std::vector<std::string> SplitString(std::string str, std::string delimiter) {
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  std::string token;
  std::vector<std::string> res;
  while ((pos_end = str.find(delimiter, pos_start)) != std::string::npos) {
    token = str.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }
  res.push_back(str.substr(pos_start));
  return res;
}

/// TF memory management notes.
/// <https://github.com/miglopst/cs263_spring2018/wiki/Memory-management-for-tensorflow>
///
/// Merge consecutive/overlapping segments of vec_color_preoccupied
/// input: the collection of color ranges that is once occupied by some block
/// during a block's life time.
/// output: merged segments in ascending order.
std::vector<std::pair<size_t, size_t>> MergeColoredSegments(
    std::vector<std::pair<size_t, size_t>> vec_color_preoccupied) {
  std::sort(vec_color_preoccupied.begin(), vec_color_preoccupied.end());
  if (vec_color_preoccupied.size() <= 1) {
    return vec_color_preoccupied;
  }

  int m = 0;
  while (m < (vec_color_preoccupied.size() - 1)) {
    if ((vec_color_preoccupied[m].second + 2) >
        vec_color_preoccupied[m + 1].first) {
      std::pair<int, int> tmp_item(
          vec_color_preoccupied[m].first,
          std::max(vec_color_preoccupied[m].second,
                   vec_color_preoccupied[m + 1].second));
      // Remove m+1 and m.
      vec_color_preoccupied.erase(vec_color_preoccupied.begin() + m + 1);
      vec_color_preoccupied.erase(vec_color_preoccupied.begin() + m);
      // Insert the combined range.
      vec_color_preoccupied.insert(vec_color_preoccupied.begin() + m, tmp_item);
    } else {
      m += 1;
    }
  }
  return vec_color_preoccupied;
}

/// First fit weighted coloring.
/// return a pair standing for color_range.
/// local_offset shifts the returned color_range, allowing multiple Plan().
/// local_offset not changable, whereas offset is changable.
std::pair<size_t, size_t> FirstFitAllocation(
    std::vector<std::pair<size_t, size_t>> vec_color_merged, size_t size,
    size_t local_offset) {
  // If no occupied, put after the local_offset.
  if (vec_color_merged.size() == 0) {
    return std::pair<size_t, size_t>(0 + local_offset, size - 1 + local_offset);
  }

  // If it is able to fit before first block, after the local_offset.
  if ((size + local_offset) < (vec_color_merged[0].first + 1)) {
    return std::pair<size_t, size_t>(0 + local_offset, size - 1 + local_offset);
  }

  size_t y_location = -1;
  if (vec_color_merged.size() > 1) {
    int n = 0;
    while (n < (vec_color_merged.size() - 1)) {
      // If it is able to fit in between middle blocks.
      if ((vec_color_merged[n + 1].first - vec_color_merged[n].second - 1) >=
          size) {
        y_location = vec_color_merged[n].second + 1;
        break;
      }
      n += 1;
    }
    // Allocate after the last block.
    if (y_location == -1) {
      y_location = vec_color_merged[vec_color_merged.size() - 1].second + 1;
    }
  }

  // If color merger len =1, allocate after the last block.
  if (vec_color_merged.size() == 1) {
    y_location = vec_color_merged[0].second + 1;
  }

  if (y_location == -1) {
    std::cout << "error in FirstFitAllocation!!!" << std::endl;
  }

  return std::pair<size_t, size_t>(y_location, y_location + size - 1);
}

/// Best fit allocation, input and output same as FirstFitAllocation.
std::pair<size_t, size_t> BestFitAllocation(
    std::vector<std::pair<size_t, size_t>> vec_color_merged, size_t size,
    size_t local_offset) {
  // If no occupied, put after the local_offset.
  if (vec_color_merged.size() == 0) {
    return std::pair<size_t, size_t>(0 + local_offset, size - 1 + local_offset);
  }
  // If size=1, it is able to fit before the first block.
  if ((vec_color_merged.size() == 1) &&
      ((size + local_offset) < (vec_color_merged[0].first + 1))) {
    return std::pair<size_t, size_t>(0 + local_offset, size - 1 + local_offset);
  }

  if ((vec_color_merged.size() == 1) &&
      ((size + local_offset) >= (vec_color_merged[0].first + 1))) {
    return std::pair<size_t, size_t>(vec_color_merged[0].second + 1,
                                     vec_color_merged[0].second + size);
  }

  size_t y_location = -1;
  std::pair<int, size_t> tmp_hole(-1, -1);
  // n, hole size between n and n+1
  if (vec_color_merged.size() > 1) {
    int n = 0;
    while (n < (vec_color_merged.size() - 1)) {
      // It is able to fit in between middle blocks, select smallest.
      if (((vec_color_merged[n + 1].first - vec_color_merged[n].second - 1) >=
           size) &&
          ((vec_color_merged[n + 1].first - vec_color_merged[n].second - 1) <
           tmp_hole.second)) {
        tmp_hole.first = n;
        tmp_hole.second =
            vec_color_merged[n + 1].first - vec_color_merged[n].second - 1;
      }
      n += 1;
    }

    if (tmp_hole.first == -1) {
      // Allocate after the last block.
      y_location = vec_color_merged[vec_color_merged.size() - 1].second + 1;
    } else {
      // Best fit in the smallest hole.
      y_location = vec_color_merged[tmp_hole.first].second + 1;
    }
  }

  if (y_location == -1) {
    std::cout << "error in BestFitAllocation!" << std::endl;
  }

  return std::pair<size_t, size_t>(y_location, y_location + size - 1);
}

// ===========================================================================
// SwapPool

std::pair<size_t, size_t> SwapPool::GetMemUsage() {
  // size_t free, total;
  // auto status = cnmemMemGetInfo(&free, &total, NULL);
  // CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
  //    << cnmemGetErrorString(status);
  // return std::make_pair(free, total);

  // TODO: Implement GetMemUsage.
  return std::make_pair(0, 0);
}

std::pair<size_t, size_t> SwapPool::GetMemUsage(int id) {
  CHECK_EQ(cudaSetDevice(id), cudaError_t::cudaSuccess);
  // size_t free, total;
  // auto status = cnmemMemGetInfo(&free, &total, NULL);
  // CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
  //    << cnmemGetErrorString(status);
  // return std::make_pair(free, total);

  // TODO: Implement GetMemUsage.
  return std::make_pair(0, 0);
}

// SwapPool::SwapPool(int numDevices, size_t init_size, size_t max_size) {
//  for (int i = 0; i < numDevices; i++) conf_.add_device(i);
//  conf_.set_init_size(init_size);
//  conf_.set_max_size(max_size);
//}

SwapPool::SwapPool(const MemPoolConf &conf) { conf_ = conf; }

void SwapPool::Init() {
  mtx_.lock();
  if (!initialized_) {
    // const size_t kNBytesPerMB = (1u << 20);
    // CHECK_GE(conf_.device_size(), 1);
    // cnmemDevice_t *settingPtr = new cnmemDevice_t[conf_.device_size()];
    // CHECK_GT(conf_.init_size(), 0u);
    // int i = 0;
    // for (auto device : conf_.device()) {
    //  settingPtr[i].device = device;
    //  settingPtr[i].size = conf_.init_size() * kNBytesPerMB;
    //  settingPtr[i].numStreams = 0;
    //  settingPtr[i].streams = NULL;
    //  settingPtr[i].streamSizes = 0;
    //  i++;
    //}
    // auto status = cnmemInit(conf_.device_size(), settingPtr, conf_.flag());
    // CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
    //    << " " << cnmemGetErrorString(status);
    // delete[] settingPtr;
    initialized_ = true;
  }
  mtx_.unlock();
}

SwapPool::~SwapPool() {
  // mtx_.lock();
  // if (initialized_) {
  //  cnmemStatus_t status = cnmemFinalize();
  //  CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
  //      << " " << cnmemGetErrorString(status);
  //  initialized_ = false;
  //}
  // mtx_.unlock();
}

void SwapPool::Malloc(void **ptr, const size_t size) {
  // if (!initialized_) Init();
  // cnmemStatus_t status = cnmemMalloc(ptr, size, NULL);
  // CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
  //    << " " << cnmemGetErrorString(status);

  void *allocated_ptr = nullptr;
  if (pool_flag == 0) {
    cudaError_t status = cudaMalloc(ptr, size);
    CHECK_EQ(status, cudaError_t::cudaSuccess);
  } else {
    // pool_flag is 1.
    if (pool_index < iteration_length_mf) {
      if ((table_pool_meta.find(pool_index - iteration_length_mf) ==
           table_pool_meta.end()) ||
          (!(size ==
             table_pool_meta.find(pool_index - iteration_length_mf)
                 ->second.size))) {
        // Not in table of negative r_idx.
        cudaError_t status = cudaMalloc(ptr, size);
        CHECK_EQ(status, cudaError_t::cudaSuccess);
      } else {
        // In the table of negative r_idx.
        auto tmp_meta =
            table_pool_meta.find(pool_index - iteration_length_mf)->second;
        allocated_ptr = tmp_meta.ptr;
        *ptr = allocated_ptr;
        table_ptr_to_ridx[allocated_ptr] = pool_index - iteration_length_mf;
      }
    } else {
      // 8 9 10th iteration.
      int r_pool_index = pool_index % iteration_length_mf;
      if ((table_pool_meta.find(r_pool_index) == table_pool_meta.end()) ||
          (!(size == table_pool_meta.find(r_pool_index)->second.size))) {
        // Not here, should be abnormal.
        cudaError_t status = cudaMalloc(ptr, size);
        CHECK_EQ(status, cudaError_t::cudaSuccess);
      } else {
        // In the table.
        auto tmp_meta = table_pool_meta.find(r_pool_index)->second;
        allocated_ptr = tmp_meta.ptr;
        *ptr = allocated_ptr;
        table_ptr_to_ridx[allocated_ptr] = r_pool_index;
      }
    }
  }
  ++pool_index;
}

void SwapPool::Free(void *ptr) {
  // CHECK(initialized_)
  //    << "Cannot free the memory as the pool is not initialzied";
  // cnmemStatus_t status = cnmemFree(ptr, NULL);
  // CHECK_EQ(status, cnmemStatus_t::CNMEM_STATUS_SUCCESS)
  //    << " " << cnmemGetErrorString(status);
  if (pool_flag == 0) {
    cudaError_t status = cudaFree(ptr);
    // JSON LEE: trace cudaFree failure status
    if (status == cudaErrorInvalidValue) {
      std::cout << __FILE__ << ":" << __LINE__ << " " << __func__ << " "
                << cudaGetErrorName(status) << '\n';
    }
    CHECK_EQ(status, cudaError_t::cudaSuccess);
  } else {
    // pool_flag is 1.
    if (table_ptr_to_ridx.find(ptr) == table_ptr_to_ridx.end()) {
      cudaError_t status = cudaFree(ptr);
      CHECK_EQ(status, cudaError_t::cudaSuccess);
    }
  }
}

/// We use PoolOpt to construct swap memory pool.
void SwapPool::PoolOpt(std::vector<std::string> &vec_mf) {
  std::vector<PoolOptInfo> vec_pool_opt_info;
  // Input vec_mf is of 3 iteration.
  iteration_length_mf = vec_mf.size() / 3;

  // Convert the raw opt info into struct PoolOptInfo.
  for (int i = 0; i < vec_mf.size(); i++) {
    std::vector<std::string> v = SplitString(vec_mf[i], " ");
    if (v[0] == "Malloc") {
      size_t result;
      std::stringstream cvt(v[2]);
      if (!(cvt >> result)) {
        result = -1;
        std::cout << "ERROR for convert size from str to int." << std::endl;
      }
      PoolOptInfo tmp_msg(v[1], result, 1, i - iteration_length_mf);
      vec_pool_opt_info.push_back(tmp_msg);
    } else if (v[0] == "Free") {
      PoolOptInfo tmp_msg(v[1], -1, -1, i - iteration_length_mf);
      vec_pool_opt_info.push_back(tmp_msg);
    } else {
      std::cout << "error for process the onePriceMsg." << std::endl;
    }
  }
  // Sort by ptr and then idx.
  std::sort(vec_pool_opt_info.begin(), vec_pool_opt_info.end(),
            sort_by_ptr_idx_ascending());

  // Convert into block lifetime.
  std::vector<PoolBlockLifeTime> vec_block_life_time;
  int i = 0;
  while (i < (vec_pool_opt_info.size() - 1)) {
    if (vec_pool_opt_info[i].mem_op_type == -1) {
      // If start with free. do nothing.
      i += 1;
    } else {
      if ((vec_pool_opt_info[i].mem_op_type == 1) &&
          (vec_pool_opt_info[i + 1].mem_op_type == -1) &&
          ((vec_pool_opt_info[i].ptr == vec_pool_opt_info[i + 1].ptr))) {
        // If start with Malloc, next item same ptr and is free.
        if ((vec_pool_opt_info[i].idx >= 0 &&
             vec_pool_opt_info[i].idx < iteration_length_mf) ||
            (vec_pool_opt_info[i + 1].idx >= 0 &&
             vec_pool_opt_info[i + 1].idx < iteration_length_mf)) {
          // If at least one of the index in range [0,iteration_length_mf]
          PoolBlockLifeTime tmp_block_life_time(
              vec_pool_opt_info[i].idx, vec_pool_opt_info[i].size,
              vec_pool_opt_info[i].idx, vec_pool_opt_info[i + 1].idx);
          vec_block_life_time.push_back(tmp_block_life_time);
        }
        // No matter in the middle iteration or not, plus 2.
        i += 2;
      } else {
        // If not one pair, Malloc-only block, no free.
        i += 1;
      }
    }
  }
  std::sort(vec_block_life_time.begin(), vec_block_life_time.end(),
            sort_by_size_r_idx_descending());

  /// Get E, V of the blocks and coloring.
  // Get V.
  int m = static_cast<int>(vec_block_life_time.size());
  std::vector<Vertex> vertices;
  for (int i = 0; i < m; i++) {
    Vertex tmp_vertex(vec_block_life_time[i].name, vec_block_life_time[i].size,
                      vec_block_life_time[i].r_idx,
                      vec_block_life_time[i].d_idx);
    vertices.push_back(tmp_vertex);
  }

  // E and coloring
  int offset = 0;
  int **adj;
  adj = new int *[m];

  // Build edges with values 1 and 0; combine with mergeSeg and
  // FirstFitAllocation in the loop.
  for (int i = 0; i < m; i++) {
    adj[i] = new int[m];
    for (int j = 0; j < m; j++) {
      if ((std::max(vertices[i].r, vertices[j].r)) <
              (std::min(vertices[i].d, vertices[j].d)) ||
          (std::min(vertices[i].d, vertices[j].d) < 0 &&
           std::min(vertices[i].d, vertices[j].d) + 2 * iteration_length_mf <
               std::max(vertices[i].r, vertices[j].r))) {
        adj[i][j] = 1;
        if (vertices[j].color_range.second) {
          // Since second never be 0, if not empty.
          vertices[i].vec_color_preoccupied.push_back(vertices[j].color_range);
        }
      } else {
        adj[i][j] = 0;
      }
    }

    std::vector<std::pair<size_t, size_t>> vec_color_merged =
        MergeColoredSegments(vertices[i].vec_color_preoccupied);

    // vertices[i].color_range =
    // FirstFitAllocation(vec_color_merged,vertices[i].size, local_offset);
    vertices[i].color_range =
        BestFitAllocation(vec_color_merged, vertices[i].size, offset);

    // Update of offset, largest memory footprint as well.
    if (vertices[i].color_range.second >= offset) {
      offset = vertices[i].color_range.second + 1;
    }
  }  // end of for.

  // Release adj, the edges.
  for (int i = 0; i < m; i++) {
    delete[] adj[i];
  }
  delete[] adj;

  // Malloc pool.
  // FIXME: check status of cudaMalloc. Here offset is the pool size or memory
  // footprint.
  cudaMalloc(&ptr_pool, offset);

  // Make table to record necessary info.
  for (int i = 0; i < vertices.size(); i++) {
    PoolBlockMeta item;
    item.r_idx = vertices[i].r;
    item.d_idx = vertices[i].d;
    item.size = vertices[i].size;
    item.offset = vertices[i].color_range.first;
    item.ptr = (void *)((char *)ptr_pool + item.offset * sizeof(char));
    item.occupied = 0;
    table_pool_meta[vertices[i].r] = item;
  }
  pool_flag = 1;
}

void SwapPool::Append(std::string blockInfo) {}

}  // namespace singa
#endif

#endif
