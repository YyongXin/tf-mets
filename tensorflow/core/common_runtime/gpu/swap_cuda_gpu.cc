/**
 * Cuda gpu device with variable swap in / out supporting.
 */
#include "singa/singa_config.h"
#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "singa/core/device.h"
#include "singa/utils/cuda_utils.h"

namespace singa {

const cudaMemcpyKind copyKind[] = {cudaMemcpyHostToHost, cudaMemcpyHostToDevice,
                                   cudaMemcpyDeviceToHost,
                                   cudaMemcpyDeviceToDevice};

// ===========================================================================
// Helper Utilities.

/// Sort DeviceOptInfo by ptr and then idx.
struct sort_by_ptr_idx_ascending {
  inline bool operator()(const DeviceOptInfo& info1,
                         const DeviceOptInfo& info2) {
    return ((info1.ptr < info2.ptr) ||
            ((info1.ptr == info2.ptr) && (info1.idx < info2.idx)));
  }
};

/// Sort DeviceOptInfo by idx.
struct sort_by_idx_ascending {
  inline bool operator()(const DeviceOptInfo& info1,
                         const DeviceOptInfo& info2) {
    return (info1.idx < info2.idx);
  }
};

/// SwapCudaGPU device info: DeviceOptInfo is defined in
/// include/singa/core/device.h.
/// Format of DeviceOptInfo [ptr, size/-1, mem_op_type, idx, time_stamp].
/// Simplified device info: [idx, mem_op_type, size_delta].
struct DeviceOptSimplifiedInfo {
  // If mem_op_type is Malloc, size_delta is size, else delta to the last index.
  size_t size_delta;
  int mem_op_type;
  int idx;
  DeviceOptSimplifiedInfo(size_t size, int opt, int i)
      : size_delta(size), mem_op_type(opt), idx(i) {}
};

/// Sort DeviceOptSimplifiedInfo by idx.
struct sort_by_DeviceOptSimplifiedInfo_idx_ascending {
  inline bool operator()(const DeviceOptSimplifiedInfo& info1,
                         const DeviceOptSimplifiedInfo& info2) {
    return (info1.idx < info2.idx);
  }
};

/// Sort SwapBlock by DOA_origin, descending.
struct sort_by_DOA_origin_descending {
  inline bool operator()(const SwapBlock& blk1, const SwapBlock& blk2) {
    return (blk1.DOA_origin > blk2.DOA_origin);
  }
};

/// Sort SwapBlock by weighted DOA_origin, descending.
struct sort_by_WDOA_descending {
  inline bool operator()(const SwapBlock& blk1, const SwapBlock& blk2) {
    return (blk1.WDOA > blk2.WDOA);
  }
};

/// Sort SwapBlock by AOA, descending.
struct sort_by_AOA_descending {
  inline bool operator()(const SwapBlock& blk1, const SwapBlock& blk2) {
    return (blk1.AOA > blk2.AOA);
  }
};

/// Sort DeviceOptInfo_Swap by idx.
struct sort_by_idx_ascending_swap {
  inline bool operator()(const SwapBlock& blk1, const SwapBlock& blk2) {
    return (blk1.r_idx < blk2.r_idx);
  }
};

/// Sort DeviceOptInfo_Swap by idx. reverse.
struct sort_by_idx_descending_swap {
  inline bool operator()(const SwapBlock& blk1, const SwapBlock& blk2) {
    return (blk1.d_idx > blk2.d_idx);
  }
};

/// Sort majority voting, ascending.
struct sort_by_majority_voting_ascending {
  inline bool operator()(const SwapBlock& blk1, const SwapBlock& blk2) {
    return (blk1.majority_voting < blk2.majority_voting);
  }
};

/// String delimiter.
/// Example: given input str: "Malloc 1000 Free 500", delimiter: " ",
/// return ["Malloc", "1000", "Free", "500"].
std::vector<std::string> SplitOptString(std::string str,
                                        std::string delimiter) {
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

/// TODO: Wrap these utility functions into common.h.
/// Convert vector of string into vector of DeviceOptInfo, sorted by ptr
/// and then idx, and update idx_range to pieceMsgVec size.
/// Format of DeviceOptInfo: [ptr, size/-1, flag, idx, time_stamp], where
/// flag values: <1, Malloc>; <-1, Free>; <2, Read>; <3, Layer>; <4, Mutable>.
std::vector<DeviceOptInfo> DeviceOptSeqStrToStruct(std::vector<std::string> vec,
                                                   int& idx_range) {
  std::vector<DeviceOptInfo> vec_opt_info;
  for (int i = 0; i < vec.size(); i++) {
    std::vector<std::string> v = SplitOptString(vec[i], " ");
    int op_type;
    if (v[0] == "Malloc") {
      op_type = 1;
    } else if (v[0] == "Free") {
      op_type = -1;
    } else if (v[0] == "Read") {
      op_type = 2;
    } else if (v[0] == "Layer") {
      op_type = 3;
    } else if (v[0] == "Mutable") {
      op_type = 4;
    }
    size_t result;
    std::stringstream cvt(v[2]);
    if (!(cvt >> result)) {
      result = -1;
      std::cout << "ERROR for convert size from str to int." << std::endl;
    }
    // ptr, size, mem_op_type, idx.
    DeviceOptInfo item(v[1], result, op_type, i);
    double tmp_time;
    std::stringstream cvt2(v[3]);
    cvt2 >> tmp_time;
    item.time_stamp = tmp_time;
    vec_opt_info.push_back(item);
  }
  std::sort(vec_opt_info.begin(), vec_opt_info.end(),
            sort_by_ptr_idx_ascending());
  idx_range = static_cast<int>(vec_opt_info.size());
  return vec_opt_info;
}

/// Pre-process device operation sequence struct info for repeatable test,
/// return a vector of int for fast detection.
std::vector<size_t> DeviceOptSeqRepeatableTestPreProcess(
    std::vector<DeviceOptInfo> vec_opt_info) {
  // For DeviceOptSimplifiedInfo, if mem_op_type is Malloc, size_delta is
  // size, else delta to last index.
  std::vector<DeviceOptSimplifiedInfo> vec_opt_simplified_info;
  std::string tmp_str;
  int tmp_idx = 0;
  for (int i = 0; i < vec_opt_info.size(); i++) {
    if (vec_opt_info[i].mem_op_type == 1) {
      // Update tmp_str and idx.
      tmp_str = vec_opt_info[i].ptr;
      tmp_idx = vec_opt_info[i].idx;
      DeviceOptSimplifiedInfo item(vec_opt_info[i].size, 1,
                                   vec_opt_info[i].idx);
      vec_opt_simplified_info.push_back(item);
    } else {
      DeviceOptSimplifiedInfo item(vec_opt_info[i].idx - tmp_idx,
                                   vec_opt_info[i].mem_op_type,
                                   vec_opt_info[i].idx);
      tmp_idx = vec_opt_info[i].idx;
      vec_opt_simplified_info.push_back(item);
    }
  }

  std::sort(vec_opt_simplified_info.begin(), vec_opt_simplified_info.end(),
            sort_by_DeviceOptSimplifiedInfo_idx_ascending());
  // Only after sort then can create vec_rep.
  // Vector of size_delta.
  std::vector<size_t> vec_rep;
  for (int i = 0; i < vec_opt_simplified_info.size(); i++) {
    vec_rep.push_back(vec_opt_simplified_info[i].size_delta);
  }
  return vec_rep;
}

/// Repeatable test, input vector of int,
/// in-place update max_legth (length of iteration)
/// and location_of_2nd_iteration (where 2nd iteration starts).
void RepeatableTest(std::vector<size_t> rep, int& iteration_length,
                    int& location_of_2nd_iteration,
                    int iteration_length_threshold, int global_index) {
  int idx_range = (int)rep.size();
  int threshold = iteration_length_threshold;
  std::vector<std::pair<int, int>> iteration_length_location_of_2nd_iteration;

  for (int i = 0; i < idx_range; i++) {
    if (iteration_length > threshold) {
      break;
    }
    for (int len = 1; len < (idx_range - i); len++) {
      if (iteration_length > threshold) {
        break;
      }
      if ((std::equal(rep.begin() + i, rep.begin() + i - 1 + len,
                      rep.begin() + i + len)) &&
          (iteration_length < len)) {
        iteration_length = len;
        location_of_2nd_iteration = i;
        iteration_length_location_of_2nd_iteration.push_back(
            std::make_pair(iteration_length, location_of_2nd_iteration));
      }
    }
  }
}

/// Linux PCIe: <https://www.cnblogs.com/lsgxeva/p/9542975.html>
/// Host motherboard: MAXIMUS VI EXTREME.
///
/// Measured by cuda-samples/1_Utilities/bandwidthTest
/// Device 0: TITAN Xp COLLECTORS EDITION
///
/// Host to Device Bandwidth, 1 Device(s)
/// PINNED Memory Transfers
///   Transfer Size (Bytes)	Bandwidth(GB/s)
///      32000000              6.3
///
/// Device to Host Bandwidth, 1 Device(s)
/// PINNED Memory Transfers
///   Transfer Size (Bytes)	Bandwidth(GB/s)
///      32000000              6.4
///
/// Device to Device Bandwidth, 1 Device(s)
/// PINNED Memory Transfers
///   Transfer Size (Bytes)	Bandwidth(GB/s)
///      32000000              422.7
///
/// TODO: How to measure swap in / out time?
/// Pinned PCIe bus and gpu memory bandwidth and get the linear function?
/// 20 round result:
/// ===---------------- swap in ----------------===
/// k: [[0.15742704]]
/// b: [[12352.18468599]]
/// f(1): [[12352.34211303]]
/// ===---------------- swap out ----------------===
/// k: [[0.156329]]
/// b: [[8242.79951244]]
/// f(1): [[8242.95584144]]
int SwapOutTime(size_t size) {
  int ans = 0;
  if (size == 0) {
    ans = 8242;
  } else {
    ans = 0.1563 * size + 8242;
  }
  return ans;
}

int SwapInTime(size_t size) {
  int ans = 0;
  if (size == 0) {
    ans = 12352;
  } else {
    ans = 0.1574 * size + 12352;
  }
  return ans;
}

/// Get operation index (range) that above the load limit.
/// Input: vec_load, mem_limit, range [start_idx, end_idx),
/// return range overlimit [first_over_limit, first_below_limit).
std::pair<int, int> GetOptIdxAboveLoadLimit(std::vector<double> vec_load,
                                            size_t mem_limit, int start_idx,
                                            int end_idx, int iteration_length) {
  int first_over_limit = start_idx;
  int first_below_limit = end_idx;
  for (int i = start_idx + iteration_length; i < end_idx + iteration_length;
       i++) {
    if (vec_load[i] > mem_limit) {
      first_over_limit = i - iteration_length;
      break;
    }
  }
  for (int i = end_idx + iteration_length;
       i > first_over_limit + iteration_length; i--) {
    if (vec_load[i] > mem_limit) {
      first_below_limit = i - 1 - iteration_length;
      break;
    }
  }
  if (first_over_limit == start_idx) first_over_limit = -1;
  if (first_below_limit == end_idx) first_below_limit = -1;
  return std::make_pair(first_over_limit, first_below_limit);
}

/// Return memory load value and index of load peak.
std::pair<double, int> GetLoadPeak(std::vector<double> vec_load_test,
                                   int iteration_length) {
  double max_load_test = 0;
  int max_idx_test = 0;
  for (int i = iteration_length; i < iteration_length * 2; i++) {
    if (max_load_test < vec_load_test[i]) {
      max_load_test = vec_load_test[i];
      max_idx_test = i - iteration_length;
    }
  }
  return std::make_pair(max_load_test, max_idx_test);
}

/// Update load [start_idx, end_idx) by plus_minus*size.
/// Here plus_minus maybe -1.
void UpdateLoad(std::vector<double>& vec_load, int start_idx, int end_idx,
                int plus_minus, size_t size, int iteration_length) {
  for (int i = start_idx + iteration_length; i < end_idx + iteration_length;
       i++) {
    vec_load[i] = vec_load[i] + static_cast<double>(size) * plus_minus;
  }
}

/// Select swapping blocks based on a cetain priority score or BO score with
/// load updated. After get the score info, we can select block one by one till
/// updated peak load is no larger than limit.
/// Return the candidate swapping memory blocks.
std::vector<SwapBlock> SwapCudaGPU::SelectBlock(std::vector<SwapBlock> vec_swap,
                                                std::vector<double> tmp_load,
                                                double mem_limit,
                                                std::string mode) {
  std::vector<SwapBlock> vec_swap_select;
  if (mode == "DOA_origin") {
    std::sort(vec_swap.begin(), vec_swap.end(),
              sort_by_DOA_origin_descending());
  }

  if (mode == "AOA") {
    std::sort(vec_swap.begin(), vec_swap.end(), sort_by_AOA_descending());
  }

  if (mode == "WDOA") {
    for (int i = 0; i < vec_swap.size(); i++) {
      auto item = vec_swap[i];
      for (int j = item.r_idx; j < item.d_idx; j++) {
        item.WDOA += origin_load[i + iteration_length] - mem_limit;
      }
    }
    std::sort(vec_swap.begin(), vec_swap.end(), sort_by_WDOA_descending());
  }

  if (mode == "majority_voting") {
    // Add order for DOA.
    std::sort(vec_swap.begin(), vec_swap.end(),
              sort_by_DOA_origin_descending());
    for (int i = 0; i < vec_swap.size(); i++) {
      vec_swap[i].majority_voting += i;
    }
    // Add order for AOA.
    std::sort(vec_swap.begin(), vec_swap.end(), sort_by_AOA_descending());
    for (int i = 0; i < vec_swap.size(); i++) {
      vec_swap[i].majority_voting += i;
    }
    // Add order for WDOA.
    for (int i = 0; i < vec_swap.size(); i++) {
      auto item = vec_swap[i];
      for (int j = item.r_idx; j < item.d_idx; j++) {
        item.WDOA += origin_load[i + iteration_length] - mem_limit;
      }
    }
    std::sort(vec_swap.begin(), vec_swap.end(), sort_by_WDOA_descending());
    for (int i = 0; i < vec_swap.size(); i++) {
      vec_swap[i].majority_voting += i;
    }
    std::sort(vec_swap.begin(), vec_swap.end(),
              sort_by_majority_voting_ascending());
  }
  // Select block one by one till updated peak load is no larger than limit.
  for (int i = 0; i < vec_swap.size(); i++) {
    UpdateLoad(tmp_load, vec_swap[i].r_idx_ready, vec_swap[i].d_idx, -1,
               vec_swap[i].size, iteration_length);
    vec_swap_select.push_back(vec_swap[i]);
    auto tmp_over_limit_ = GetOptIdxAboveLoadLimit(
        tmp_load, mem_limit, 0, iteration_length, iteration_length);
    auto max_current = GetLoadPeak(tmp_load, iteration_length);
    auto newmax_load = max_current.first;
    if (newmax_load < mem_limit) {
      break;
    }
  }
  return vec_swap_select;
}

/// Get ideal load, which is equivalent to load by synchronous swapping.
std::vector<double> SwapCudaGPU::GetIdealLoad(
    std::vector<double> vec_load, std::vector<SwapBlock> vec_swap_select) {
  auto vec_load_return = vec_load;
  for (int i = 0; i < vec_swap_select.size(); i++) {
    int auto_buffer = 0;
    auto item = vec_swap_select[i];
    if (item.cat == "A2") auto_buffer = data_buffer;
    if (item.cat == "A3") auto_buffer = mutable_data_buffer;
    UpdateLoad(vec_load_return, item.r_idx + auto_buffer, item.d_idx, -1,
               item.size, iteration_length);
  }
  return vec_load_return;
}

/// Swap scheduling algorothm.
/// vec_swap_select contains the swapping blocks.
/// Update idx_out_end, idx_in_start,
/// compute overhead time.
/// Two mode selection: stick-to-limit or no-overhead.
void SwapCudaGPU::Scheduling(std::vector<SwapBlock>& vec_swap_select,
                             std::vector<double>& vec_load_tmp,
                             double& overhead, double mem_limit,
                             std::string mode) {
  overhead = 0;
  /// Stick to memory load limit mode.
  if (mode == "stick-to-limit") {
    std::sort(vec_swap_select.begin(), vec_swap_select.end(),
              sort_by_idx_ascending_swap());
    for (int i = 0; i < vec_swap_select.size(); i++) {
      auto item = vec_swap_select[i];
      int ready_idx = item.r_idx_ready;
      if (i > 0) {
        ready_idx = std::max(ready_idx, vec_swap_select[i - 1].idx_out_end);
      }
      item.idx_out_start = ready_idx;
      item.t_out_start = vec_run[ready_idx + iteration_length].time_stamp;
      item.t_out_end = item.t_out_start + SwapOutTime(item.size);
      total_swap_out_time += SwapOutTime(item.size);
      while (item.t_out_end >
             vec_run[ready_idx + iteration_length].time_stamp) {
        // Here ready means when able to finish the swap-out, with or without
        // overhead.
        ready_idx++;
      }
      // Get min compare with max_idx and ready_idx.
      ready_idx = std::min(max_idx, ready_idx);
      UpdateLoad(vec_load_tmp, ready_idx + 1, item.d_idx, -1, item.size,
                 iteration_length);
      // tmp_over_limit_ is the operation index range that above the load limit.
      auto tmp_over_limit_ = GetOptIdxAboveLoadLimit(
          vec_load_tmp, mem_limit, 0, iteration_length, iteration_length);
      if ((tmp_over_limit_.first != -1) &&
          (tmp_over_limit_.first <= ready_idx)) {
        UpdateLoad(vec_load_tmp, tmp_over_limit_.first - 1, ready_idx + 1, -1,
                   item.size, iteration_length);
        ready_idx = tmp_over_limit_.first - 1;
        overhead +=
            (item.t_out_end - vec_run[ready_idx + iteration_length].time_stamp);
      }
      item.idx_out_end = ready_idx;
      vec_swap_select[i] = item;
    }
    std::sort(vec_swap_select.begin(), vec_swap_select.end(),
              sort_by_idx_descending_swap());
    for (int i = 0; i < vec_swap_select.size(); i++) {
      auto item = vec_swap_select[i];
      int need_idx = item.d_idx;
      if (i > 0) {
        need_idx = std::min(need_idx, vec_swap_select[i - 1].idx_in_start);
      }
      item.idx_in_end = need_idx;
      double prepare_time = vec_run[need_idx + iteration_length].time_stamp -
                            SwapInTime(item.size);
      total_swap_in_time += SwapInTime(item.size);
      while (prepare_time < vec_run[need_idx + iteration_length].time_stamp) {
        need_idx--;
      }
      need_idx = std::max(need_idx, max_idx + 1);
      item.idx_in_start = need_idx;
      item.t_in_start = prepare_time;
      UpdateLoad(vec_load_tmp, item.idx_in_start, item.d_idx, 1, item.size,
                 iteration_length);
      auto tmp_over_limit_3 = GetOptIdxAboveLoadLimit(
          vec_load_tmp, mem_limit, 0, iteration_length, iteration_length);

      if ((tmp_over_limit_3.second != -1) &&
          (vec_run[tmp_over_limit_3.second + iteration_length].time_stamp >
           item.t_in_start)) {
        overhead +=
            (vec_run[tmp_over_limit_3.second + iteration_length].time_stamp -
             item.t_in_start);
        UpdateLoad(vec_load_tmp, item.idx_in_start, tmp_over_limit_3.second + 1,
                   -1, item.size, iteration_length);
        item.idx_in_start = tmp_over_limit_3.second + 1;
        auto tmp_over_limit_4 = GetOptIdxAboveLoadLimit(
            vec_load_tmp, mem_limit, 0, iteration_length, iteration_length);
      }
      vec_swap_select[i] = item;
    }
  }

  /// Zero-overhead mode.
  if (mode == "no-overhead") {
    // Update idx_out_end.
    // Sort by r_idx for idx_out_end updating.
    std::sort(vec_swap_select.begin(), vec_swap_select.end(),
              sort_by_idx_ascending_swap());
    for (int i = 0; i < vec_swap_select.size(); i++) {
      auto item = vec_swap_select[i];
      int ready_idx = 0;
      if (item.cat == "A1") {
        ready_idx = item.r_idx;
      }
      if (item.cat == "A2") {
        ready_idx = item.r_idx + data_buffer;
      }
      if (item.cat == "A3") {
        ready_idx = item.r_idx + mutable_data_buffer;
      }
      if (i > 0) {
        ready_idx = std::max(ready_idx, vec_swap_select[i - 1].idx_out_end);
      }
      item.idx_out_start = ready_idx;
      item.t_out_start = vec_run[ready_idx].time_stamp;
      item.t_out_end = item.t_out_start + SwapOutTime(item.size);
      while (item.t_out_end > vec_run[ready_idx].time_stamp) {
        ready_idx++;
      }
      item.idx_out_end = ready_idx;
      vec_swap_select[i] = item;
    }
    // Update idx_in_start.
    std::sort(vec_swap_select.begin(), vec_swap_select.end(),
              sort_by_idx_descending_swap());
    for (int i = 0; i < vec_swap_select.size(); i++) {
      auto item = vec_swap_select[i];
      int need_idx = item.d_idx;
      if (i > 0) {
        need_idx = std::min(need_idx, vec_swap_select[i - 1].idx_in_start);
      }
      item.idx_in_end = need_idx;
      double prepare_time =
          vec_run[need_idx].time_stamp - SwapInTime(item.size);
      while (prepare_time < vec_run[need_idx].time_stamp) {
        --need_idx;
      }
      item.idx_in_start = need_idx;
      item.t_in_start = prepare_time;
      vec_swap_select[i] = item;
      UpdateLoad(vec_load_tmp, item.idx_out_end, item.idx_in_start + 1, -1,
                 item.size, iteration_length);
    }
  }
}

/// Construct table_sched and table_meta.
void SwapCudaGPU::BuildMetaTables(std::vector<SwapBlock> vec_swap_select) {
  cudaStream_t cu_strm1;
  cudaStream_t cu_strm2;
  std::sort(vec_swap_select.begin(), vec_swap_select.end(),
            sort_by_idx_ascending_swap());
  // For each swap select, make table_sched and table_meta.
  // table_sched value is <swap_idx, swap_dir, sync_idx, sync_dir>
  // for (int i = static_cast<int>(vec_swap_select.size() - 1); i>=0; i--) {
  for (int i = 0; i < vec_swap_select.size(); i++) {
    auto item = vec_swap_select[i];
    if (table_sched.find(item.idx_out_start) == table_sched.end()) {
      // swap_dir 0 is swap-out.
      table_sched[item.idx_out_start] = std::make_tuple(item.r_idx, 0, -1, -1);
    } else {
      std::get<0>(table_sched.find(item.idx_out_start)->second) = item.r_idx;
      std::get<1>(table_sched.find(item.idx_out_start)->second) = 0;
    }
    // idx_in_start swap.
    if (table_sched.find(item.idx_in_start) == table_sched.end()) {
      // swap_dir 1 is swap-in.
      table_sched[item.idx_in_start] = std::make_tuple(item.r_idx, 1, -1, -1);
    } else {
      std::get<0>(table_sched.find(item.idx_in_start)->second) = item.r_idx;
      std::get<1>(table_sched.find(item.idx_in_start)->second) = 1;
    }
    // idx_out_end sync.
    if (table_sched.find(item.idx_out_end) == table_sched.end()) {
      table_sched[item.idx_out_end] = std::make_tuple(-1, -1, item.r_idx, 0);
    } else {
      // Update sync_dir. 0 is sync swap-out.
      std::get<2>(table_sched.find(item.idx_out_end)->second) = item.r_idx;
      std::get<3>(table_sched.find(item.idx_out_end)->second) = 0;
    }
    // i2 sync
    if (table_sched.find(item.idx_in_end) == table_sched.end()) {
      table_sched[item.idx_in_end] = std::make_tuple(-1, -1, item.r_idx, 1);
    } else {
      // Update sync_dir. 1 is sync swap-in.
      std::get<2>(table_sched.find(item.idx_in_end)->second) = item.r_idx;
      std::get<3>(table_sched.find(item.idx_in_end)->second) = 1;
    }

    /// Make table_meta.
    void* tmp_ptr = nullptr;
    // Malloc host cpu memory, pinned memory.
    cudaMallocHost(&tmp_ptr, item.size);
    BlockMeta meta;
    meta.size = item.size;
    meta.cpu_ptr = tmp_ptr;
    meta.out_stream = cu_strm1;
    meta.in_stream = cu_strm2;
    table_meta[item.r_idx] = meta;
  }
}

/// Update table_meta's block_ and data_;
/// Update once atfer swap test is passed.
/// Enable to update negative r_idx.
/// It's safe in below procedure, as r_global_index and relative_counter should
/// never be the same.
void SwapCudaGPU::UpdateMetaTables(Block* block_ptr) {
  if (past_test_flag == 1) {
    // Update positive r_idx.
    // location_of_2nd_iteration is the index of start of 2nd iteration.
    int r_global_index =
        (global_index - location_of_2nd_iteration) % iteration_length;
    if (!(table_meta.find(r_global_index) == table_meta.end())) {
      table_meta.find(r_global_index)->second.block_ = block_ptr;
      table_meta.find(r_global_index)->second.data_ = block_ptr->get_data();
    }

    // Update negative r_idx.
    int relative_counter = r_global_index - iteration_length;
    if (!(table_meta.find(relative_counter) == table_meta.end())) {
      table_meta.find(relative_counter)->second.block_ = block_ptr;
      table_meta.find(relative_counter)->second.data_ = block_ptr->get_data();
    }
  }
}

/// Test repeatability, detect iteration, and return global_index_threshold.
int SwapCudaGPU::Detection(std::vector<std::string> vec_block,
                           int& iteration_length,
                           int& location_of_2nd_iteration) {
  /// vec_str (vec_block) to vec_opt_info, sort by ptr and idx.
  int idx_range = 0;
  std::vector<DeviceOptInfo> vec_opt_info =
      DeviceOptSeqStrToStruct(vec_block, idx_range);
  /// Repeatable test.
  std::vector<size_t> vec_rep =
      DeviceOptSeqRepeatableTestPreProcess(vec_opt_info);
  RepeatableTest(vec_rep, iteration_length, location_of_2nd_iteration,
                 iteration_length_threshold, global_index);

  // Note here location_of_2nd_iteration not exactly start of one iteration,
  // adjust to nearly start of one by restricting "Malloc".
  int shift_counter = 0;
  for (int i = 0; i < iteration_length; i++) {
    std::vector<std::string> v =
        SplitOptString(vec_block[location_of_2nd_iteration + i], " ");
    if (v[0] == "Malloc") {
      shift_counter = i;
      break;
    }
  }
  location_of_2nd_iteration = location_of_2nd_iteration + shift_counter;
  if (iteration_length < iteration_length_threshold) {
    return -1;
  }
  return global_index + iteration_length -
         (global_index - location_of_2nd_iteration) % iteration_length;
}

/// Major stream of functions: from make candidate blocks, selection swaps, make
/// tables, etc.
void SwapCudaGPU::Plan() {
  int idx_range = 0;
  // Convert DeviceOptInfo squence from string to struct.
  std::vector<DeviceOptInfo> vec_opt_info =
      DeviceOptSeqStrToStruct(vec_block, idx_range);
  std::sort(vec_opt_info.begin(), vec_opt_info.end(), sort_by_idx_ascending());
  // Scale down idx, to middle iteration.
  tmp_time_baseline = vec_opt_info[location_of_5th_iteration].time_stamp;
  for (int i = 0; i < vec_opt_info.size(); i++) {
    vec_opt_info[i].idx =
        vec_opt_info[i].idx - location_of_5th_iteration - iteration_length;
    vec_opt_info[i].time_stamp = vec_opt_info[i].time_stamp - tmp_time_baseline;
  }

  // Build op sequence and size sequence.
  std::vector<DeviceOptInfo> one_iter(
      &vec_opt_info[location_of_2nd_iteration + 4 * iteration_length],
      &vec_opt_info[location_of_2nd_iteration + 5 * iteration_length]);
  for (int i = 0; i < one_iter.size(); i++) {
    operation_sequence.push_back(one_iter[i].mem_op_type);
    size_sequence.push_back(one_iter[i].size);
  }

  // 3 iterations of vec_run and vec_load, max_idx and max_load.
  std::vector<DeviceOptInfo> tmp_vec_run(
      &vec_opt_info[location_of_2nd_iteration + 3 * iteration_length],
      &vec_opt_info[location_of_2nd_iteration + 6 * iteration_length]);
  vec_run = tmp_vec_run;
  std::vector<DeviceOptInfo> tmp_vec_run2(
      &vec_opt_info[location_of_2nd_iteration],
      &vec_opt_info[location_of_2nd_iteration + 3 * iteration_length]);
  auto vec_run2 = tmp_vec_run2;

  std::vector<double> vec_load(
      &global_load[location_of_2nd_iteration],
      &global_load[location_of_2nd_iteration + 3 * iteration_length]);
  // origin_load is 3 iteration load, for scheduling plan.
  origin_load = vec_load;

  auto max_current = GetLoadPeak(vec_load, iteration_length);
  max_load = max_current.first;
  max_idx = max_current.second;

  // Sort by ptr & idx, sorting the duplicate.
  auto vec_run_dup = vec_run;
  std::sort(vec_run_dup.begin(), vec_run_dup.end(),
            sort_by_ptr_idx_ascending());

  /// Formulate swappable items and calculate PS: DOA and AOA.
  std::vector<SwapBlock> vec_swap;
  for (int i = 1; i < vec_run_dup.size(); i++) {
    // SwapBlock: [ptr, size, r_idx, d_idx, r_time, d_time].
    if ((vec_run_dup[i].size >= smallest_block) &&
        (vec_run_dup[i - 1].idx < max_idx) && (vec_run_dup[i].idx > max_idx) &&
        (vec_run_dup[i - 1].ptr == vec_run_dup[i].ptr) &&
        ((vec_run_dup[i - 1].mem_op_type == 3) or
         (vec_run_dup[i - 1].mem_op_type == 2) or
         (vec_run_dup[i - 1].mem_op_type == 4))) {
      SwapBlock item(vec_run_dup[i].ptr, vec_run_dup[i].size,
                     vec_run_dup[i - 1].idx, vec_run_dup[i].idx,
                     vec_run_dup[i - 1].time_stamp, vec_run_dup[i].time_stamp);
      item.DOA_origin = item.d_time - item.r_time;
      // TODO: To simulate the gpu swap io time, system clock will be better?
      // https://stackoverflow.com/questions/16177295/get-time-since-epoch-in-milliseconds-preferably-using-c11-chrono
      item.DOA = item.d_time - item.r_time - SwapOutTime(item.size) -
                 SwapOutTime(item.size);
      if (item.DOA >= 0) {
        item.AOA = item.DOA * item.size;
      } else {
        item.AOA = item.DOA * 1 / item.size;
      }
      // Categories.
      if (vec_run_dup[i - 1].mem_op_type == 3) {
        item.cat = "A1";
        item.r_idx_ready = item.r_idx;
      }
      if (vec_run_dup[i - 1].mem_op_type == 2) {
        item.cat = "A2";
        item.r_idx_ready = item.r_idx + data_buffer;
      }
      if (vec_run_dup[i - 1].mem_op_type == 4) {
        item.cat = "A3";
        item.r_idx_ready = item.r_idx + mutable_data_buffer;
      }
      vec_swap.push_back(item);
    }
  }

  /// Load ideal, swap all vec_swap, least possible memory by one-swap, for data
  /// collection only.
  auto vec_load_ideal = GetIdealLoad(vec_load, vec_swap);
  std::fstream file_load_ideal("load_ideal.csv",
                               std::ios::in | std::ios::out | std::ios::app);
  for (int i = iteration_length; i < iteration_length * 2; i++) {
    file_load_ideal << vec_load_ideal[i] << std::endl;
  }

  auto max_ideal = GetLoadPeak(vec_load_ideal, iteration_length);
  size_t max_load_ideal = max_ideal.first;
  int max_idx_ideal = max_ideal.second;

  /// Majority voting, can specify mode here, can specify load_limit.
  auto tmp_load = origin_load;
  // FIXME: mem_limit_majority_voting is a tunable parm.
  // TODO: auto determine the mem_limit_majority_voting.
  // auto mem_limit_majority_voting = 550 << 20;
  int mem_limit = GetWorkloadMemoryLimit();
  auto mem_limit_majority_voting = mem_limit << 20;
  // Select swapping blocks based on the PS(priority score) or BO score with
  // load updated. vec_swap_majority_voting contains the candidate swapping
  // blocks.
  std::string blk_sel_mode = GetBlockSelectMode();
  auto vec_swap_majority_voting =
      SelectBlock(vec_swap, tmp_load, mem_limit_majority_voting, blk_sel_mode);
  // vec_swap_select_global = vec_swap_majority_voting;

  // Here origin_load is 3 iteration load, for scheduling plan.
  // std::vector<double>
  auto vec_load_WDOA = origin_load;
  // Here we set the scheduling mode to: stick to the memory load limit.
  std::string blk_sched_mode = GetBlockScheduleMode();

  double overhead_WDOA = 0;
  Scheduling(vec_swap_majority_voting, vec_load_WDOA, overhead_WDOA,
             mem_limit_majority_voting, blk_sched_mode);
  BuildMetaTables(vec_swap_majority_voting);
}

SwapCudaGPU::~SwapCudaGPU() {
  // Dump each memory block info.
  std::fstream file_blk_full("vec_block_full.csv",
                             std::ios::in | std::ios::out | std::ios::app);
  for (int i = 0; i < vec_block.size(); i++) {
    file_blk_full << vec_block[i] << std::endl;
  }
  if (ctx_.cublas_handle) CUBLAS_CHECK(cublasDestroy(ctx_.cublas_handle));
  if (ctx_.curand_generator)
    CURAND_CHECK(curandDestroyGenerator(ctx_.curand_generator));
#ifdef USE_CUDNN
  if (ctx_.cudnn_handle) {
    auto status = cudnnDestroy(ctx_.cudnn_handle);
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(status);
  }
#endif
}

const int kNumCudaStream = 1;

SwapCudaGPU::SwapCudaGPU(const std::string& blk_sel_mode,
                         const std::string& blk_sched_mode,
                         const int& mem_limit, int id)
    : Device(DT_SwapCudaGPU, id, kNumCudaStream),
      blk_select_mode(blk_sel_mode),
      blk_scheduling_mode(blk_sched_mode),
      workload_mem_limit(mem_limit) {
  MemPoolConf conf;
  conf.add_device(id);
  // Replace this pool with swap in/out support.
  pool_ = std::make_shared<SwapPool>(conf);
  Setup();
}

SwapCudaGPU::SwapCudaGPU(const std::string& blk_sel_mode,
                         const std::string& blk_sched_mode,
                         const int& mem_limit, int id,
                         std::shared_ptr<DeviceMemPool> pool)
    : Device(DT_SwapCudaGPU, id, kNumCudaStream),
      blk_select_mode(blk_sel_mode),
      blk_scheduling_mode(blk_sched_mode),
      workload_mem_limit(mem_limit) {
  CHECK(pool != nullptr);
  pool_ = pool;
  Setup();
}

void SwapCudaGPU::Setup() {
  device_type_ = DT_SwapCudaGPU;
  lang_ = kCuda;
  ctx_.stream = NULL;  // use the default sync stream

  // TODO: create one handle for each steam?
  // Preserse for future use instead of default sync stream, for concurrency
  // cudaStreamCreate(&ctx_.stream);

  CUDA_CHECK(cudaSetDevice(id_));
  // use curandCreateGeneratorHost for CudaHost device
  CURAND_CHECK(
      curandCreateGenerator(&ctx_.curand_generator, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetStream(ctx_.curand_generator, ctx_.stream));
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  SetRandSeed(seed);
  // TODO: if one generator per stream, then need diff offset per gen?
  CURAND_CHECK(curandSetGeneratorOffset(ctx_.curand_generator, 0));
  CUBLAS_CHECK(cublasCreate(&(ctx_.cublas_handle)));
  CUBLAS_CHECK(cublasSetStream(ctx_.cublas_handle, ctx_.stream));

#ifdef USE_CUDNN
  // TODO: create one handle for each stream?
  auto status = cudnnCreate(&ctx_.cudnn_handle);
  CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(status);
  cudnnSetStream(ctx_.cudnn_handle, ctx_.stream);
#endif  // USE_CUDNN
}

void SwapCudaGPU::SetRandSeed(unsigned seed) {
  CHECK(ctx_.curand_generator);
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(ctx_.curand_generator, seed));
}

void SwapCudaGPU::DoExec(function<void(Context*)>&& fn, int executor) {
  fn(&ctx_);
}

void SwapCudaGPU::CopyToFrom(void* dst, const void* src, size_t nBytes,
                             CopyDirection direction, Context* ctx) {
  cudaMemcpy(dst, src, nBytes, copyKind[direction]);
  // cudaMemcpyAsync(dst, src, nBytes, copyKind[direction], ctx_.stream);
}

size_t SwapCudaGPU::GetAllocatedMem() {
  if (pool_ != nullptr) {
    auto ret = pool_->GetMemUsage();
    return ret.second - ret.first;
  }
  LOG(ERROR) << "The memory pool is not set";
  return 0u;
}

/// Allocate gpu memory.
void* SwapCudaGPU::Malloc(int size) {
  void* ptr = nullptr;
  if (size > 0) {
    CUDA_CHECK(cudaSetDevice(id_));
    pool_->Malloc((void**)&ptr, size);

    // Append vec_block_mf, for swap & pool.
    if ((async_swap_flag == 1) &&
        ((global_index - 4 * iteration_length) <
         three_more_iteration_global_index_threshold) &&
        ((global_index - iteration_length) >=
         three_more_iteration_global_index_threshold)) {
      std::string tmp_str1 = "Malloc ";
      std::stringstream ss2;
      ss2 << ptr;
      std::string tmp_str2 = ss2.str();
      std::stringstream ss3;
      ss3 << size;
      std::string tmp_str3 = ss3.str();
      // String tmp would be like: "Malloc <ptr> <size>".
      std::string tmp = tmp_str1 + tmp_str2 + " " + tmp_str3;
      vec_block_mf.push_back(tmp);
    }
    // Record Malloc/Free semantics after swap plan done.
    if ((async_swap_flag == 1) &&
        ((global_index - 4 * iteration_length) <
         three_more_iteration_global_index_threshold)) {
      std::fstream file_mf_one_iter(
          "mf_one_iter.csv", std::ios::in | std::ios::out | std::ios::app);
      file_mf_one_iter << "Malloc " << ptr << " " << size;
      file_mf_one_iter << std::endl;
    }
    // TODO: remove the memset.
    CUDA_CHECK(cudaMemset(ptr, 0, size));
    // Comment out for future analysis: without cnmem
    // CUDA_CHECK(cudaMemsetAsync(ptr, 0, size, ctx_.stream));
  }
  return ptr;
}

/// Free gpu memory.
void SwapCudaGPU::Free(void* ptr) {
  if (ptr != nullptr) {
    CUDA_CHECK(cudaSetDevice(id_));
    pool_->Free(ptr);
    /// Append vec_block_mf, for swap & pool.
    if ((async_swap_flag == 1) &&
        ((global_index - 4 * iteration_length) <
         three_more_iteration_global_index_threshold) &&
        ((global_index - iteration_length) >=
         three_more_iteration_global_index_threshold)) {
      std::string tmp_str1 = "Free ";
      std::stringstream ss2;
      ss2 << ptr;
      std::string tmp_str2 = ss2.str();
      // String tmp would be like: "Free <ptr>".
      std::string tmp = tmp_str1 + tmp_str2;
      vec_block_mf.push_back(tmp);
    }

    if ((async_swap_flag == 1) &&
        ((global_index - 4 * iteration_length) <
         three_more_iteration_global_index_threshold)) {
      std::fstream file_mf_one_iter(
          "mf_one_iter.csv", std::ios::in | std::ios::out | std::ios::app);
      file_mf_one_iter << "Free " << ptr << std::endl;
    }
  }
}

/// Test after every index, at append order and index changed.
void SwapCudaGPU::DetectionPlan() {
  // Test iteration.
  if (((global_index + 1) % (iteration_length_threshold) == 0) &&
      (async_swap_flag == 0) && (past_test_flag == 0)) {
    global_index_threshold =
        Detection(vec_block, iteration_length, location_of_2nd_iteration);
    iteration_length_threshold =
        std::max(iteration_length_threshold, global_index / 10);
    iteration_length_threshold = std::min(2000, iteration_length_threshold);
    if (iteration_length > iteration_length_threshold) {
      past_test_flag = 1;
      three_more_iteration_global_index_threshold =
          global_index_threshold + 3 * iteration_length;
      location_of_5th_iteration =
          location_of_2nd_iteration + 3 * iteration_length;
    }
  }
  // Switch flag, next idx.
  if ((global_index + 1) == three_more_iteration_global_index_threshold) {
    // If we not reach here, we will get error.
    Plan();
    async_swap_flag = 1;
  }
}

/// Append info right after Malloc; make block_ptr - data_ptr pair-wise table.
/// Since Block* is not available till Malloc() done.
void SwapCudaGPU::AppendAfterMalloc(Block* block_ptr, void* data_ptr,
                                    int size) {
  // Append necessary instrument info.
  std::stringstream ss;
  ss << block_ptr;
  std::string tmp_str = ss.str();
  DeviceOptInfoToAppend dev_opt_info("Malloc", tmp_str, size);
  auto t = (std::chrono::system_clock::now()).time_since_epoch().count();
  dev_opt_info.time_stamp = t;
  Append(dev_opt_info);
}

/// Swap and sync as per schedule, at every index, by calling DeploySwapExec().
void SwapCudaGPU::DeploySwap() {
  int r_global_index =
      (global_index - location_of_2nd_iteration) % iteration_length;
  int r_global_index_n = r_global_index - iteration_length;
  if (async_swap_flag == 1) {
    if ((global_index <
         three_more_iteration_global_index_threshold + iteration_length) &&
        (!(table_sched.find(r_global_index_n) == table_sched.end()))) {
      DeploySwapExec(r_global_index_n);
    }
    if ((global_index >=
         three_more_iteration_global_index_threshold + iteration_length) &&
        (!(table_sched.find(r_global_index_n) == table_sched.end()))) {
      DeploySwapExec(r_global_index_n);
    }
    if ((global_index >=
         three_more_iteration_global_index_threshold + iteration_length) &&
        (!(table_sched.find(r_global_index) == table_sched.end()))) {
      DeploySwapExec(r_global_index);
    }
  }
}

// Execute DeploySwap.
void SwapCudaGPU::DeploySwapExec(int r_global_index) {
  auto swap_idx = std::get<0>(table_sched.find(r_global_index)->second);
  auto swap_dir = std::get<1>(table_sched.find(r_global_index)->second);
  auto sync_idx = std::get<2>(table_sched.find(r_global_index)->second);
  auto sync_dir = std::get<3>(table_sched.find(r_global_index)->second);
  if (swap_dir == 0) {
    SwapOut(swap_idx);
  }
  if (swap_dir == 1) {
    SwapIn(swap_idx);
  }
  if (sync_dir == 0) {
    /// Sync swap-out, actions to perform including sync, update block's data_
    /// to nullptr, free
    /// data_, update meta.
    auto last_meta = table_meta.find(sync_idx)->second;
    auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
    // FIXME: check status of cudaEventSynchronize.
    cudaEventSynchronize(last_meta.in_event);
    auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();

    // Record the r_idx of the block/meta.
    table_not_at_device[last_meta.block_] = sync_idx;

    last_meta.block_->update_data(nullptr);
    // Free this memory block from gpu memory pool.
    pool_->Free(last_meta.data_);
    /// Append vec_block_mf.
    if ((async_swap_flag == 1) &&
        ((global_index - 4 * iteration_length) <
         three_more_iteration_global_index_threshold) &&
        ((global_index - iteration_length) >=
         three_more_iteration_global_index_threshold)) {
      std::string tmp_str1 = "Free ";
      std::stringstream ss2;
      ss2 << last_meta.data_;
      std::string tmp_str2 = ss2.str();
      // String tmp would be like: "Free <data_>".
      std::string tmp = tmp_str1 + tmp_str2;
      vec_block_mf.push_back(tmp);
    }
    if ((async_swap_flag == 1) &&
        ((global_index - 4 * iteration_length) <
         three_more_iteration_global_index_threshold)) {
      std::fstream file_mf_one_iter(
          "mf_one_iter.csv", std::ios::in | std::ios::out | std::ios::app);
      file_mf_one_iter << "Free " << last_meta.data_ << " SwapOut(Sync)"
                       << std::endl;
    }
    last_meta.data_ = nullptr;
    // Update table_meta.
    table_meta.find(sync_idx)->second = last_meta;
  }
  if (sync_dir == 1) {
    /// Sync swap-in, actions to perform including sync, update block's data_ to
    /// new gpu address,
    /// update meta.
    auto last_meta = table_meta.find(sync_idx)->second;
    auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
    cudaEventSynchronize(last_meta.out_event);
    auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
    // Remove the block from table_not_at_device.
    table_not_at_device.erase(last_meta.block_);
    // Update the block's data_ ptr to new gpu address.
    last_meta.block_->update_data(last_meta.data_);
    // Update table_meta.
    table_meta.find(sync_idx)->second = last_meta;
  }
}

// DeviceOptInfoToAppend: [mem_op_type, block_ptr, size, time_stamp].
void SwapCudaGPU::Append(DeviceOptInfoToAppend dev_opt_info) {
  // Convert block_ptr from string to Block*.
  void* tmp_ptr;
  std::stringstream cvt(dev_opt_info.block_ptr);
  cvt >> tmp_ptr;
  auto block_ptr = static_cast<Block*>(tmp_ptr);
  // Update the global memory load.
  if (iteration_length < iteration_length_threshold) {
    if (dev_opt_info.mem_op_type == "Malloc") {
      if (global_load.size() > 0) {
        // Global memory load increases with the Malloc operation.
        global_load.push_back(global_load[global_load.size() - 1] +
                              block_ptr->size());
      } else {
        // If we reach here, first Malloc.
        global_load.push_back(block_ptr->size());
      }
    } else if (dev_opt_info.mem_op_type == "Free") {
      global_load.push_back(global_load[global_load.size() - 1] -
                            block_ptr->size());
    } else {
      // For other mem_op_type, e.g, Read or Mutable, the global
      // memory load maintains.
      global_load.push_back(global_load[global_load.size() - 1]);
    }
  }

  // Append into vec_block.
  std::stringstream ss1;
  ss1 << dev_opt_info.size;
  std::string tmp_str1 = ss1.str();
  std::stringstream ss4;
  ss4 << dev_opt_info.time_stamp;
  std::string tmp_str4 = ss4.str();
  // String block_info would be like: "<mem_op_type> <blk_ptr> <size>
  // <time_stamp>".
  std::string block_info = dev_opt_info.mem_op_type + " " +
                           dev_opt_info.block_ptr + " " + tmp_str1 + " " +
                           tmp_str4;
  // std::cout << "1 " << block_info << std::endl;
  vec_block.push_back(block_info);

  // Change swap flag on and off.
  // 0: sync, 1: async.
  if (async_swap_flag == 1) {
    int r_global_index =
        (global_index - location_of_2nd_iteration) % iteration_length;
    if (block_ptr->size() != size_sequence[r_global_index]) {
      async_swap_flag = 0;
      std::cout << "!!!! async_swap_flag changed back to 0" << std::endl;
    }
  }
  // Update table_meta and table_sched.
  UpdateMetaTables(block_ptr);
  // Deploy swap at every index.
  DeploySwap();
  // Test moved from start of malloc/free to end of append, only global_index+1
  // changed.
  DetectionPlan();
  // NOTE: This global_index includes read/write and AppendLayer as well, in
  // addition to malloc/free.
  global_index++;
  // Call PoolOpt to Construct Pool.
  if ((async_swap_flag == 1) && ((global_index - 4 * iteration_length) ==
                                 three_more_iteration_global_index_threshold)) {
    pool_->PoolOpt(vec_block_mf);
  }
}

/// In case that block is not at device memory, swap-in ad hoc.
/// Used in the Block class to update ptr after swap-in is done, if variable is
/// not
/// swapped back yet as expected.
void* SwapCudaGPU::UpdateGpuPtr(const Block* block_ptr) {
  auto r_idx = table_not_at_device.find(block_ptr)->second;
  cudaError_t err;
  BlockMeta meta = table_meta.find(r_idx)->second;
  cudaEventCreate(&meta.in_event);
  void* ptr = nullptr;
  // FIXME: Malloc here?
  pool_->Malloc((void**)&ptr, meta.size);
  meta.data_ = ptr;
  err = cudaMemcpyAsync(meta.data_, meta.cpu_ptr, meta.size,
                        cudaMemcpyHostToDevice, meta.in_stream);
  cudaEventRecord(meta.in_event, meta.in_stream);
  cudaEventSynchronize(meta.out_event);
  table_meta.find(r_idx)->second = meta;
  return ptr;
}

/// Memory copy asynchronously from GPU to CPU and update meta.
void SwapCudaGPU::SwapOut(const int idx) {
  cudaError_t err;
  BlockMeta meta = table_meta.find(idx)->second;
  cudaEventCreate(&meta.out_event);
  err = cudaMemcpyAsync(meta.cpu_ptr, meta.data_, meta.size,
                        cudaMemcpyDeviceToHost, meta.out_stream);
  cudaEventRecord(meta.out_event, meta.out_stream);
  table_meta.find(idx)->second = meta;
}

/// Memory copy asynchronously from CPU to GPU and update meta.
void SwapCudaGPU::SwapIn(const int idx) {
  cudaError_t err;
  BlockMeta meta = table_meta.find(idx)->second;
  cudaEventCreate(&meta.in_event);
  void* ptr = nullptr;
  pool_->Malloc((void**)&ptr, meta.size);

  /// Append vec_block_mf.
  if ((async_swap_flag == 1) && ((global_index - 4 * iteration_length) <
                                 three_more_iteration_global_index_threshold) &&
      ((global_index - iteration_length) >=
       three_more_iteration_global_index_threshold)) {
    std::string tmp_str1 = "Malloc ";
    std::stringstream ss2;
    ss2 << ptr;
    std::string tmp_str2 = ss2.str();
    std::stringstream ss3;
    ss3 << meta.size;
    std::string tmp_str3 = ss3.str();
    // String tmp would be like: "Malloc <ptr> <size>".
    std::string tmp = tmp_str1 + tmp_str2 + " " + tmp_str3;
    vec_block_mf.push_back(tmp);
  }
  if ((async_swap_flag == 1) && ((global_index - 4 * iteration_length) <
                                 three_more_iteration_global_index_threshold)) {
    std::fstream file_mf_one_iter("mf_one_iter.csv",
                                  std::ios::in | std::ios::out | std::ios::app);
    file_mf_one_iter << "Malloc " << ptr << " " << meta.size << " swapIn"
                     << std::endl;
  }

  meta.data_ = ptr;
  err = cudaMemcpyAsync(meta.data_, meta.cpu_ptr, meta.size,
                        cudaMemcpyHostToDevice, meta.in_stream);
  cudaEventRecord(meta.in_event, meta.in_stream);
  table_meta.find(idx)->second = meta;
}

/// Synchronous swap, collect speed info.
void SwapCudaGPU::SwapOutSynchronous(const Block* block_ptr) {
  if (global_index < 1000 && block_ptr->size() > 1 << 20) {
    std::fstream file_block5("speed.csv",
                             std::ios::in | std::ios::out | std::ios::app);
    BlockMeta meta;
    meta.data_ = meta.block_->get_data();
    void* tmp_ptr = nullptr;
    // Pinned memory.
    cudaMallocHost(&tmp_ptr, block_ptr->size());
    meta.cpu_ptr = tmp_ptr;
    table_block_meta[block_ptr] = meta;
    auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
    cudaError_t err;
    err = cudaMemcpy(meta.cpu_ptr, meta.data_, block_ptr->size(),
                     cudaMemcpyDeviceToHost);
    auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
    file_block5 << "Out " << block_ptr->size() << ' ' << t2 - t1 << std::endl;
  }
}

/// Synchronous swap, collect speed info.
void SwapCudaGPU::SwapInSynchronous(const Block* block_ptr) {
  if (global_index < 1000 && block_ptr->size() > 1 << 20) {
    std::fstream file_block5("speed.csv",
                             std::ios::in | std::ios::out | std::ios::app);
    BlockMeta meta = table_block_meta.find(block_ptr)->second;
    auto t1 = (std::chrono::system_clock::now()).time_since_epoch().count();
    cudaError_t err;
    err = cudaMemcpy(meta.data_, meta.cpu_ptr, block_ptr->size(),
                     cudaMemcpyHostToDevice);
    auto t2 = (std::chrono::system_clock::now()).time_since_epoch().count();
    file_block5 << "In " << block_ptr->size() << ' ' << t2 - t1 << std::endl;
  }
}

void SwapCudaGPU::Sync() {
  Exec([this](Context* ctx) { CUDA_CHECK(cudaStreamSynchronize(ctx_.stream)); },
       OpType::kSync, {}, {});
}

}  // namespace singa
#endif  // USE_CUDA
