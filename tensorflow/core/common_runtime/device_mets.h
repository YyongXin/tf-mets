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

#ifndef SINGA_CORE_DEVICE_H_
#define SINGA_CORE_DEVICE_H_

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "singa/core/common.h"
#include "singa/core/memory.h"
#include "singa/core/scheduler.h"
#include "singa/proto/core.pb.h"
#include "singa/singa_config.h"
#include "singa/utils/safe_queue.h"

#ifdef USE_CUDA
#include "singa/utils/cuda_utils.h"
#endif  // USE_CUDA

#ifdef USE_OPENCL
#include "singa/utils/opencl_utils.h"
#endif  // USE_OPENCL

using std::function;
using std::shared_ptr;
using std::string;
using std::vector;

namespace singa {

enum DeviceType {
  DT_Unknow,
  DT_CppCPU,
  DT_CudaGPU,
  DT_SwapCudaGPU,
  DT_OpenclDevice
};

/// Allocate memory and execute Tensor operations.
/// There are three types of devices distinguished by their programming
/// languages, namely cpp, cuda and opencl.
class Device {
 public:
  // Device() = default;
  virtual ~Device();
  /// Constructor with device ID, num of executors (e.g., cuda streams),
  /// max mem size to use (in MB)
  Device(int id, int num_executors);
  Device(DeviceType dt, int id, int num_executors);

  virtual void SetRandSeed(unsigned seed) = 0;

  void EnableGraph(bool enable) { graph_enabled_ = enable; }

  static void EnableLazyAlloc(bool enbale) { lazy_alloc_ = enbale; }

  void EstimateGraphNodeTime();
  void EstimateBlockSwapTime();

  /// Called by Tensor.
  Block* NewBlock(int size);

  /// Called by Tensor.
  void FreeBlock(Block* block);

  /// For SwapCudaGPU.
  void* UpdateGpuPtrInfo(const Block* block_ptr);

  /// Return the size (bytes) of memory in use
  /// TODO(wangwei) override this function for all devices.
  virtual size_t GetAllocatedMem() { return 0u; }

  /// Copy data within or across devices.
  virtual void CopyDataToFrom(Block* dst, Block* src, size_t nBytes,
                              CopyDirection direction, int dst_offset,
                              int src_offset);

  void CopyDataFromHostPtr(Block* dst, const void* src, size_t nBytes,
                           size_t dst_offset = 0);
  /// Submit the operation to the device, which may execute it right now or
  /// delay it depending on the scheduler.
  void Exec(function<void(Context*)>&& fn, OpType type,
            const vector<Block*> read_blocks, const vector<Block*> write_blocks,
            bool use_rand_generator = false);

  void RunGraph(bool serial = false);

  void ResetGraph() { graph_->Reset(); }

  // Wait for one event.
  // void WaitFor();

  /// wait for all operations submitted to this device.
  virtual void Sync();

  int id() const { return id_; }

  /// Return the programming language for this device.
  LangType lang() const { return lang_; }

  Context* context(int k) { return &ctx_; }

  bool graph_enabled() const { return graph_enabled_; }

  virtual std::shared_ptr<Device> host() const { return host_; }

  /// FIXME: Implement in SwapCudaGPU to avoid effecting other devices.
  /// Support for SwapCudaGPU device.
  virtual void* UpdateGpuPtr(const Block* block_ptr) = 0;
  virtual void Append(DeviceOptInfoToAppend dev_opt_info) = 0;

 protected:
  /// Execute one operation on one executor.
  virtual void DoExec(function<void(Context*)>&& fn, int executor) = 0;

  virtual void CopyToFrom(void* dst, const void* src, size_t nBytes,
                          CopyDirection direction, Context* ctx) = 0;

  /// Allocate device memory.
  virtual void* Malloc(int size) = 0;

  /// Free device memory.
  virtual void Free(void* ptr) = 0;

  /// Support for SwapCudaGPU device.
  virtual void AppendAfterMalloc(Block* block, void* data_ptr, int size) = 0;

 private:
  Device(){};

  // Note: class Graph is defined in file: include/singa/core/scheduler.h
 protected:
  friend class Block;
  friend class Graph;

  int id_ = 0;
  int num_executors_ = 0;
  unsigned seed_ = 0;
  bool graph_enabled_ = false;
  /// The computational graph
  Graph* graph_ = nullptr;
  /// Programming language type, could be kCpp, kCuda, kOpencl
  LangType lang_;
  /// The host device
  std::shared_ptr<Device> host_;
  // TODO(wangwei) define multiple contexts, one per executor
  Context ctx_;
  // Scheduler* scheduler_ = nullptr;
  // VirtualMemory* vm_ = nullptr;
  // SafeQueue<Operation> op_queue_;
  // SafeQueue<Operation> op_log_;

  /// Device type, could be DT_CppCPU, DT_CudaGPU, DT_SwapCudaGPU,
  /// DT_OpenclDevice.
  DeviceType device_type_;

  static bool lazy_alloc_;
};

/// a singleton CppDevice as the host for all devices.
extern std::shared_ptr<Device> defaultDevice;

/// Represent a CPU device which may have multiple threads/executors.
/// It runs cpp code.
class CppCPU : public Device {
 public:
  ~CppCPU();
  CppCPU();

  std::shared_ptr<Device> host() const override { return defaultDevice; }
  void SetRandSeed(unsigned seed) override;
  /// Ignore for CppCPU device.
  void Append(DeviceOptInfoToAppend info) override {}

 protected:
  void DoExec(function<void(Context*)>&& fn, int executor) override;

  void CopyToFrom(void* dst, const void* src, size_t nBytes,
                  CopyDirection direction, Context* ctx) override;

  /// Allocate cpu memory.
  void* Malloc(int size) override;

  /// Free cpu memory.
  void Free(void* ptr) override;

  /// Ignore for CppCPU device.
  void AppendAfterMalloc(Block* block, void* data_ptr, int size) override {}

  void* UpdateGpuPtr(const Block* block) override { return nullptr; }
};

// Implement Device using OpenCL libs.
// class OpenclDevice : public Device { };

#ifdef USE_CUDA
// Represent a Nvidia GPU which runs cuda code.
class CudaGPU : public Device {
 public:
  ~CudaGPU();
  /// Construct the device using default mem pool setting.
  CudaGPU(int id = 0);
  /// Construct the device given the physical device ID and memory pool.
  CudaGPU(int id, std::shared_ptr<DeviceMemPool> pool);

  void SetRandSeed(unsigned seed) override;
  size_t GetAllocatedMem() override;
  void Sync() override;
  /// Ignore for CudaGPU device.
  void Append(DeviceOptInfoToAppend info) override {}

 protected:
  void DoExec(function<void(Context*)>&& fn, int executor) override;

  void CopyToFrom(void* dst, const void* src, size_t nBytes,
                  CopyDirection direction, Context* ctx) override;

  /// Allocate cpu memory.
  void* Malloc(int size) override;

  /// Free cpu memory.
  void Free(void* ptr) override;

  /// Ignore for CudaGPU device.
  void AppendAfterMalloc(Block* block, void* data_ptr, int size) override {}
  void* UpdateGpuPtr(const Block* block_ptr) override;

 private:
  void Setup();

 private:
  shared_ptr<DeviceMemPool> pool_;
};

/// CudaCPU which uses cudaMallocHost to allocate pinned memory for host.

// ===========================================================================
// Helper Utilities.

/// SwapCudaGPU device info.
/// Format: [ptr, mem_op_type, idx, time_stamp]
/// ptr is the block ptr.
/// mem_op_type can be: Malloc, Free, Read, Mutable, etc.
struct DeviceOptInfo {
  std::string ptr;
  size_t size;
  int mem_op_type;
  int idx;
  double time_stamp;
  DeviceOptInfo(std::string ptr_, size_t size_, int op_type, int idx_)
      : ptr(ptr_), size(size_), mem_op_type(op_type), idx(idx_) {}
};

/// Meta info of the swapping blocks.
struct BlockMeta {
  Block* block_ = nullptr;
  void* data_ = nullptr;
  void* cpu_ptr = nullptr;
  size_t size = 0;
  cudaEvent_t out_event;
  cudaEvent_t in_event;
  // FIXME: Place stream operation in scheduler maybe more elegant?
  cudaStream_t out_stream;
  cudaStream_t in_stream;
};

/// Meta info of the candidate swapping blocks.
struct SwapBlock {
  std::string ptr;
  /// Category of the candidate blocks, read-read, write-read, etc.
  std::string cat;
  int name;
  size_t size;
  /// Index of last read/write before swap out, and first read/write after swap
  /// in.
  /// Out index.
  int r_idx;
  /// In index.
  int d_idx;
  /// Index of last read/write before swap out, and first read/write after swap
  /// in.
  /// Out time.
  double r_time;
  /// In time.
  double d_time;
  /// Duration of absence.
  double DOA;
  /// Area of absence.
  double AOA;
  /// t2-t1, DOA without taking out time spent.
  double DOA_origin;
  /// Weighted DOA.
  double WDOA = 0;
  /// TODO: Submodular WDOA.
  double SWDOA = 0;
  double majority_voting = 0;
  /// r_idx + buffer.
  int r_idx_ready;

  // below are index and time for scheduling
  int idx_out_start = 0;
  int idx_out_end = 0;
  int idx_in_end = 0;
  int idx_in_start = 0;
  double t_out_start = 0;
  double t_out_end = 0;
  double t_in_end = 0;
  double t_in_start = 0;

  SwapBlock(string p, size_t s, int idx_out_start, int idx_in_end,
            double t_out_start, double t_in_end)
      : ptr(p),
        size(s),
        r_idx(idx_out_start),
        d_idx(idx_in_end),
        r_time(t_out_start),
        d_time(t_in_end) {}
};

// Represent a Nvidia GPU which runs cuda code and supports variable swapping
// into host CPU.
class SwapCudaGPU : public Device {
 public:
  ~SwapCudaGPU();
  /// Construct the device using default mem pool setting.
  SwapCudaGPU(const std::string& blk_sel_mode,
              const std::string& blk_sched_mode, const int& mem_limit,
              int id = 0);
  /// Construct the device given the physical device ID and memory pool.
  SwapCudaGPU(const std::string& blk_sel_mode,
              const std::string& blk_sched_mode, const int& mem_limit, int id,
              std::shared_ptr<DeviceMemPool> pool);

  void SetRandSeed(unsigned seed) override;
  size_t GetAllocatedMem() override;
  void Sync() override;

  /// Append at every index: free, read, mutable
  void Append(DeviceOptInfoToAppend info) override;

  std::string GetBlockSelectMode() const { return blk_select_mode; }

  std::string GetBlockScheduleMode() const { return blk_scheduling_mode; }

  int GetWorkloadMemoryLimit() const { return workload_mem_limit; }

 protected:
  void DoExec(function<void(Context*)>&& fn, int executor) override;

  void CopyToFrom(void* dst, const void* src, size_t nBytes,
                  CopyDirection direction, Context* ctx) override;

  /// Allocate cpu memory.
  void* Malloc(int size) override;

  /// Free cpu memory.
  void Free(void* ptr) override;

  /// Append info after Malloc, since Block* is not available till Malloc()
  /// is done.
  void AppendAfterMalloc(Block* block, void* data_ptr, int size) override;

  /// Detect and plan.
  void DetectionPlan();

  /// Test iteration, return GC
  int Detection(std::vector<std::string> vec_block, int& iteration_length,
                int& location_of_2nd_iteration);

  /// Entry for the whole scheduling plan, from SelectBlock() to Scheduling(),
  /// BuildMetaTables().
  void Plan();

  /// TODO: Block selection algorithm.
  /// mode represents different selection methods.
  std::vector<SwapBlock> SelectBlock(std::vector<SwapBlock> vec_swap,
                                     std::vector<double> tmp_load,
                                     double mem_limit, std::string mode);

  /// Schedule algorithm.
  /// mode represents different selection methods.
  void Scheduling(std::vector<SwapBlock>& vec_swap_selct,
                  std::vector<double>& vec_load_tmp, double& overhead,
                  double mem_limit, string mode);

  /// Build table_sched and table_meta.
  void BuildMetaTables(std::vector<SwapBlock> vec_swap_selct);

  /// Update table_meta, during Append().
  void UpdateMetaTables(Block* block_ptr);

  /// swap / sync during Append().
  void DeploySwap();

  /// Do DelpoySwap.
  void DeploySwapExec(int relative_counter);

  /// Load profile as per synchronous swap.
  std::vector<double> GetIdealLoad(std::vector<double> vec_load,
                                   std::vector<SwapBlock> vec_swap_selct);

  /// In case gpu ptr is wrong, we need to update the gpu_ptr after swap_in ad
  /// hoc.
  void* UpdateGpuPtr(const Block* block_ptr) override;

  /// Swap synchronous, for early iterations.
  void SwapOutSynchronous(const Block* block_ptr);
  void SwapInSynchronous(const Block* block_ptr);

  /// Swap asynchronous, for middle iteraions.
  void SwapOut(const int idx);
  void SwapIn(const int idx);

 private:
  void Setup();

  /// Key is r_idx.
  std::map<int, BlockMeta> table_meta;
  /// for measure speed only.
  std::map<const Block*, BlockMeta> table_block_meta;
  /// int refers to its r_idx of the block/meta.
  std::map<const Block*, int> table_not_at_device;
  /// changed to with sync_r_idx
  /// value is <swap_idx, swap_dir, sync_idx, sync_dir>
  std::map<int, std::tuple<int, int, int, int>> table_sched;

  /// vec_block
  /// Iterations for detection, i.e. to detect iterations.
  std::vector<string> vec_block;
  /// Iterations that are used for planning.
  std::vector<string> vec_block_fresh;
  /// Iterations used to construct pool.
  std::vector<string> vec_block_mf;
  /// Global memory load from begining.
  std::vector<double> global_load;
  /// 3 iteration load, for scheduling plan.
  std::vector<double> origin_load;
  std::vector<DeviceOptInfo> vec_run;
  /// sequence of operations of one middle iteration.
  std::vector<int> operation_sequence;
  /// size of all operations of one middle iteration.
  std::vector<size_t> size_sequence;

  /// 0: sync, 1: async.
  int async_swap_flag = 0;
  /// 0: need to test, 1: no need test anymore.
  int past_test_flag = 0;
  /// Global counter index, increase 1 after each Malloc / Free / read / write.
  /// We can use this counter to capture the memory footprint of GPU.
  int global_index = 0;

  int global_index_threshold = -1;
  int iteration_length = 0;
  /// Index of start of 2nd iteration.
  int location_of_2nd_iteration = 0;
  /// Index of start of 5th iteration
  int location_of_5th_iteration = 0;
  int three_more_iteration_global_index_threshold = -1;

  /// Design specs
  float mem_limit_ratio = 0.70;
  /// Here the smallest block size is 1 MB.
  // size_t smallest_block = 1u << 20;
  size_t smallest_block = 1 << 20;
  /// Used to control ready idx.
  int data_buffer = 4;
  int mutable_data_buffer = 6;
  double max_load;
  int max_idx;
  double total_swap_in_time = 0;
  double total_swap_out_time = 0;
  double tmp_time = 0;
  /// Baseline is vec_run[0] time.
  double tmp_time_baseline;
  int iteration_length_threshold = 1000;

  /// User-configurable scheduling hyper-parameters.
  std::string blk_select_mode = "majority_voting";
  std::string blk_scheduling_mode = "stick-to-limit";
  /// peak memory load limit, measured in MB.
  int workload_mem_limit;

 private:
  shared_ptr<DeviceMemPool> pool_;
};

#endif  // USE_CUDA

#ifdef USE_OPENCL

// Implement Device using OpenCL libs.
class OpenclDevice : public singa::Device {
 public:
  // TODO: Constructor arguments to consider:
  // Path to kernel sources?
  // Select only certain device types?
  OpenclDevice(int id = 0, int num_executors = 1);
  ~OpenclDevice();

  // Overridden, inherited methods
  void SetRandSeed(unsigned seed) override;

  virtual void CopyDataToFrom(Block* dst, Block* src, size_t nBytes,
                              CopyDirection direction, int dst_offset = 0,
                              int src_offset = 0) override;

  /// Ignore for OpenclDevice.
  void Append(DeviceOptInfoToAppend info) override {}

 protected:
  /// The OpenCL device that this object represents.
  /// Each OpenclDevice contains exactly one cl::Device for the lifetime of the
  /// object.
  viennacl::ocl::device this_device;

  /// Each OpenclDevice has one OpenCL context. It is created along with the
  /// creation of this object.
  viennacl::ocl::context vcl_ctx;

  /// Searches the given paths for all .cl files and builds
  /// OpenCL programs, then stores them in the Kernels map.
  void BuildPrograms();

  // Overridden, inherited methods.

  void DoExec(function<void(Context*)>&& fn, int executor) override;

  void CopyToFrom(void* dst, const void* src, size_t nBytes,
                  CopyDirection direction, Context* ctx = nullptr) override;

  /// Allocates memory on this OpenCL device
  /// by creating and returning an empty cl::Buffer object.
  /// with the indicated size.
  void* Malloc(int size) override;

  /// Converts the void pointer into a Buffer object, then deletes the object.
  /// This has the effect of freeing up device memory.
  void Free(void* ptr) override;

  /// Ignore for OpenclDevice.
  void AppendAfterMalloc(Block* block, void* data_ptr, int size) override {}
  void* UpdateGpuPtr(const Block* block_ptr) override { return nullptr; }

 private:
  static const std::string cl_src_path;
};
#endif  // USE_OPENCL

/// This class queries all available calculating devices on a given machine
/// grouped according to manufacturer or device drivers. All methods should be
/// static.
/// If CUDA or OPENCL are not enabled, then the respective related methods
/// should
/// return something that indicates their absence (for example, 0 devices);
/// however they should always be available regardless of compile-time switches.
class Platform {
 public:
  /// Return the default host device
  static std::shared_ptr<Device> GetDefaultDevice() { return defaultDevice; }

#ifdef USE_CUDA
  /// Return the number of total available GPUs
  static int GetNumGPUs();

  /// Return the device IDs of available GPUs.
  /// TODO(wangwei) return the IDs according to free memory in decending order
  static const std::vector<int> GetGPUIDs();

  static const std::pair<size_t, size_t> GetGPUMemSize(const int device);

  /// Return the memory of a GPU <free, total>
  static const std::vector<std::pair<size_t, size_t>> GetGPUMemSize();

  /// Return a string containing all hardware info, e.g., version, memory size.
  static const std::string DeviceQuery(int id, bool verbose = false);

  /// Create a set of CudaGPU Device using 'num_devices' free GPUs.
  static const std::vector<std::shared_ptr<Device>> CreateCudaGPUs(
      const size_t num_devices, size_t init_size = 0);

  /// Create a set of CudaGPU Device using given GPU IDs.
  static const std::vector<std::shared_ptr<Device>> CreateCudaGPUsOn(
      const std::vector<int>& devices, size_t init_size = 0);

  /// Create a set of SwapCudaGPU Device using 'num_devices' free GPUs.
  static const std::vector<std::shared_ptr<Device>> CreateSwapCudaGPUs(
      const std::string& blk_sel_mode, const std::string& blk_sched_mode,
      const int& mem_limit, const size_t num_devices, size_t init_size = 0);

  /// Create a set of SwapCudaGPU Device using given GPU IDs.
  static const std::vector<std::shared_ptr<Device>> CreateSwapCudaGPUsOn(
      const std::string& blk_sel_mode, const std::string& blk_sched_mode,
      const int& mem_limit, const std::vector<int>& devices,
      size_t init_size = 0);

  static std::vector<std::shared_ptr<Device>> UsedDevice;
  /// This function is implementd by Caffe (http://caffe.berkeleyvision.org/).
  /// This function checks the availability of GPU #device_id.
  /// It attempts to create a context on the device by calling cudaFree(0).
  /// cudaSetDevice() alone is not sufficient to check the availability.
  /// It lazily records device_id, however, does not initialize a
  /// context. So it does not know if the host thread has the permission to use
  /// the device or not.
  ///
  /// In a shared environment where the devices are set to EXCLUSIVE_PROCESS
  /// or EXCLUSIVE_THREAD mode, cudaSetDevice() returns cudaSuccess
  /// even if the device is exclusively occupied by another process or thread.
  /// Cuda operations that initialize the context are needed to check
  /// the permission. cudaFree(0) is one of those with no side effect,
  /// except the context initialization.
  static bool CheckDevice(const int device_id);
  static std::mutex mtx_;
#endif  // USE_CUDA

#ifdef USE_OPENCL

  const int GetNumOpenclPlatforms();

  const int GetNumOpenclDevices();

  static const std::shared_ptr<Device> GetDefaultOpenclDevice();

/// Create a \p num_devices set of valid OpenCL devices, regardless of
/// platforms.  If there are fewer valid devices than requested, then this
/// method will return as many as possible. If OpenCL is not in use, this
/// method will return an empty array.
//  static const std::vector<std::shared_ptr<Device>>
//  CreateOpenclDevices(const size_t num_devices);

/// Create a set of valid OpenCL devices, regardless of platforms, assigning
/// \p id to each device in sequence.
/// If there are fewer valid devices than requested, then this method will
/// return as many as possible.
/// If OpenCL is not in use, this method will return an empty array.
//  const std::vector<std::shared_ptr<Device>>
//  CreateOpenclDevices(const vector<int> &id);
#endif  // USE_OPENCL
};

}  // namespace singa

#endif  // SINGA_CORE_DEVICE_H_
