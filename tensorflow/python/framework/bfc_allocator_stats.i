/* Copyright 2019, 2020. IBM All Rights Reserved.

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

// typemap for gpu ID
%typemap(in) int gpu_id {
    $1 = PyLong_AsLong($input);
}

// typemaps for returned stats
%typemap(out) int64 {
    $result = PyLong_FromLongLong($1);
}

%{
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h" // for GPUProcessState class
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"            // for TfGpuId type
#include "tensorflow/core/framework/allocator.h"                  // for Allocator class
#include "tensorflow/core/common_runtime/bfc_allocator.h"         // for BFCAllocator
#include "tensorflow/core/platform/logging.h"                     // for VLOG

#include <iostream>                                                // for stringstream

absl::optional<tensorflow::AllocatorStats> GetBFCAllocatorStats( int gpu_id )
{

    tensorflow::GPUProcessState * ps = tensorflow::GPUProcessState::singleton();
    bool gpu_registered = ps->HasGPUDevice();

    if(gpu_registered)
    {
        // placeholder variable for input to `GetGPUAllocator`
        // It will be ignored as we are making sure the gpu device has been created
        // before we attempt to get the gpu allocator
        size_t total_bytes = 1;

        tensorflow::TfGpuId tf_gpu_id(gpu_id);
        tensorflow::GPUOptions options;
        std::string bfc = "BFC";
        options.set_allocator_type(bfc);

        tensorflow::Allocator * allocator = ps->GetGPUAllocator(options,
                                                    tf_gpu_id,
                                                    total_bytes);

        std::string name = allocator->Name();

        tensorflow::BFCAllocator * bfc_allocator = static_cast<tensorflow::BFCAllocator *>(allocator);

        return bfc_allocator->GetStats();
    }
    else
    {
        LOG(ERROR) << "(GetBFCAllocatorStats) No GPU device registered. Skipping getting stats\n";
        return absl::nullopt;
    }
}

int64 getNumAllocs( int gpu_id )
{
    int64 result = -1;
    absl::optional<tensorflow::AllocatorStats> allocator_stats = GetBFCAllocatorStats( gpu_id );

    if( allocator_stats != absl::nullopt )
    {
        result = allocator_stats->num_allocs;
    }
    else
    {
        LOG(ERROR) << "(getNumAllocs) - Could not retrieve BFC Allocator Stats";
    }

    return result;
}

int64 getBytesInUse( int gpu_id )
{
    int64 result = -1;
    absl::optional<tensorflow::AllocatorStats> allocator_stats = GetBFCAllocatorStats( gpu_id );

    if( allocator_stats != absl::nullopt )
    {
        result = allocator_stats->bytes_in_use;
    }
    else
    {
        LOG(ERROR) << "(getBytesInUse) - Could not retrieve BFC Allocator Stats";
    }

    return result;
}

int64 getPeakBytesInUse( int gpu_id )
{
    int64 result = -1;
    absl::optional<tensorflow::AllocatorStats> allocator_stats = GetBFCAllocatorStats( gpu_id );

    if( allocator_stats != absl::nullopt )
    {
        result = allocator_stats->peak_bytes_in_use;
    }
    else
    {
        LOG(ERROR) << "(getPeakBytesInUse) - Could not retrieve BFC Allocator Stats";
    }

    return result;
}

int64 getLargestAllocSize( int gpu_id )
{
    int64 result = -1;
    absl::optional<tensorflow::AllocatorStats> allocator_stats = GetBFCAllocatorStats( gpu_id );

    if( allocator_stats != absl::nullopt )
    {
        result = allocator_stats->largest_alloc_size;
    }
    else
    {
        LOG(ERROR) << "(getLargestAllocSize) - Could not retrieve BFC Allocator Stats";
    }

    return result;
}

int64 getBytesLimit( int gpu_id )
{
    int64 result = -1;
    absl::optional<tensorflow::AllocatorStats> allocator_stats = GetBFCAllocatorStats( gpu_id );

    if( allocator_stats != absl::nullopt )
    {
        if( allocator_stats->bytes_limit.has_value() )
        {
            result = allocator_stats->bytes_limit.value();
        }
        else
        {
            LOG(INFO) << "(getBytesLimit) - Optional value is empty";
        }
    }
    else
    {
        LOG(ERROR) << "(getBytesLimit) - Could not retrieve BFC Allocator Stats";
    }

    return result;
}

int64 getBytesReserved( int gpu_id )
{
    int64 result = -1;
    absl::optional<tensorflow::AllocatorStats> allocator_stats = GetBFCAllocatorStats( gpu_id );

    if( allocator_stats != absl::nullopt )
    {
        result = allocator_stats->bytes_reserved;
    }
    else
    {
        LOG(ERROR) << "(getBytesReserved) - Could not retrieve BFC Allocator Stats";
    }

    return result;
}

int64 getPeakBytesReserved( int gpu_id )
{
    int64 result = -1;
    absl::optional<tensorflow::AllocatorStats> allocator_stats = GetBFCAllocatorStats( gpu_id );

    if( allocator_stats != absl::nullopt )
    {
        result = allocator_stats->peak_bytes_reserved;
    }
    else
    {
        LOG(ERROR) << "(getPeakBytesReserved) - Could not retrieve BFC Allocator Stats";
    }

    return result;
}

int64 getBytesReservableLimit( int gpu_id )
{
    int64 result = -1;
    absl::optional<tensorflow::AllocatorStats> allocator_stats = GetBFCAllocatorStats( gpu_id );

    if( allocator_stats != absl::nullopt )
    {
        if( allocator_stats->bytes_reservable_limit.has_value() )
        {
            result = allocator_stats->bytes_reservable_limit.value();
        }
        else
        {
            LOG(INFO) << "(getBytesReservableLimit) - Optional value is empty";
        }
    }
    else
    {
        LOG(ERROR) << "(getBytesReservableLimit) - Could not retrieve BFC Allocator Stats";
    }

    return result;
}

int64 getBytesInactive( int gpu_id )
{

    int64 result = -1;
    absl::optional<tensorflow::AllocatorStats> allocator_stats = GetBFCAllocatorStats( gpu_id );

    if( allocator_stats != absl::nullopt )
    {
        result = allocator_stats->bytes_inactive;
    }
    else
    {
        LOG(ERROR) << "(getBytesInactive) - Could not retrieve BFC Allocator Stats";
    }

    return result;
}

int64 getBytesActive( int gpu_id )
{
    int64 result = -1;
    absl::optional<tensorflow::AllocatorStats> allocator_stats = GetBFCAllocatorStats( gpu_id );

    if( allocator_stats != absl::nullopt )
    {
        result = allocator_stats->bytes_active();
    }
    else
    {
        LOG(ERROR) << "(getBytesActive) - Could not retrieve BFC Allocator Stats";
    }

    return result;
}

int64 getPeakBytesActive( int gpu_id )
{
    int64 result = -1;
    absl::optional<tensorflow::AllocatorStats> allocator_stats = GetBFCAllocatorStats( gpu_id );

    if( allocator_stats != absl::nullopt )
    {
        result = allocator_stats->peak_bytes_active;
    }
    else
    {
        LOG(ERROR) << "(getPeakBytesActive) - Could not retrieve BFC Allocator Stats";
    }

    return result;
}

int64 getBytesReclaimed( int gpu_id )
{
    int64 result = -1;
    absl::optional<tensorflow::AllocatorStats> allocator_stats = GetBFCAllocatorStats( gpu_id );

    if( allocator_stats != absl::nullopt )
    {
        result = allocator_stats->bytes_reclaimed;
    }
    else
    {
        LOG(ERROR) << "(getBytesReclaimed) - Could not retrieve BFC Allocator Stats";
    }

    return result;
}

int64 getNumSingleReclaims( int gpu_id )
{
    int64 result = -1;
    absl::optional<tensorflow::AllocatorStats> allocator_stats = GetBFCAllocatorStats( gpu_id );

    if( allocator_stats != absl::nullopt )
    {
        result = allocator_stats->num_single_reclaims;
    }
    else
    {
        LOG(ERROR) << "(getNumSingleReclaims) - Could not retrieve BFC Allocator Stats";
    }

    return result;
}

int64 getNumFullReclaims( int gpu_id )
{
    int64 result = -1;
    absl::optional<tensorflow::AllocatorStats> allocator_stats = GetBFCAllocatorStats( gpu_id );

    if( allocator_stats != absl::nullopt )
    {
        result = allocator_stats->num_full_reclaims;
    }
    else
    {
        LOG(ERROR) << "(getNumFullReclaims) - Could not retrieve BFC Allocator Stats";
    }

    return result;
}

int64 getNumDefragmentations( int gpu_id )
{
    int64 result = -1;
    absl::optional<tensorflow::AllocatorStats> allocator_stats = GetBFCAllocatorStats( gpu_id );

    if( allocator_stats != absl::nullopt )
    {
        result = allocator_stats->num_defragmentations;
    }
    else
    {
        LOG(ERROR) << "(getNumDefragmentations) - Could not retrieve BFC Allocator Stats";
    }

    return result;
}

int64 getBytesDefragged( int gpu_id )
{
    int64 result = -1;
    absl::optional<tensorflow::AllocatorStats> allocator_stats = GetBFCAllocatorStats( gpu_id );

    if( allocator_stats != absl::nullopt )
    {
        result = allocator_stats->bytes_defragged;
    }
    else
    {
        LOG(ERROR) << "(getBytesDefragged) - Could not retrieve BFC Allocator Stats";
    }

    return result;
}

void LogBFCAllocatorStats( int gpu_id )
{

    std::stringstream ss;
    ss << "\nEnter> LogBFCAllocatorStats\n";

    absl::optional<tensorflow::AllocatorStats> allocator_stats = GetBFCAllocatorStats( gpu_id );

    if ( allocator_stats != absl::nullopt )
    {
        ss << allocator_stats->DebugString();
    }
    else
    {
        ss << "Unable to log stats due to error retrieving Allocator\n\n\n";
    }

    ss << "<Exit LogBFCAllocatorStats";

    // Log the stream
    VLOG(2) << ss.str();
}

%}

// Function to log allocator stats for requested GPU
void LogBFCAllocatorStats( int gpu_id );

// Getter functions for BFC Allocator statistics
int64 getNumAllocs( int gpu_id );
int64 getBytesInUse( int gpu_id );
int64 getPeakBytesInUse( int gpu_id );
int64 getLargestAllocSize( int gpu_id );
int64 getBytesLimit( int gpu_id );
int64 getBytesReserved( int gpu_id );
int64 getPeakBytesReserved( int gpu_id );
int64 getBytesReservableLimit( int gpu_id );
int64 getBytesInactive( int gpu_id );
int64 getBytesActive( int gpu_id );
int64 getPeakBytesActive( int gpu_id );
int64 getBytesReclaimed( int gpu_id );
int64 getNumSingleReclaims( int gpu_id );
int64 getNumFullReclaims( int gpu_id );
int64 getNumDefragmentations( int gpu_id );
int64 getBytesDefragged( int gpu_id );
