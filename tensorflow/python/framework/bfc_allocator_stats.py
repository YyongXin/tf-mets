# Copyright 2019, 2020. IBM All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from tensorflow.python import pywrap_tensorflow as pywrap_tf
from tensorflow.python.util.tf_export import tf_export

def log_bfc_allocator_stats( gpu_id ):
    """Wrapper for Print Allocator stats"""
    pywrap_tf.LogBFCAllocatorStats( gpu_id )

@tf_export("experimental.get_num_allocs")
def get_num_allocs( gpu_id ):
    return pywrap_tf.getNumAllocs( gpu_id )

@tf_export("experimental.get_bytes_in_use")
def get_bytes_in_use( gpu_id ):
    return pywrap_tf.getBytesInUse( gpu_id )

@tf_export("experimental.get_peak_bytes_in_use")
def get_peak_bytes_in_use( gpu_id ):
    return pywrap_tf.getPeakBytesInUse( gpu_id )

@tf_export("experimental.get_largest_alloc_size")
def get_largest_alloc_size( gpu_id ):
    return pywrap_tf.getLargestAllocSize( gpu_id )

@tf_export("experimental.get_bytes_limit")
def get_bytes_limit( gpu_id ):
    return pywrap_tf.getBytesLimit( gpu_id )

@tf_export("experimental.get_bytes_reserved")
def get_bytes_reserved( gpu_id ):
    return pywrap_tf.getBytesReserved( gpu_id )

@tf_export("experimental.get_peak_bytes_reserved")
def get_peak_bytes_reserved( gpu_id ):
    return pywrap_tf.getPeakBytesReserved( gpu_id )

@tf_export("experimental.get_bytes_reservable_limit")
def get_bytes_reservable_limit( gpu_id ):
    return pywrap_tf.getBytesReservableLimit( gpu_id )

@tf_export("experimental.get_bytes_inactive")
def get_bytes_inactive( gpu_id ):
    return pywrap_tf.getBytesInactive( gpu_id )

@tf_export("experimental.get_bytes_active")
def get_bytes_active( gpu_id ):
    return pywrap_tf.getBytesActive( gpu_id )

@tf_export("experimental.get_peak_bytes_active")
def get_peak_bytes_active( gpu_id ):
    return pywrap_tf.getPeakBytesActive( gpu_id )

@tf_export("experimental.get_bytes_reclaimed")
def get_bytes_reclaimed( gpu_id ):
    return pywrap_tf.getBytesReclaimed( gpu_id )

@tf_export("experimental.get_num_single_reclaims")
def get_num_single_reclaims( gpu_id ):
    return pywrap_tf.getNumSingleReclaims( gpu_id )

@tf_export("experimental.get_num_full_reclaims")
def get_num_full_reclaims( gpu_id ):
    return pywrap_tf.getNumFullReclaims( gpu_id )

@tf_export("experimental.get_num_defragmentations")
def get_num_defragmentations( gpu_id ):
    return pywrap_tf.getNumDefragmentations( gpu_id )

@tf_export("experimental.get_bytes_defragged")
def get_bytes_defragged( gpu_id ):
    return pywrap_tf.getBytesDefragged( gpu_id )
