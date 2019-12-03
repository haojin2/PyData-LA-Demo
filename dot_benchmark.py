# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from mxnet import np, npx
npx.set_np()
import time
import numpy as onp

skip_gpu = False if npx.num_gpus() >= 1 else True

lhs_shape = (2048, 2048)
rhs_shape = (2048, 4096)

onp_lhs = onp.random.uniform(-1.0, 1.0, size=lhs_shape).astype(np.float32)
onp_rhs = onp.random.uniform(-1.0, 1.0, size=rhs_shape).astype(np.float32)
onp_start = time.time()
onp_out = onp.dot(onp_lhs, onp_rhs)
onp_end = time.time()
print("official numpy consumed:", onp_end - onp_start, "seconds.")

dnp_cpu_lhs = np.random.uniform(-1.0, 1.0, size=lhs_shape)
dnp_cpu_rhs = np.random.uniform(-1.0, 1.0, size=rhs_shape)
dnp_cpu_start = time.time()
dnp_cpu_out = np.dot(dnp_cpu_lhs, dnp_cpu_rhs)
dnp_cpu_end = time.time()
print("NP on MXNet consumed:", dnp_cpu_end - dnp_cpu_start, "seconds on CPU.")

if not skip_gpu:
    dnp_gpu_lhs = np.random.uniform(-1.0, 1.0, size=lhs_shape).as_in_ctx(npx.gpu(0))
    dnp_gpu_rhs = np.random.uniform(-1.0, 1.0, size=rhs_shape).as_in_ctx(npx.gpu(0))
    dnp_gpu_start = time.time()
    dnp_gpu_out = np.dot(dnp_gpu_lhs, dnp_gpu_rhs)
    dnp_gpu_end = time.time()
    print("NP on MXNet consumed:", dnp_gpu_end - dnp_gpu_start, "seconds on GPU.")
