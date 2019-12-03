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

from mxnet import np, npx, autograd
npx.set_np()

lhs_shape = (2048, 1024)
rhs_shape = (1024, 4096)

lhs = np.random.uniform(-1.0, 1.0, size=lhs_shape)
rhs = np.random.uniform(-1.0, 1.0, size=rhs_shape)

lhs.attach_grad()  # attach a gradient buffer to lhs
rhs.attach_grad()  # attach a gradient buffer to rhs

with autograd.record():  # autograd.record() gives a scope captures code
                         # that needs gradient computation
    out = np.dot(lhs, rhs)  # a normal NumPy opration

out.backward()  # compute the gradients with respect to its variables
                # in this case out = dot(lhs, rhs) so both lhs and rhs
                # are out's variables

# lhs's gradient is now in lhs.grad
# rhs's gradient is now in rhs.grad
