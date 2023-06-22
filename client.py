# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
from functools import partial
import numpy as np
import queue
import random
import time
import signal
import os

from tritonclient.utils import *
import tritonclient.grpc as grpcclient

class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)

start_time = None

def signal_handler(signal, frame):
    print("\nCtrl+C pressed! Exiting...")
    calculate_and_print_duration()
    sys.exit(0)

def calculate_and_print_duration():
    global start_time
    end_time = time.time()
    duration = end_time - start_time
    duration_hours = duration / 3600
    print("Program duration:", duration_hours, "hours.")

signal.signal(signal.SIGINT, signal_handler)

start_time = time.time()

model_name = "square_decoupled"
inputs = [grpcclient.InferInput("IN", [1], np_to_triton_dtype(np.int32))]
outputs = [grpcclient.InferRequestedOutput("OUT")]

user_data = UserData()

triton_url = os.getenv("PATH")

with grpcclient.InferenceServerClient(url=os.getenv("TRITON_URL")) as triton_client:

    # Establish stream
    triton_client.start_stream(callback=partial(callback, user_data))
    loop_count = 0
    while True:
        num_values = random.randint(1, 20)
        in_values = [random.randint(1, 30) for _ in range(num_values)]
        for i in range(len(in_values)):
            in_data = np.array([in_values[i]], dtype=np.int32)
            inputs[0].set_data_from_numpy(in_data)

            triton_client.async_stream_infer(model_name=model_name,
                                             inputs=inputs,
                                             request_id=str(i),
                                             outputs=outputs)

        # Retrieve results...
        recv_count = 0
        expected_count = sum(in_values)
        result_dict = {}
        while recv_count < expected_count:
            data_item = user_data._completed_requests.get()
            if type(data_item) == InferenceServerException:
                calculate_and_print_duration()
                raise data_item
            else:
                this_id = data_item.get_response().id
                if this_id not in result_dict.keys():
                    result_dict[this_id] = []
                result_dict[this_id].append((recv_count, data_item))

            recv_count += 1

        # Validate results...
        for i in range(len(in_values)):
            this_id = str(i)
            if in_values[i] != 0 and this_id not in result_dict.keys():
                print("response for request id {} not received".format(this_id))
                sys.exit(1)
            elif in_values[i] == 0 and this_id in result_dict.keys():
                print("received unexpected response for request id {}".format(
                    this_id))
                sys.exit(1)
            if in_values[i] != 0:
                if len(result_dict[this_id]) != in_values[i]:
                    print("expected {} many responses for request id {}, got {}".
                          format(in_values[i], this_id, result_dict[this_id]))
                    sys.exit(1)

            if in_values[i] != 0:
                result_list = result_dict[this_id]
                expected_data = np.array([in_values[i]], dtype=np.int32)
                for j in range(len(result_list)):
                    this_data = result_list[j][1].as_numpy('OUT')
                    if not np.array_equal(expected_data, this_data):
                        print("incorrect data: expected {}, got {}".format(
                            expected_data, this_data))
                        sys.exit(1)

        loop_count += 1
        if loop_count % 15 == 0:
            print("PASS: {} iterations".format(loop_count))

    sys.exit(0)
