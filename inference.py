#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
import numpy as np

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device="CPU", cpu_extension=None):
        ### TODO: Load the model ###
        ### TODO: Check for supported layers ###
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Initialize the plugin
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if os.path.isfile(cpu_extension)  and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

        # Read the IR as a IENetwork
        if os.path.isfile(cpu_extension) :
            # considring its old openvino versio which requires extentions to be loaded
            self.network = IENetwork(model=model_xml, weights=model_bin)
        else:
            # new openvino versions dont need extention
            self.network = self.plugin.read_network(model=model_xml, weights=model_bin)
        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)

        # Get the input layer
        total_input = 0
        for i in iter(self.network.inputs):
            total_input = total_input+1

        if total_input>1: # incase of faster RCNN models
            self.input_blob = 'image_tensor' #
        else:
            self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, cur_request_id,image):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        total_input = 0
        for i in iter(self.network.inputs):
            total_input = total_input+1
        if total_input>1:
            b =self.network.inputs['image_tensor'].shape[0]
            H = self.network.inputs['image_tensor'].shape[2]
            W = self.network.inputs['image_tensor'].shape[3]
            arr = np.array([[H, W, 1]], dtype=np.int16)
             
            self.exec_network.start_async(request_id=cur_request_id, 
                inputs={'image_tensor': image, 'image_info':arr})
        else:
            self.exec_network.start_async(request_id=cur_request_id, 
                inputs={self.input_blob: image})

        return

    def wait(self,  cur_request_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.exec_network.requests[cur_request_id].wait(-1)
        return status

    def get_output(self, cur_request_id):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        # print(self.exec_network.requests[0].outputs)
        # print("Output",self.output_blob)
        # for i in iter(self.network.outputs):
            # print(self.exec_network.requests[0].outputs[i].shape)
        
        return self.exec_network.requests[cur_request_id].outputs[self.output_blob]
