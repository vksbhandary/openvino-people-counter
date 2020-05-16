"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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
import time
import socket
import json
import cv2
import numpy as np
import time

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    
    return client

# def get_class_names(class_nums):
#     class_names= []
#     for i in class_nums:
#         class_names.append(CLASSES[int(i)])
#     return class_names

def draw_masks(frame, result, args, width, height):
    '''
    Draw semantic mask classes onto the frame.
    '''
    # Create a mask with color by class
    classes = []
    count = 0
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if int(box[0]) not in classes:
            classes.append(int(box[0]))

        if int(box[1])== 1 and conf >= args.prob_threshold:
            count = count+1
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    return frame, classes, count

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    # print(args)
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    single_img_flag = False
    total_count = 0
    total_frames = 0
    last_count = 0
    duration_time = 0
    cur_request_id = 0
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device)
    net_input_shape = infer_network.get_input_shape()
    ### TODO: Handle the input stream ###
    video_file = args.input
    if video_file == 'CAM': # Check for live feed
        input_stream = 0

    elif video_file.endswith('.jpg') or video_file.endswith('.bmp') :    # Check for input image
        single_img_flag = True
        input_stream = video_file

    else:     # Check for video file
        input_stream = video_file
        assert os.path.isfile(video_file), "Specified input file doesn't exist"

    try:
        cap=cv2.VideoCapture(input_stream)
        cap.open(input_stream)
    except FileNotFoundError:
        print("Cannot locate file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    
    width = int(cap.get(3))
    height = int(cap.get(4))
    # print(width, height)
    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        start_time = time.time()
        infer_network.exec_net(cur_request_id, p_frame)

        ### TODO: Wait for the result ###
        # Get the output of inference
        if infer_network.wait(cur_request_id) == 0:
            stop_time = time.time() - start_time
            result = infer_network.get_output(cur_request_id)
            # print(result)
            # print(result.shape)
            
            ### TODO: Get the results of the inference request ###
            frame, classes, current_count = draw_masks(frame, result, args, width, height)
            # print(classes.shape)
            inference_message = "Inference time: {:.3f}ms".format(stop_time * 1000)
            cv2.putText(frame, inference_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)

            if current_count > last_count: # New entry
                duration_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))            
            
            if current_count < last_count: # Average Time
                duration = int(time.time() - duration_time) 
                client.publish("person/duration", json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": current_count})) # People Count
            last_count = current_count
            ### TODO: Extract any desired stats from the results ###

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###


        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        if key_pressed == 27:
            break
        ### TODO: Write an output image if `single_image_mode` ###
        #Save the Image
        if single_img_flag:
            cv2.imwrite('output_image.jpg', frame)

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
