# OpenVINO Project 1: People Counter App

OpenVINO project to count the number of people in a video frame.


# About the project

This project contains necessary files to run an Object detection application on edge devices. This uses OpenVINO toolkit for 


## Converting a Tensorflow model to OpenVINO IR
```
python3 mo_tf.py --input_model frozen_inference_graph.pb --input_checkpoint model.ckpt.meta

```