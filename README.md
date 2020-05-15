# OpenVINO Project 1: People Counter App

OpenVINO project to count the number of people in a video frame.


# About the project

This project contains necessary files to run an Object detection application on edge devices. This uses OpenVINO toolkit for 


## Setup Environment variables


```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7

```

## Converting Tensorflow model yolo-v3 to IR
```

git clone https://github.com/mystic123/tensorflow-yolo-v3
git checkout ed60b900
cd ..

python3 tensorflow-yolo-v3/convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights


python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model yolov3.pb --reverse_input_channels --input_shape=[1,416,416,3] --scale_values=input_1[255] --input=input_1 --transformations_config=yolo_v3_new.json

```


## converting SSD

```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model=frozen_inference_graph.pb --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ssd_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels

```

## converting Faster RCNN RESNET 101 coco

```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --reverse_input_channels --input_shape=[1,600,1024,3] --input=image_tensor --output=detection_scores,detection_boxes,num_detections

```