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


python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_darknet_yolov3_model.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolo_v3.json --batch 1

```


## converting SSD

```
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model=frozen_inference_graph.pb --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ssd_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels

```