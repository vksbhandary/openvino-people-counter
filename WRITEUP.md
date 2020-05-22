# Project Write-Up

The people counter application will demonstrate how to create a smart video IoT solution using Intel® hardware and software tools. The app will detect people in a designated area, providing the number of people in the frame, average duration of people in frame, and total count.


Running this model needs a pretrained model which detects objects (1 is considred as a person's class id). For this app various object detection models were compared. User can train their own model and use this app.


## Converting models to IR

### Converting SSD MobileNet V1 COCO

To convert SSD MobileNet V1 COCO* to IR run the following commands

```
cd models/
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
tar -xvzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
cd ssd_mobilenet_v1_coco_2018_01_28

# for newer version of openvino
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model=frozen_inference_graph.pb --tensorflow_custom_operations_config_update /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --input_shape=[1,300,300,3] --input=image_tensor

# for older version of openvino (openvino_2019.3.376)
python3  /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model=frozen_inference_graph.pb  --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --input_shape=[1,300,300,3] --input=image_tensor --output=detection_scores,detection_boxes,num_detections --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```

### Converting SSD MobileNet V2 COCO

```
cd models/
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xvzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
cd ssd_mobilenet_v2_coco_2018_03_29

# for newer version of openvino
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model=frozen_inference_graph.pb --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --input_shape=[1,300,300,3] --input=image_tensor --output=detection_classes,detection_scores,detection_boxes,num_detections

# for older version of openvino (openvino_2019.3.376)
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model=frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --input_shape=[1,300,300,3] --input=image_tensor --output=detection_classes,detection_scores,detection_boxes,num_detections

```

### Converting Faster R-CNN Inception V2 COCO

```
cd models/
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar -xvzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
cd faster_rcnn_inception_v2_coco_2018_01_28

# for newer version of openvino
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --reverse_input_channels --input_shape=[1,600,1024,3] --input=image_tensor --output=detection_scores,detection_boxes,num_detections --data_type=FP16

# for older version of openvino (openvino_2019.3.376)
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --reverse_input_channels --input_shape=[1,600,1024,3] --input=image_tensor --output=detection_scores,detection_boxes,num_detections --data_type=FP16

```


### Converting SSD ResNet50 FPN COCO

To convert SSD ResNet50 FPN COCO to IR run the following commands

```
cd models/
wget http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
tar -xvzf ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
cd ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
# for newer version of openvino
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model=frozen_inference_graph.pb --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --input_shape=[1,640,640,3] --input=image_tensor --output=detection_scores,detection_boxes,num_detections  --data_type=FP16

# for older version of openvino (openvino_2019.3.376)
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model=frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --input_shape=[1,640,640,3] --input=image_tensor --output=detection_scores,detection_boxes,num_detections  --data_type=FP16
```

## Comparing Model Performance


| Model name  |   Model Precision |   Model size  | COCO mAP[^1] | Inference time |
|-----------|-----------|-----------|---------------|---------------|
| [SSD Mobilenet v1 coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) |   FP32  |  27 Mb | 21 |  ~43ms |
| [SSD MobileNet V2 COCO](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)  |   FP32  |  67 Mb | 22 |  ~69ms |
| [Faster R-CNN Inception V2 COCO](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)  |   FP16  |  27 Mb | 28 |  ~1259ms |
| [SSD ResNet50 FPN COCO](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz)  |   FP16  |  103 Mb | 32 |  ~2590ms |
| [Intel's pretrained person-detection-retail-0013 ](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)  |   FP32  |  1.4 Mb | ~21 (similar to SSD Mobilenet v1 coco) |  ~47ms |

Note :
1. the inference time is average time taken to infer a video frame on a virtual machine provided by udacity for classroom exercise. Inference time can vary based on performance of a machine.
1. The accuracy is taken from [link](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). So it is pre-conversion accuracy.

## Assess Model Use Cases

There are many use cases of people counter in daily life. Some of these are as follows:

1. Surveillance system: This application can be used to count number of people at a given location. This could be helpful in disaster/dangerous situations as it can predict how much time will be required to evacuate the area.
1. Customer services: This application can be used in banks or offices to estimate time required to service the customer or may be stop more customers from entering the premises.
1. Disaster relief:
 - It can be used to detect/count the number of people (survivors) from arial edge device.
 - This app maybe further devevloped to drop supplies for survivors in remote locations based on number of people, using arial edge device.


## Performance issues

Based on the accuracy of the model used to detect people, this app's performance can varry. Edge devices are supposed to work almost instantly. Inference time of even 1000ms is not optimal for getting instantaneous results.