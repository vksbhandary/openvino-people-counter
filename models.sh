#!/bin/bash
rm -R models/
mkdir models
cd models/

# these commands works for older and newer versions

wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
tar -xvzf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
cd ssd_mobilenet_v1_coco_2018_01_28
python3  /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model=frozen_inference_graph.pb  --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --input_shape=[1,300,300,3] --input=image_tensor --output=detection_scores,detection_boxes,num_detections --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

# run ssd_mobilenet_v1_coco_2018_01_28 model
# python3 main.py  -m models/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.xml -i resources/Pedestrian_Detect_2_1_1.mp4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
# python3 main.py -i CAM -m models/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.xml -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

cd ..

wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xvzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
cd ssd_mobilenet_v2_coco_2018_03_29
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model=frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --input_shape=[1,300,300,3] --input=image_tensor --output=detection_classes,detection_scores,detection_boxes,num_detections

# run ssd_mobilenet_v2_coco_2018_03_29 model
# python3 main.py  -m models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -i resources/Pedestrian_Detect_2_1_1.mp4   | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

cd ..

wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar -xvzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
cd faster_rcnn_inception_v2_coco_2018_01_28
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json --reverse_input_channels --input_shape=[1,600,1024,3] --input=image_tensor --output=detection_scores,detection_boxes,num_detections --data_type=FP16

# run faster_rcnn_inception_v2_coco_2018_01_28 model
# python3 main.py  -m models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -i resources/Pedestrian_Detect_2_1_1.mp4  | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

cd ..


wget http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
tar -xvzf ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
cd ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model=frozen_inference_graph.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --input_shape=[1,640,640,3] --input=image_tensor --output=detection_scores,detection_boxes,num_detections  --data_type=FP16

# run ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03 model
# python3 main.py  -m models/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.xml -i resources/Pedestrian_Detect_2_1_1.mp4  | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

cd ..

python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name person-detection-retail-0013 --precisions FP16  --output_dir .

# run person-detection-retail-0013 model
# python3 main.py  -m models/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -i resources/Pedestrian_Detect_2_1_1.mp4  | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm


