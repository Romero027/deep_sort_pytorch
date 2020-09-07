# Deep Sort with PyTorch

## Introduction
This is an implement of MOT tracking algorithm deep sort. Deep sort is basicly the same with sort but added a CNN model to extract features in image of human part bounded by a detector. This CNN model is indeed a RE-ID model and the detector used in [PAPER](https://arxiv.org/abs/1703.07402) is FasterRCNN , and the original source code is [HERE](https://github.com/nwojke/deep_sort).  
However in original code, the CNN model is implemented with tensorflow, which I'm not familier with. SO I re-implemented the CNN feature extraction model with PyTorch, and changed the CNN model a little bit. Also, I use **YOLOv3** to generate bboxes instead of FasterRCNN.

## Quick Start
0. Check all dependencies installed
```bash
pip install -r requirements.txt
```

1. Download YOLOv3 parameters
```
cd detector/YOLOv3/weight/
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
cd ../../../
```

2. Compile nms module
```bash
cd detector/YOLOv3/nms
sh build.sh
cd ../../..
```

2. Download Videos
```bash
cd data
sh download_MOT16.sh
cd ..
```

Notice:
If compiling failed, the simplist way is to **Upgrade your pytorch >= 1.1 and torchvision >= 0.3" and you can avoid the troublesome compiling problems which are most likely caused by either `gcc version too low` or `libraries missing`.

5. Run demo
```
usage: python yolov3_deepsort.py VIDEO_PATH
                                [--help]
                                [--frame_interval FRAME_INTERVAL]
                                [--config_detection CONFIG_DETECTION]
                                [--config_deepsort CONFIG_DEEPSORT]
                                [--detection_model DETECTION MODEL]
                                [--display]
                                [--display_width DISPLAY_WIDTH]
                                [--display_height DISPLAY_HEIGHT]
                                [--sample_rate SAMPLE_RATE]
                                [--save_path SAVE_PATH]         
                                [--save_file SAVE_FILE]
                                [--cpu]          

# yolov3 + deepsort
python yolov3_deepsort.py [VIDEO_NAME] --save_file [VIDEO_NAME]
Example:  python yolov3_deepsort.py data/MOT16-02.avi --save_file MOT16-02

# yolov3_tiny + deepsort
python yolov3_deepsort.py [VIDEO_PATH] --config_detection ./configs/yolov3_tiny.yaml --detection_model yolov3-tiny --save_file [VIDEO_NAME]

```
Use `--display` to enable display.  
Results will be saved to `./output/results.avi` and `./output/results.txt`.

6. [Optional] Evaluate results

```
python -m motmetrics.apps.evaluateTracking --help

Example: python -m motmetrics.apps.evaluateTracking MOT16Labels/train output/ seq
```

## References
- paper: [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)

- code: [nwojke/deep_sort](https://github.com/nwojke/deep_sort)

- paper: [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

- code: [Joseph Redmon/yolov3](https://pjreddie.com/darknet/yolo/)
