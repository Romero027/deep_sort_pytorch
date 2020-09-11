#!/bin/sh

fpss=( 0.33 0.15 0.066 )
resolutions=( 1080.avi 720.avi 480.avi 360.avi 240.avi )
models=(yolov3)

for fps in "${fpss[@]}"
do
  for res in "${resolutions[@]}"
  do 
    for model in "${models[@]}"
    do
      if [ "${model}" = "yolov3-tiny" ]; then
        python yolov3_deepsort.py data/${res} --sample_rate ${fps} --detection_model ${model} --config_detection ./configs/yolov3_tiny.yaml --save_file MOT16-02
      else
        python yolov3_deepsort.py data/${res} --sample_rate ${fps} --detection_model ${model} --save_file MOT16-02
      fi
      #python -m motmetrics.apps.evaluateTracking MOT16Labels/train output/ seq
    done
  done
done
