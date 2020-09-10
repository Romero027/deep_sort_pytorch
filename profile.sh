#!/bin/sh

fpss=( 1.0 0.66 0.33 0.15 0.066 )
resolutions=( 1080p.avi 720p.avi 480p.avi 360p.avi 240p.avi )
models=(detr yolov3 yolov3-tiny)

for fps in "${fpss[@]}"
do
  for res in "${resolutions[@]}"
  do 
    for model in "${models[@]}"
    do
      if [ "${model}" = "yolov3-tiny" ]; then
        python yolov3_deepsort.py data/${res} --sample_rate ${fps} --detection_model ${model} --config_detection ./configs/yolov3_tiny.yaml --save_detection True
      else
        python yolov3_deepsort.py data/${res} --sample_rate ${fps} --detection_model ${model} --save_detection True
      fi
    done
  done
done
