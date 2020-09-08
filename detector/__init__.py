from .YOLOv3 import YOLOv3
from .DETR import DETR
from .TensorflowObjectDetector import TensorflowObjectDetector


__all__ = ['build_detector']

def build_detector(model, cfg, use_cuda):
    if model == 'yolov3' or model == 'yolov3-tiny':
        return YOLOv3(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES, 
                    score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH, 
                    is_xywh=True, use_cuda=use_cuda)
    elif model == 'detr':
        return DETR()
    else:
        return TensorflowObjectDetector(model)
