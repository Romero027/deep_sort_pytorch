import torch
import logging
import numpy as np
import cv2

from .yolo_utils import get_all_boxes, xywh_to_xyxy, xyxy_to_xywh



class DETR(object):
    def __init__(self, score_thresh=0.7, conf_thresh=0.01, nms_thresh=0.45,
                 is_xywh=False, use_cuda=True):
        # net definition
        self.net = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.device = "cuda" if use_cuda else "cpu"
        self.net.eval()
        self.net.to(self.device)

        # constants
        self.score_thresh = score_thresh
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.use_cuda = use_cuda
        self.is_xywh = is_xywh
        self.num_classes = 80
        self.class_names = self.load_class_names("detector/DETR/coco.names")
    
    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = box_xywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def __call__(self, ori_img):
        # img to tensor
        assert isinstance(ori_img, np.ndarray), "input must be a numpy array!"
        img = ori_img.astype(np.float) / 255.

        img = cv2.resize(img, (416, 416))
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)

        # forward
        with torch.no_grad():
            img = img.to(self.device)
            outputs = self.net(img)
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.7
            boxes = outputs['pred_boxes'][0, keep]
            
            # boxes = get_all_boxes(out_boxes, self.conf_thresh, self.num_classes,
            #                      use_cuda=self.use_cuda)  # batch size is 1
            # boxes = nms(boxes, self.nms_thresh)

            #boxes = post_process(boxes, self.net.num_classes, self.conf_thresh, self.nms_thresh)[0].cpu()
            #boxes = boxes[boxes[:, -2] > self.score_thresh, :]  # bbox xmin ymin xmax ymax

        if len(boxes) == 0:
            bbox = torch.FloatTensor([]).reshape([0, 4])
            cls_conf = torch.FloatTensor([])
            cls_ids = torch.LongTensor([])
        else:
            height, width = ori_img.shape[:2]
            bbox = boxes[:, :4]
            if self.is_xywh:
                # bbox x y w h
                bbox = xyxy_to_xywh(bbox)

            bbox *= torch.FloatTensor([[width, height, width, height]])
            cls_conf = probas[keep]
            cls_ids = torch.argmax(probas, dim=1, keepdim=True)

        return bbox.numpy(), cls_conf.numpy(), cls_ids.numpy()

    def load_class_names(self, namesfile):
        with open(namesfile, 'r', encoding='utf8') as fp:
            class_names = [line.strip() for line in fp.readlines()]
        return class_names
