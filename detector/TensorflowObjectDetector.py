import cv2
import numpy as np

#model_dir = '/tank/kuntai/code/deep_sort_pytorch/models/'
model_dir = '/home/ubuntu/deep_sort_pytorch/models/'
class_names = {0: u'__background__',
 1: u'person',
 2: u'bicycle',
 3: u'car',
 4: u'motorcycle',
 5: u'airplane',
 6: u'bus',
 7: u'train',
 8: u'truck',
 9: u'boat',
 10: u'traffic light',
 11: u'fire hydrant',
 12: u'stop sign',
 13: u'parking meter',
 14: u'bench',
 15: u'bird',
 16: u'cat',
 17: u'dog',
 18: u'horse',
 19: u'sheep',
 20: u'cow',
 21: u'elephant',
 22: u'bear',
 23: u'zebra',
 24: u'giraffe',
 25: u'backpack',
 26: u'umbrella',
 27: u'handbag',
 28: u'tie',
 29: u'suitcase',
 30: u'frisbee',
 31: u'skis',
 32: u'snowboard',
 33: u'sports ball',
 34: u'kite',
 35: u'baseball bat',
 36: u'baseball glove',
 37: u'skateboard',
 38: u'surfboard',
 39: u'tennis racket',
 40: u'bottle',
 41: u'wine glass',
 42: u'cup',
 43: u'fork',
 44: u'knife',
 45: u'spoon',
 46: u'bowl',
 47: u'banana',
 48: u'apple',
 49: u'sandwich',
 50: u'orange',
 51: u'broccoli',
 52: u'carrot',
 53: u'hot dog',
 54: u'pizza',
 55: u'donut',
 56: u'cake',
 57: u'chair',
 58: u'couch',
 59: u'potted plant',
 60: u'bed',
 61: u'dining table',
 62: u'toilet',
 63: u'tv',
 64: u'laptop',
 65: u'mouse',
 66: u'remote',
 67: u'keyboard',
 68: u'cell phone',
 69: u'microwave',
 70: u'oven',
 71: u'toaster',
 72: u'sink',
 73: u'refrigerator',
 74: u'book',
 75: u'clock',
 76: u'vase',
 77: u'scissors',
 78: u'teddy bear',
 79: u'hair drier',
 80: u'toothbrush'}

class TensorflowObjectDetector:

    def __init__(self,
                 model_name,
                 confidence_threshold=0.01):
        self.confidence_threshold = confidence_threshold
        model_path = model_dir + model_name + '.pb'
        self.net = cv2.dnn.readNetFromTensorflow(model_path, model_path+'txt')
        self.model_path = model_path
        self.model_name = model_path.split('/')[-1].split('.')[0]
        self.class_names = class_names

    def infer(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # the default input size of the model.
        blob_size = (300, 300)
        if 'faster_rcnn' in self.model_path:
            blob_size = (1024, 576)
        blob = cv2.dnn.blobFromImage(frame, size=blob_size, swapRB=True, crop=False)
        self.net.setInput(blob)
        detections = self.net.forward()
        return detections

    def postprocess_detection(self, detections, frameshape):
        bboxes = []
        confs = []
        labels = []

        x_scale = frameshape[0]
        y_scale = frameshape[1]
        

        for detection in detections[0, 0]:
            ymin = detection[3]
            xmin = detection[4]
            ymax = detection[5]
            xmax = detection[6]

            conf = float(detection[2])
            label = int(detection[1])

            x, y, w, h = xmin, ymin, xmax-xmin, ymax-ymin
            x *= x_scale
            w *= x_scale
            y *= y_scale
            h *= y_scale
            
            if w<=0 or h <=0:
                continue
            if conf < self.confidence_threshold:
                continue

            bboxes.append([x,y,w,h])
            confs.append(conf)
            labels.append(label)
        return np.array(bboxes), np.array(confs), np.array(labels)

    def __call__(self, img):
        return self.postprocess_detection(self.infer(img), img.shape)

    # def plot(self, frame, plot_class_name, save_path):

    #     rows, cols, channels = frame.shape
    #     detections = self.infer(frame)

    #     detections = [detection for detection in detections[0, 0]]
    #     detections = sorted(detections, key = lambda x: -x[2])
    #     i = 0
    #     for detection in detections:
    #         i += 1
    #         score = float(detection[2])
    #         class_id = int(detection[1])

    #         # filters
    #         #if class_id not in [3, 6, 7, 8]:
    #         #    continue
    #         if score <= self.confidence_threshold:
    #             continue

    #         # plot bounding box
    #         left = detection[3] * cols
    #         top = detection[4] * rows
    #         right = detection[5] * cols
    #         bottom = detection[6] * rows

    #         if left > right or top > bottom:
    #             continue

    #         if (detection[5]-detection[3]) * (detection[6]-detection[4]) > 0.05:
    #             continue

    #         #draw a red rectangle around detected objects
    #         cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)

    #     cv2.imwrite(save_path + self.model_name + '.jpg', frame)
 
