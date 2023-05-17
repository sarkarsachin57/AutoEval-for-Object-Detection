
# Necessary Imports
import os, requests, torch, torchvision, math, cv2, sys, PIL, argparse, imutils, time, json, shutil, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from datetime import datetime
from typing import List, Optional
from imutils.video import VideoStream
from imutils.video import FPS
from tqdm import tqdm 
from itertools import combinations
from PIL import Image

# Imports for YoloV6 for Object Detection
from YOLOv6 import yolov6
sys.modules['yolov6'] = yolov6

from YOLOv6.yolov6.utils.events import LOGGER, load_yaml
from YOLOv6.yolov6.layers.common import DetectBackend
from YOLOv6.yolov6.data.data_augment import letterbox
from YOLOv6.yolov6.utils.nms import non_max_suppression
from YOLOv6.yolov6.core.inferer import Inferer

# Imports for Diversity Selection
from submodlib.helper import create_kernel
from submodlib import FacilityLocationFunction

pd.set_option('display.max_columns', None)


print('Imported all libraries and frameworks.')



cfg = json.load(open('eval.config.json', 'r'))
default_cfg = cfg

print("Configurations : ", cfg)

eval_fps = int(cfg['eval_fps'])
show_fps = True if int(cfg['show_fps_in_frames']) == 1 else False
debug = True if int(cfg['debug']) == 1 else False

compute_device = cfg['device']

assert eval_fps >= 1 and type(eval_fps) == int, f"eval_fps should not be < 1 and expected type int"


cuda = torch.cuda.is_available()

if compute_device == 'gpu' and cuda is False:
    print("GPU or CUDA Not Found!!")

device = torch.device('cpu') if compute_device == 'cpu' else torch.device('cuda' if cuda else 'cpu')

print("Computing device taken :", device)
     

def check_img_size(img_size, s=32, floor=0):
  def make_divisible( x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor
  """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
  if isinstance(img_size, int):  # integer i.e. img_size=640
      new_size = max(make_divisible(img_size, int(s)), floor)
  elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
      new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
  else:
      raise Exception(f"Unsupported type of img_size: {type(img_size)}")

  if new_size != img_size:
      print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
  return new_size if isinstance(img_size,list) else [new_size]*2

def precess_image(path, img_size, stride):
  '''Process image before image inference.'''
  try:
    from PIL import Image
    if type(path) == str:
      img_src = np.asarray(Image.open(path))
    else:
      img_src = path
    assert img_src is not None, f'Invalid image: {path}'
  except Exception as e:
    LOGGER.Warning(e)
  image = letterbox(img_src, img_size, stride=stride)[0]

  # Convert
  image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
  image = torch.from_numpy(np.ascontiguousarray(image))
  image = image.float()  # uint8 to fp16/32
  image /= 255  # 0 - 255 to 0.0 - 1.0

  return image, img_src





def get_model_from_path(model_path, device, img_size = (640, 640)):
    model = DetectBackend(model_path, device=device)
    model.model.float()
    if device.type != 'cpu':
        model(torch.zeros(1, 3, *img_size).to(device).type_as(next(model.model.parameters())))  
    return model

def list_concat(list_in):
    list_out = []
    for x in list_in:
        list_out = list_out + x
    return list_out

def xywh2xyxy(x):
    '''Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right.'''
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
 
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """

    num_classes = prediction.shape[2] - 5  # number of classes
    pred_candidates = torch.logical_and(prediction[..., 4] > conf_thres, torch.max(prediction[..., 5:], axis=-1)[0] > conf_thres)  # candidates
    # Check the parameters.
    assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
    assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

    # Function settings.
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
    time_limit = 10.0  # quit the function when nms cost time exceed the limit time.
    multi_label &= num_classes > 1  # multiple labels per box

    tik = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence

        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # confidence multiply the objectness
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), 1)
        else:  # Only keep the class with highest scores.
            conf, class_idx = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, class_idx.float(), x[:, 5:]), 1)[conf.view(-1) > conf_thres]

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores
        keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = x[keep_box_idx]
        if (time.time() - tik) > time_limit:
            print(f'WARNING: NMS cost time exceed the limited {time_limit}s.')
            break  # time limit exceeded

    return output




def detect(image, model):

    hide_labels: bool = False 
    hide_conf: bool = False 
    stride = model.stride

    img_size:int = 640

    img_size = check_img_size(img_size, s=stride)

    img, img_src = precess_image(image, img_size, stride)
    img = img.to(device)
    if len(img.shape) == 3:
        img = img[None]
        # expand for batch dim

    pred_results = model(img)

    return [pred_results.cpu().detach().numpy(), img.shape, img_src.shape]

    

def get_final_detections_post_nms(pred_results, conf_thresh, iou_thresh, class_names_list, target_classes=None, ignore_classes=[]):

    max_det:int =  1000
    agnostic_nms: bool = False
    target_classes = class_names_list if target_classes is None else target_classes

    pred_results, img_shape, img_src_shape = pred_results
    pred_results = torch.tensor(pred_results)

    dets = non_max_suppression(pred_results, conf_thresh, iou_thresh, None, agnostic_nms, max_det=max_det)[0]

    gn = torch.tensor(img_src_shape)[[1, 0, 1, 0]]  # normalization gain whwh
    if len(dets):
        dets[:, :4] = Inferer.rescale(img_shape[2:], dets[:, :4], img_src_shape).round()

    dets = dets.numpy()

    dets_final = []
    class_probs = []
    
    for det in dets:
     
        x1, y1, x2, y2, conf, cls_id = det[:6]

        class_name = class_names_list[int(cls_id)]

        if class_name not in ignore_classes and class_name in target_classes:

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = round(conf, 4)

            dets_final.append([x1, y1, x2, y2, conf, class_name])
            class_probs.append(det[6:].tolist())
    
    return dets_final, class_probs



def draw_text(img, text,
          pos=(0, 0),
          font=cv2.FONT_HERSHEY_PLAIN,
          font_scale=3,
          text_color=(0, 255, 0),
          font_thickness=2,
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    font_scale = 1
    font = cv2.FONT_HERSHEY_PLAIN
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x, y - text_h - 10), (x + text_w + 10, y), text_color_bg, -1)
    cv2.putText(img, text, (x+5, y-5), font, font_scale, text_color, font_thickness)


def draw_bb_text(frame, text,
          bbox,
          font=cv2.FONT_HERSHEY_PLAIN,
          font_scale=3,
          text_color=(0, 255, 0),
          font_thickness=2,
          text_color_bg=(255, 255, 255)
          ):

    startX, startY, endX, endY = bbox
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    startY = 20 if startY < 20 else startY
    startX = 1 if startX < 1 else startX
    bg = np.ones_like(frame[startY-20:startY,startX-1:startX+text_w+3]).astype('uint8') * 255
    bg[:,:] = text_color_bg
    frame[startY-20:startY,startX-1:startX+text_w+3] = cv2.addWeighted(frame[startY-20:startY,startX-1:startX+text_w+3], 0.0, bg, 1.0, 1)
    cv2.putText(frame, text, (startX, startY-text_h+2), font, font_scale, text_color, font_thickness)



def draw_text_center_top(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          font_scale=3,
          text_color=(0, 255, 0),
          font_thickness=2,
          text_color_bg=(0, 0, 0)
          ):

    font = cv2.FONT_HERSHEY_DUPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    
    frame_text = np.ones((text_h+30,img.shape[1],3)) * text_color_bg

    x = int((frame_text.shape[1]*0.5) - (text_w*0.5))
    y = int((frame_text.shape[0]*0.5) + (text_h*0.5))
    
    final = np.vstack([cv2.putText(frame_text, text, (x, y), font, font_scale, text_color, font_thickness), img])

    return final.astype('uint8')



def add_box_color_legends_to_image(img, text_list, color_list, pad):

    
    img_u = np.ones((50,img.shape[1],3)) * [225, 220, 220]
    # pos=(0, 0),
    font=cv2.FONT_HERSHEY_DUPLEX
    font_scale=0.7
    text_color=(0, 255, 0)
    font_thickness=2
    text_color_bg=(0, 0, 0)

    cx = pad
    cr = 15

    for text, color in zip(text_list, color_list):

        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size

        if color is not None:

            cv2.circle(img_u, (cx, img_u.shape[0] // 2), cr+1, [0,0,0], 1)
            cv2.circle(img_u, (cx, img_u.shape[0] // 2), cr, color, -1)

            x = cx + cr + cr
        
        else:
            x = cx

        y = (img_u.shape[0] // 2) + (text_h // 2)
        cv2.putText(img_u, text, (x, y), font, font_scale, (0, 0, 0), font_thickness)

        cx = x + text_w + pad
        
    final = np.vstack([img, img_u])

    return final.astype('uint8')





def draw_text_center_bottom(img, text,
          pos=(0, 0),
          font=cv2.FONT_HERSHEY_PLAIN,
          font_scale=3,
          text_color=(0, 255, 0),
          font_thickness=2,
          text_color_bg=(0, 0, 0)
          ):

    
    #font_thickness=1
    x, y = pos
    # font_scale = 1
    font = cv2.FONT_HERSHEY_PLAIN
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    # cv2.rectangle(img, (x, y - text_h - 10), (x + text_w + 10, y), text_color_bg, -1)
    # cv2.putText(img, text, (x+5, y-5), font, font_scale, text_color, font_thickness)
    x = int((img.shape[1]*0.5) - (text_w*0.5))
    
    frame_text = np.ones((text_h+20,img.shape[1],3)) * text_color_bg
    final = np.vstack([img, cv2.putText(frame_text, text, (x, y), font, font_scale, text_color, font_thickness)])

    return final.astype('uint8')



def get_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union = box1_area + box2_area - intersection

    return intersection / (union+1e-10)



def get_color(idx):
    idx = idx * 3
    color = (int((37 * idx) % 255), int((17 * idx) % 255), int((29 * idx) % 255))

    return color



def frame_no_to_timestamp_str(frame_no, fps):

    time_stamp = frame_no // fps

    time_stamp_str = "%.2d"%(time_stamp // 3600)+":%.2d"%((time_stamp // 60) % 60)+":%.2d"%(time_stamp % 60)

    return time_stamp_str


def get_estd_processing_time(current_frame, total_frame_count, current_fps):

    time_stamp = (total_frame_count - current_frame) // current_fps

    time_stamp_str = "%.2d"%(time_stamp // 3600)+":%.2d"%((time_stamp // 60) % 60)+":%.2d"%(time_stamp % 60)

    return time_stamp_str
