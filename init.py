
# Necessary Imports
import os, requests, torch, math, cv2, sys, PIL, argparse, imutils, time, json, shutil, random
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

    pred_results, img_shape, img_src_shape = pred_results
    pred_results = torch.tensor(pred_results)

    classes:Optional[List[int]] = target_classes # the classes to keep
    dets = non_max_suppression(pred_results, conf_thresh, iou_thresh, classes, agnostic_nms, max_det=max_det)[0]

    gn = torch.tensor(img_src_shape)[[1, 0, 1, 0]]  # normalization gain whwh
    if len(dets):
        dets[:, :4] = Inferer.rescale(img_shape[2:], dets[:, :4], img_src_shape).round()

    dets = dets.numpy()

    dets_final = []

    for det in dets:
     
        x1, y1, x2, y2, conf, cls_id = det

        class_name = class_names_list[int(cls_id)]

        if class_name not in ignore_classes:

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = round(conf, 4)
            

            dets_final.append([x1, y1, x2, y2, conf, class_name])
    
    return dets_final



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
