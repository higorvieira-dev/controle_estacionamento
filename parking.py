import base64
import io
from shapely.geometry import Polygin as shapely_poly
from shapely.geometry import box
import argparse
import pickle
from pathlib import Path
from mrcnn.model import MaskRCNN
import mrcnn.utils
import mrcnn.config
import cv2
import numpy as np
import os 
import git 

if not os.path.exists("Mask_RCNN"):
    git.Git("./").clone("https://github.com/matterport/Mask_RCNN.git")
    
    
class Config(mrcnn.config.Config):
    NAME = "PARKING"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASS = 81
    
config = Config()
config.display()
ROOT_DIR  = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
print(COCO_MODEL_PATH)

if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=Config())
model.load_weights(COCO_MODEL_PATH, by_name=True)

def get_cars(boxes, class_ids):
    cars= []
    for i, box in enumerate(boxes):
        if class_ids[i] in [3,8,6]:
            cars.append(box)
    return np.array(cars)