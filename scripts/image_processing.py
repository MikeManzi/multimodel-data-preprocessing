import cv2
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)

def extract_features(img_arr):
    if len(img_arr.shape) == 2: 
        hist = cv2.calcHist([img_arr], [0], None, [32], [0, 256]).flatten()
    else:
        chans = cv2.split(img_arr)
        hist = np.concatenate([
            cv2.calcHist([c], [0], None, [32], [0, 256]).flatten() for c in chans
        ])
    return hist / np.sum(hist) 

def process_image(image_path):
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
    img_arr = np.array(img)
    feats = extract_features(img_arr)
    return feats.reshape(1, -1)


