#Script for converting images from Mapillary dataset to LeNet format

import pandas as pd
import os
import cv2
import json
import numpy as np


annotations = os.listdir('./mtsd_v2_fully_annotated/annotations')
images = os.listdir('./images')

#df = pd.Dataframe({"Sign":[], "Type":[]})
signs = []
types = []
new_shape = (64,64)

for image in images:
    
    name = image.split('.')[0]
    if name +'.json' in annotations:
        img = cv2.imread(f'./images/{image}')
        with open(f'./mtsd_v2_fully_annotated/annotations/{name}.json', 'r') as fcc_file:
            data = json.load(fcc_file)
            for i, object in enumerate(data["objects"]):
                #print(object)
                
                #crop = img[int(object["bbox"]["ymin"]):int(np.ceil(object["bbox"]["ymax"])), int(object["bbox"]["xmin"]):int(np.ceil(object["bbox"]["xmax"]))]
                #new = cv2.resize(crop, new_shape, interpolation=cv2.INTER_AREA)
                #cv2.imshow('image', new)
                #cv2.waitKey(0)

                signs.append(f'./lenet_images/{name}_{i}.jpg')
                types.append(object['label'])

                #cv2.imwrite(f'./lenet_images/{name}_{i}.jpg', new)

df = pd.DataFrame({"Sign":signs, "Type":types})
df.to_csv('lenet_data.csv')