# https://stackoverflow.com/questions/11021047/webcam-stream-and-opencv-python
import cv2
import numpy as np
import pandas as pd
import torch
import keras
import time
# Types definitions
SClass = int
SignLoc = object # 
Frame = np.array # shape(x,y,3)

# Model imports
g_yolo_model =  torch.hub.load('yolov7', 'custom', 'model.pt', source='local') 
g_lenet_model = keras.models.load_model('lenet_model')

# Configurations
g_classes = pd.read_csv('lenet_classes.csv', header=0)[1:]
g_classes = g_classes.columns[1:]
print(g_classes)
g_crop_shape = (64,64)

def find_signs(frame:Frame)->list[SignLoc]:
    start = time.perf_counter()
    detected = g_yolo_model(frame)
    print(time.perf_counter()-start)
    return detected

def classify_sign(frame:Frame,d:SignLoc)->SClass:
    # Cut out sign
    crop = frame[int(d.ymin):int(np.ceil(d.ymax)), int(d.xmin):int(np.ceil(d.xmax))]
    # Resize the cut into correct size
    new_crop = cv2.resize(crop, g_crop_shape, interpolation=cv2.INTER_AREA)
    # Predict correct class
    s_class = g_lenet_model.predict(np.array([new_crop]), verbose=0).argmax()
    return s_class

def outline_sign(frame:Frame,d:SignLoc, s_class:SClass)->Frame:
    new_frame = frame#.copy()
    cv2.rectangle(new_frame, (int(d.xmin), int(d.ymin)), (int(np.ceil(d.xmax)), int(np.ceil(d.ymax))), (255,0,0), 2)
    new_frame = cv2.putText(new_frame, g_classes[s_class], (int(d.xmin), int(d.ymin)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return new_frame


def main():
    window_name = "Show me traffic signs"
    cv2.namedWindow(window_name)

    # Get access to web cam
    vc = cv2.VideoCapture(0)
    
    frame_width = int(vc.get(3))
    frame_height = int(vc.get(4))
    size = (frame_width, frame_height)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
        raise RuntimeError('Problem while opening web cam :(')

    while rval:
        cv2.imshow(window_name, frame)
        rval, frame = vc.read()

        # Find and classify traffic signs
        
        signs_locs = find_signs(frame)

        # Draw boxes around signs with classes
        for i, d in signs_locs.pandas().xyxy[0].iterrows():
            s_class = classify_sign(frame,d) 
            frame = outline_sign(frame,d,s_class)
        

        key = cv2.waitKey(10)
        if key == 27: # exit on ESC
            break

    vc.release()
    cv2.destroyWindow(window_name)

if __name__ == '__main__':
    main()