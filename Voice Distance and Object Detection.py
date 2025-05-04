import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO

from PIL import Image
import cv2
from gtts import gTTS
import pygame

import time
from time import strftime

if tf.__version__ < '1.10.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.10.* or later!')

sys.path.append("..")

from utils import label_map_util

from utils import visualization_utils as vis_util


# What model to use
MODEL_NAME = 'inference_graph'
CWD_PATH = os.getcwd()
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

NUM_CLASSES = 11

# Load a frozen graph

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

# Loading label map

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed
video = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_file = 'vid_' + strftime("%Y%m%d_%H%M%S") + '.avi'
out = cv2.VideoWriter(os.path.join('video_records', video_file), fourcc, 5, (1920, 1080))
ret = video.set(3, 1920)
ret = video.set(4, 1080)

lists = []
lists.clear()
pygame.init()
frame_count = 0;

while (True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    ret, frame1 = video.read()
    frame_count += 1
    frame_expanded = np.expand_dims(frame, axis=0)
    frame1 = cv2.GaussianBlur(frame1, (5, 5), 0)
    hls = cv2.cvtColor(frame1, cv2.COLOR_BGR2HLS)
    low_yellow = np.array([22, 93, 0])
    up_yellow = np.array([45, 255, 255])
    mask = cv2.inRange(hls, low_yellow, up_yellow)
    edges = cv2.Canny(mask, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 150, maxLineGap=100)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame1, (x1, y1), (x2, y2), (0,0, 255), 3)
    cv2.imshow("Takip", frame1)
   

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    f, names = vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    for i, b in enumerate(boxes[0]):
        #                 agac araba bisiklet insan kedi köpek levha motosiklet yaya geçidi
        if classes[0][i] == 1 or classes[0][i] == 2 or classes[0][i] == 3 or classes[0][i] == 4 or classes[0][i] == 5 or \
                classes[0][i] == 7 or classes[0][i] == 8 or classes[0][i] == 9 or classes[0][i] == 10:
            if scores[0][i] >= 0.5:
                mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 4), 1)
                cv2.putText(f, '{}'.format(apx_distance), (int(mid_x * 800), int(mid_y * 450)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if apx_distance < 0.2:
                    if (frame_count % 5 == 0):
                        for i in names:

                            if i not in lists:
                                lists.append(i)

                                human_string = i

                                lst = human_string.split()
                                human_string = " ".join(lst[0:4])
                                human_string_filename = str(lst[0])

                                # Speech module
                                if pygame.mixer.music.get_busy() == False and human_string == i:
                                    name = human_string_filename + ".mp3"

                                # Only get from google if we dont have it
                                if not os.path.isfile(os.path.join('sounds', name)):
                                    tts = gTTS(text=human_string, lang='tr')
                                    tts.save(os.path.join('sounds', name))

                                pygame.mixer.music.load(os.path.join('sounds', name))
                                pygame.mixer.music.play()

                        if (frame_count % 30 == 0):
                            lists.clear()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Nesne Tanima Ses ve Mesafe', f)

# clean everything
cv2.destroyAllWindows()
video.release()
sess.close()





   
