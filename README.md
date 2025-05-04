🎯 Voice Distance and Object Detection
This project combines real-time object detection, approximate distance estimation, and voice alerts into a single application. 
It uses a TensorFlow-based custom-trained model to detect objects from a webcam feed, estimates how close they are, and plays audio warnings when they're too near.

🚀 Features
Object detection from webcam video (e.g., cars, people)

Lane or line detection using Hough Transform

Approximate distance calculation

Audio alerts in Turkish using gTTS when objects are too close

🛠 Requirements
Python 3

TensorFlow >= 1.10

OpenCV

gTTS

pygame

PIL (Pillow)

📦 Model Files
Make sure the following files are in place:

frozen_inference_graph.pb inside the inference_graph folder

labelmap.pbtxt inside the training folder

💡 Notes
This project was supported by TÜBİTAK 2209-A program in Turkey and was presented as a conference paper at two different conferences. 
If you would like to access the publications, feel free to contact me.
