Assistive Sign Language Learning System

Overview:

This project aims to develop an assistive sign language learning system for deaf and mute children with Alzheimer's using YOLO object recognition and holistic MediaPipe. The system is designed to identify common objects in real-time and generate corresponding sign language gestures, providing an engaging and effective learning platform.

Features:

Real-time Object Recognition: Utilizes YOLO to detect objects in real-time.
Sign Language Gesture Generation: Maps detected objects to ASL gestures.
User-Friendly Interface: Intuitive controls for interaction.
Testing Module: Evaluates user proficiency in sign language.
Feedback Mechanism: Helps refine the model based on user input.

Technology Stack:

YOLO: For real-time object detection.
MediaPipe Holistic: For gesture recognition.
Flask: For creating a user-friendly web interface.
OpenCV: For image processing and capturing.
Python: The main programming language used for development.

Usage:

Object Detection and Gesture Mapping:
The system captures real-time images using a camera.
YOLO detects objects and maps them to corresponding ASL gestures.
Detected objects and their gestures are displayed on the web interface.

Testing Module:

The system presents an object image to the user.
The user performs the corresponding ASL gesture.
The system evaluates the gesture using MediaPipe Holistic.
If the gesture is incorrect, the system displays the correct gesture.

Execution:
Run app.py for either phases.

Contributing:

We welcome contributions to improve this project.

File access:
All the required files are present in the repository. 
The files on the first page are related to the first phase of the project where YOLO is used to detect an object and provide the sign for it.
The folder "SecondPart" contains all the files for the second phase where the model evaluates the sign performed by user.
Change the paths according to your convinience. It is recommended to store all these files in "D:/FirstPart/" if changing paths is not preferred.
