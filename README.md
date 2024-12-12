Assistive Sign Language Learning System
Overview
This project focuses on developing an assistive sign language learning system designed specifically for deaf and mute children. The system combines YOLO object detection and MediaPipe Holistic technologies to identify common objects in real-time and generate corresponding American Sign Language (ASL) gestures. By providing an interactive and adaptive learning platform, this system aims to make sign language education more engaging and effective.

Key Features
Real-time Object Recognition

Uses YOLO to detect objects in real-time.
Sign Language Gesture Mapping

Maps detected objects to their corresponding ASL gestures for easy learning.
Interactive User Interface

Built with Flask, the interface offers intuitive controls for user interaction.
Testing and Evaluation Module

Evaluates the user's ASL gesture performance and provides immediate feedback.
Feedback Mechanism

Allows users to refine the model by offering input, enhancing accuracy over time.
Technology Stack
YOLO: For real-time object detection.
MediaPipe Holistic: For tracking and evaluating user gestures.
Flask: To create an intuitive web-based user interface.
OpenCV: For image capture and processing.
Python: Core programming language for development.
System Workflow
Phase 1: Object Detection and Gesture Mapping
The system captures real-time images using a connected camera.
YOLO identifies objects in the camera feed.
Detected objects are mapped to corresponding ASL gestures.
The object name and its ASL gesture are displayed on the user-friendly web interface.
Phase 2: Gesture Testing and Feedback
The system displays an object image to the user.
The user performs the corresponding ASL gesture.
The system evaluates the gesture using MediaPipe Holistic.
If the gesture is incorrect:
The system displays the correct gesture for guidance.
Usage Instructions
Running the System
Object Detection and Gesture Mapping (Phase 1)

Navigate to the appropriate directory for Phase 1 (default: D:/FirstPart/).
Run the app.py script to launch the object detection module.
Gesture Testing and Feedback (Phase 2)

Navigate to the SecondPart directory.
Run the corresponding app.py file to launch the testing module.
Contributing
We welcome contributions to enhance the system’s features and functionality. Suggestions for improving gesture recognition accuracy, UI design, and feedback mechanisms are highly encouraged.

File Organization
FirstPart: Contains files for Phase 1 (Object Detection and Gesture Mapping).
SecondPart: Contains files for Phase 2 (Gesture Testing and Feedback).
Customization
Default paths are set to D:/FirstPart/. If you prefer to keep files elsewhere, update the paths in the code accordingly.
Conclusion
This system bridges the gap between object recognition and sign language learning, creating a dynamic platform tailored to the unique needs of children with disabilities. Through innovative technologies like YOLO and MediaPipe, it fosters an accessible and inclusive learning environment.
