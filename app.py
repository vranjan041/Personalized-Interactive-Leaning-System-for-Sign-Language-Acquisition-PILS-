from flask import Flask, render_template, request, Response, jsonify,json
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import base64
import random

app = Flask(__name__)


model = YOLO("yolov8n.pt")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect_objects", methods=["POST"])
def detect_objects_route():
    data = request.get_json()
    image_data = data["image_data"].split(",")[1]
    decoded_data = base64.b64decode(image_data)
    frame = cv2.imdecode(np.frombuffer(decoded_data, dtype=np.uint8), cv2.IMREAD_COLOR)

    custom = ["apple", "banana", "dog", "umbrella", "bottle", "orange"]
    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_ultralytics(result)
    detected_classes = [
        detections.data["class_name"][i]
        for i in range(len(detections.confidence)) if (detections.data["class_name"][i] in custom)
    ]
    if detected_classes:
        video_path = f"./signs/{detected_classes[0]}/{detected_classes[0]}.mp4"
        try:
            with open(video_path, 'rb') as video_file:
                video_content = video_file.read()
            return Response(video_content, mimetype="video/mp4")
        except FileNotFoundError:
            return Response("Video file not found", status=404)
    else:
        return Response("", mimetype="video/mp4") 
# @app.route("/video_feed")
# def video_feed():
#     return Response(
#         detect_objects_route(), mimetype="multipart/x-mixed-replace; boundary=frame"
#     ) 
if __name__ == "__main__":
    app.run(debug=True)