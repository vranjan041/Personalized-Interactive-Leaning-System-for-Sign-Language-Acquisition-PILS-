from flask import Flask, request, Response, send_file, render_template, send_from_directory
import cv2
import numpy as np
import mediapipe as mp
import os
import tensorflow as tf
import pandas as pd
from werkzeug.utils import secure_filename
import subprocess

app = Flask(__name__, static_folder='static')

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/<path:filename>')
def serve_signs(filename):
    return send_from_directory('static', filename)

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
ROWS_PER_FRAME = 543

def create_frame_landmarks_df(results, frame, xyz):
    xyz_skel = xyz[['type', 'landmark_index']].drop_duplicates().reset_index(drop=True).copy()
    
    face = pd.DataFrame()
    pose = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand = pd.DataFrame()

    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    face = face.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='face')
    pose = pose.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='pose')
    left_hand = left_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='left_hand')
    right_hand = right_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='right_hand')

    landmarks = pd.concat([face, pose, left_hand, right_hand]).reset_index(drop=True)
    
    landmarks = xyz_skel.merge(landmarks, on=['type', 'landmark_index'], how='left')
    landmarks = landmarks.assign(frame=frame)
    return landmarks

def process_video(file_path, xyz):
    all_landmarks = []
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error: Couldn't open video file {file_path}")
            return all_landmarks
        
        with mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            frame = 0
            while cap.isOpened():
                frame += 1
                success, image = cap.read()
                if not success:
                    print("End of video reached or error reading frame.")
                    break

                # To improve performance, optionally mark the image as not writeable to pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                landmarks = create_frame_landmarks_df(results, frame, xyz)
                all_landmarks.append(landmarks)

                # Draw landmark annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS
                )
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS
                )

                # Uncomment to see the video frame by frame
                # cv2.imshow('MediaPipe Holistic', image)
                # if cv2.waitKey(5) & 0xFF == 27:
                #     break
        
        cap.release()
    except Exception as e:
        print(f"Error processing video: {e}")
        return all_landmarks
    return all_landmarks

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

@app.route("/detect_video", methods=["POST"])
def detect_video():
    if "video" not in request.files:
        return "No video file uploaded", 400

    video_file = request.files["video"]
    filename = secure_filename(video_file.filename)
    upload_folder = 'D:/FirstPart/Second Part/videos/upload_folder'
    processed_folder = 'D:/FirstPart/Second Part/videos/processed_folder'
    webm_path = os.path.join(upload_folder, filename)
    video_file.save(webm_path)
    
    xyz = pd.read_parquet("D:/FirstPart/Second Part/static/1460359.parquet")
    mp4_filename = filename.rsplit('.', 1)[0] + '.mp4'
    mp4_path = os.path.join(processed_folder, mp4_filename)

    # Convert webm to mp4 using ffmpeg
    subprocess.run(['ffmpeg','-y','-i', webm_path, mp4_path], check=True)

    landmarks = process_video(mp4_path, xyz)
    if not landmarks:
        return "Failed to process video", 500

    landmarks_df = pd.concat(landmarks).reset_index(drop=True)
    landmarks_df.to_parquet('output.parquet')
    
    interpreter = tf.lite.Interpreter("D:/FirstPart/Second Part/new_model.tflite")
    interpreter.allocate_tensors()
    found_signatures = list(interpreter.get_signature_list().keys())
    prediction_fn = interpreter.get_signature_runner("serving_default")

    pq_file = "D:/FirstPart/Second Part/output.parquet"
    xyz_np = load_relevant_data_subset(pq_file)

    prediction = prediction_fn(inputs=xyz_np)
    sign = prediction['outputs'].argmax()

    train = pd.read_csv("D:/FirstPart/Second Part/static/train.csv")
    train['sign_ord'] = train['sign'].astype('category').cat.codes

    SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
    ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

    predicted_sign = ORD2SIGN.get(sign, "Unknown")
    print(f'Predicted sign is {predicted_sign}')

    processed_file_path = f"./static/{predicted_sign}.mp4"
    try:
        return Response(processed_file_path, mimetype="text/plain")
    except FileNotFoundError:
        return Response("Video file not found", status=404)

if __name__ == "__main__":
    app.run(debug=True)
