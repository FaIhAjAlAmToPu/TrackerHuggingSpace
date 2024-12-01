import cv2
import tempfile
import streamlit as st
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Streamlit page settings
st.title("YOLOv8 Object Tracking with DeepSORT")

# File uploader for video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# Initialize YOLOv8 and DeepSORT
model = YOLO('yolo11n.pt')  # Replace with your model path
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2)
track_history = defaultdict(lambda: [])

# Check if a video has been uploaded
if uploaded_file is not None:
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Open video with OpenCV
    cap = cv2.VideoCapture(tfile.name)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a temporary file for the output video
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
    out = cv2.VideoWriter(output_file.name, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLOv8 model detection on the frame
            results = model.predict(frame, show_labels=False, show_conf=False)

            # Ensure results are not None and boxes are available
            if results and results[0].boxes:
                boxes = results[0].boxes.xywh.cpu().numpy()  # YOLO bounding boxes
                confs = results[0].boxes.conf.cpu().numpy()  # YOLO confidences
                classes = results[0].boxes.cls.cpu().numpy()  # YOLO class IDs

                # Filter detections to keep only 'head' class if required
                head_detections = [(boxes[i], confs[i]) for i in range(len(classes))]

                # Update DeepSORT tracker
                tracked_objects = tracker.update_tracks(head_detections, frame=frame)

                # Visualize the tracking results on the frame
                for track in tracked_objects:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    bbox = track.to_tlbr()  # Get bounding box
                    x1, y1, x2, y2 = [int(i) for i in bbox]

                    # Track the center point of the bounding box
                    track_history[track_id].append(((x1 + x2) / 2, (y1 + y2) / 2))

                    # Retain the last 30 tracks for each object
                    if len(track_history[track_id]) > 30:
                        track_history[track_id].pop(0)

                    points = np.array(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                # Write the frame to the output video
                out.write(frame)
            else:
                out.write(frame)
        else:
            break

    # Release resources
    cap.release()
    out.release()

    # Display the video after processing
    st.video(output_file.name)
else:
    st.write("Please upload a video file to start tracking.")
