import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO, solutions

# Initialize Streamlit
st.title("Object Detection and Counting with YOLOv8")

# Define video path
video_path = r'D:\Deep Learning Projects\YOLOv8 Project\Object Counting\Pedestrian_counting\vid1.mp4'

# Define line points (adjust as necessary)
line_points = [(0, 400), (1918, 400)]

# Use a session state to track whether the video has been processed
if 'processed_video_path' not in st.session_state:
    st.session_state.processed_video_path = None

def process_video(input_path):
    model = YOLO(r'D:\Deep Learning Projects\YOLOv8 Project\Object Counting\Pedestrian_counting\detect\train\weights\best.pt')

    counter = solutions.ObjectCounter(
        view_img=False,
        reg_pts=line_points,
        names=model.names,
        draw_tracks=True,
        line_thickness=2,
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Error reading video file")
        return None

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Use a named temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        output_path = temp_file.name

    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break
        tracks = model.track(im0, persist=True, show=False)
        im0 = counter.start_counting(im0, tracks)
        video_writer.write(im0)

    cap.release()
    video_writer.release()

    return output_path

# Display the input video
st.video(video_path, format="video/mp4")

# Button to start processing the video
if st.button('Process Video'):
    if st.session_state.processed_video_path is None:
        # Process video only if not already processed
        with st.spinner('Processing video...'):
            st.session_state.processed_video_path = process_video(video_path)
    
    if st.session_state.processed_video_path:
        # Provide a download button for the output video
        with open(st.session_state.processed_video_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
        
        # Clean up temporary files
        os.remove(st.session_state.processed_video_path)
        st.session_state.processed_video_path = None
