from pathlib import Path
from PIL import Image
import os
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['YOLO_AUTOINSTALL'] = 'False'
import streamlit as st
import ultralytics
import settings
import helper


st.set_page_config(
    page_title="Smart Waste Segregation System Using Yolov8",
    page_icon="🤖"
)

st.title("Smart Waste Segregation System Using Yolov8")

st.sidebar.header("Selection Options")
add_image = st.sidebar.checkbox("Image")
add_video = st.sidebar.checkbox("Video")
add_webcam = st.sidebar.checkbox("Webcam")

if not (add_image or add_video or add_webcam):
    st.markdown("<h4>To start detection please click on the checkbox in the sidebar</h4>", unsafe_allow_html=True)
model_path = settings.MODEL_PATH  

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to find the model. Check specified path: {model_path}")
    st.error(ex)

if add_image:
    uploaded_image = st.sidebar.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
if add_video:
    uploaded_video = st.sidebar.file_uploader("Upload your video", type=["mp4", "avi", "mov"])
if add_webcam:
    start_webcam = st.checkbox("Start Webcam")
    stop_webcam = st.checkbox("Stop Webcam")
    def stop_flag():
        return stop_webcam
    if start_webcam and not stop_webcam:
        helper.process_webcam(model,stop_flag)

if add_image:
    if uploaded_image is not None:
        image=Image.open(uploaded_image)
        st.image(image,use_column_width=True)
        results_img=helper.process_image(image,model)
        st.image(results_img,use_column_width=True)

if add_video:
    if uploaded_video is not None:
        st.video(uploaded_video,format="video/mp4")
        with st.spinner("Processing video..."):
            try:
                processed_path=helper.process_video(uploaded_video,model)
                st.write("Processing complete")
                st.video(processed_path)
            except Exception as e:
                st.error("Error processing the video")
    

        


