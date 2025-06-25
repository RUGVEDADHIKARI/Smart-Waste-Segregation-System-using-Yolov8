import settings
import tempfile
import os
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['YOLO_AUTOINSTALL'] = 'False'
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
import PIL
from PIL import Image

@st.cache_resource
def load_model(model_path):
    """Load YOLOv8 model with caching"""
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            st.info("Please ensure your YOLOv8 model file is placed in the correct location.")
            return None
        
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(uploaded_image, model):
    image = np.asarray(uploaded_image.convert('RGB'))
    results = model.predict(
        image,
        imgsz=320,
        conf=settings.CONFIDENCE_THRESHOLD,
        iou=settings.IOU_THRESHOLD,
        stream=False,
        verbose=False
    )
    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(annotated)

def process_video(uploaded_video,model):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        tfile.write(uploaded_video.read())
        tfile.close()
    except Exception as e:
        st.error(f"Error writing to the temporary file:{e}")
    try:
        cap=cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            raise IOError("Cannot open the file")
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps=cap.get(cv2.CAP_PROP_FPS)

        output_path = tfile.name.replace(".mp4", "_output.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        while True:
            ret,frame=cap.read()
            if not ret:
                break
            results=model.predict(
                frame,
                imgsz=320,
                conf=settings.CONFIDENCE_THRESHOLD,
                iou=settings.IOU_THRESHOLD,
                stream=False,
                verbose=False
            )
            annotated_frame=results[0].plot()
            out.write(annotated_frame)
        cap.release()
        out.release()

        return output_path
    except Exception as e:
        st.error("Error Processing the video")

def process_webcam(model,stop_flag):
    st.warning("The webcam is running. Please click on stop to end the session")
    frame_placeholder=st.empty()
    cap=cv2.VideoCapture(settings.WEBCAM_PATH)
    if not cap.isOpened():
        st.error("Unable to access the webcam")
        return 
    try:
        while True:
            ret,frame=cap.read()
            if not ret:
                st.error("Failed reading from the web")
                break
            results=model.predict(
                frame,
                imgsz=320,
                conf=settings.CONFIDENCE_THRESHOLD,
                iou=settings.IOU_THRESHOLD,
                stream=False,
                verbose=False
            )
            annotated_frame=results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_frame,channels="RGB")

            if stop_flag():
                break
    finally:
        cap.release()
        st.success("Webcam stopped")




