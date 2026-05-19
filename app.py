import streamlit as st
import cv2
import tempfile
import os
import threading
import time
from tracker import VehicleTracker

st.set_page_config(page_title="Vehicle Counting App", layout="wide")

st.markdown("""
<style>
    .reportview-container {
        background: #111111;
        color: white;
    }
    .sidebar .sidebar-content {
        background: #1e1e1e;
    }
    h1 {
        color: #00e5ff;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #00e5ff;
        color: black;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00b3cc;
        box-shadow: 0 4px 15px rgba(0, 229, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.title("🚙 AI Vehicle Counter Pro")
st.markdown("Real-time vehicle detection and counting using YOLOv8.")

# Sidebar Settings
st.sidebar.header("⚙️ Configuration")

# 1. Input Source
input_type = st.sidebar.radio("Input Source", ["Sample Video", "IP Camera (Phone)", "Video File"])

video_path = None
if input_type == "Sample Video":
    video_path = "sample_traffic.mp4"
    st.sidebar.info("Using built-in sample traffic video.")
elif input_type == "Video File":
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
elif input_type == "IP Camera (Phone)":
    # Example: http://192.168.1.100:8080/video
    ip_url = st.sidebar.text_input("IP Camera Stream URL", "http://192.168.1.xxx:8080/video")
    if ip_url:
        video_path = ip_url

# 2. Vehicle Types
target_classes = st.sidebar.multiselect(
    "Select vehicles to count",
    ['car', 'motorcycle', 'bus', 'truck'],
    default=['car', 'motorcycle']
)

# 3. Virtual Line Position
line_position = st.sidebar.slider("Virtual Line Vertical Position (%)", 10, 90, 50)

# Initialize tracker (only once to load model efficiently)
@st.cache_resource
def get_tracker():
    return VehicleTracker()

tracker = get_tracker()
tracker.set_target_classes(target_classes)

class VideoStream:
    """Threaded video capture to always get the latest frame and prevent buffer lag."""
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        # Reduce buffer size for IP cameras to minimize latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.ret, self.frame = ret, frame
            time.sleep(0.01) # Small sleep to prevent high CPU usage in thread

    def read(self):
        return self.ret, self.frame
        
    def isOpened(self):
        return self.cap.isOpened()
        
    def get(self, propId):
        return self.cap.get(propId)

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

start_button = st.sidebar.button("Start Counting")

# Layout
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### Video Feed")
    video_placeholder = st.empty()

with col2:
    st.markdown("### Counters")
    in_metric = st.empty()
    out_metric = st.empty()

if start_button:
    if video_path is None or (input_type == "IP Camera (Phone)" and video_path == "http://192.168.1.xxx:8080/video"):
        st.warning("Please provide a valid video source.")
    else:
        # Use threaded reader for IP Camera to avoid lag, standard for files
        is_live_stream = input_type == "IP Camera (Phone)"
        if is_live_stream:
            cap = VideoStream(video_path)
            st.info("Using Threaded Stream for minimal latency.")
        else:
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Error opening video stream or file")
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # If width/height are 0 (sometimes happens with streams), we'll set defaults later
            
            stop_button = st.button("Stop")
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.info("Video stream ended.")
                    break
                
                if width == 0:
                    height, width, _ = frame.shape
                
                # Calculate actual Y pixel position from percentage
                y_pos = int(height * (line_position / 100.0))
                tracker.set_line(y_pos, width)
                
                # Process frame
                annotated_frame, counts = tracker.process_frame(frame)
                
                # Convert BGR (OpenCV) to RGB (Streamlit)
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                video_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                
                in_metric.metric("Vehicles Crossed (In)", counts['in'])
                out_metric.metric("Vehicles Crossed (Out)", counts['out'])
                
            cap.release()
