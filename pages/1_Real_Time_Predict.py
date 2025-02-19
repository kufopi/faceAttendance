from face_record import RealTimer
from home import st
from home import face_record
from streamlit_webrtc import webrtc_streamer
import av
import time
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

st.subheader("Real Time Prediction")

with st.spinner('Retrieving Data from Redis DB'):
    redis_face_db = face_record.retrieve_data()
    st.dataframe(redis_face_db)
st.success('Successful Retrieval')

waitTime = 20  # Time in seconds
setTime = time.time()
realtimePred = face_record.RealTimer()
course = st.selectbox("Select Course", options=("CSC401", "CSC412", "CSC403","CSC433"))

def video_frame_callback(frame):
    global setTime
    img = frame.to_ndarray(format="bgr24")

    predd_img = realtimePred.face_prediction(img, redis_face_db, 'Facial Features', ['Name', 'Role'],course=course, thresh=0.5)
    timenow = time.time()
    difftime = timenow - setTime

    if difftime >= waitTime:
        realtimePred.save_log_redis()
        setTime = time.time()
        print('Saved Data to Redis')

    return av.VideoFrame.from_ndarray(predd_img, format="bgr24")

st.set_page_config(
    page_title="Video Stream",
    layout="wide",
    initial_sidebar_state="expanded"
)

webrtc_streamer(
    key="realtimePrediction",  # or "registration" for registration page
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
            {
                "urls": ["turn:numb.viagenie.ca"],
                "username": "webrtc@live.com",
                "credential": "muazkh"
            }
        ],
        "iceTransportPolicy": "all",
        "bundlePolicy": "max-bundle",
        "rtcpMuxPolicy": "require",
        "iceCandidatePoolSize": 1,
    },
    media_stream_constraints={
        "video": {
            "width": {"min": 320, "ideal": 480, "max": 640},
            "height": {"min": 240, "ideal": 360, "max": 480},
            "frameRate": {"ideal": 30, "max": 30},
        },
        "audio": False,
    },
    async_processing=True,
    video_html_attrs={
        "style": {"width": "100%", "margin": "0 auto", "border": "2px solid red"},
        "controls": False,
        "autoPlay": True,
    },
)
