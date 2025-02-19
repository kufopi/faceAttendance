import streamlit as st
import  cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer
from face_record import RegistrationForm
import logging
logging.basicConfig(level=logging.DEBUG)

st.set_page_config(page_title="Registration Form..",layout='wide')
st.subheader("Registration Page")

# Initialize Registration form
registration_form = RegistrationForm()

# Collect person name &role
personName =st.text_input(label='Name',placeholder='First & last name')
role = st.selectbox(label='Select a role',options=('Student','Teacher'))


def video_callback_func(frame):
    img = frame.to_ndarray(format='bgr24')
    reg_img, embedding = registration_form.get_embeddings(img)

    # 1st Save data into Local Computer
    if embedding is not None:
        with open('face_embedding.txt',mode='ab') as f:
            np.savetxt(f,embedding)

    return av.VideoFrame.from_ndarray(reg_img,format='bgr24')
webrtc_streamer(
    key="realtimePrediction",  # or "registration" for registration page
    video_frame_callback=video_callback_func,
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

if st.button('Submit'):
    return_val = registration_form.save_data_in_redis(personName,role)
    if return_val:
        st.success(f'{personName} registered successfully')
    elif return_val =='name_false':
        st.error('Please enter the name, name cannot be empty')
    elif return_val == 'file_false':
        st.error('File face_embedding.txt is not found, kindly refresh page')
