import streamlit as st
import  cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer
from face_record import RegistrationForm

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
webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={"video": True, "audio": False})

if st.button('Submit'):
    return_val = registration_form.save_data_in_redis(personName,role)
    if return_val:
        st.success(f'{personName} registered successfully')
    elif return_val =='name_false':
        st.error('Please enter the name, name cannot be empty')
    elif return_val == 'file_false':
        st.error('File face_embedding.txt is not found, kindly refresh page')
