import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Load configuration from yaml file
with open('auth_configure.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

def login():
    name, authentication_status, username = authenticator.login('main')
    if 'authentication_status' not in st.session_state:
        st.session_state['authentication_status'] = authentication_status
    if authentication_status:
        return True, name
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')
    return False, None

def logout():
    authenticator.logout('main')
    st.session_state['authentication_status'] = None
    st.experimental_rerun()
