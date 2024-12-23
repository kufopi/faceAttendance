import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import face_record

st.set_page_config(page_title='Attendance System', layout='wide')

# Load authentication configuration
with open('auth_configure.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)


# Define the login function
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


# Define the logout function
def logout():
    authenticator.logout('main')
    st.session_state.clear()
    st.rerun()


# Page configuration


# Login authentication
auth_status, user_name = login()

if auth_status:
    st.header("Attendance System using Face Recognition")
    st.write(f'Welcome {user_name}!')

    # Add a logout button
    if st.button('Logout', key='logout_button'):
        logout()

    with st.spinner("Loading data and Retrieving"):
        import face_record

    st.success("Model Successfully Loaded")
    st.success("Data Retrieval Complete")
elif st.session_state['authentication_status'] == False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] == None:
    st.warning('Please enter your username and password')
