import pandas as pd
import streamlit as st
from home import face_record

st.set_page_config(page_title="Reporting..", layout='wide')
st.subheader("Report Page")


# Retrieve Logs
def load_logs(name='NHS-free-db:logs'):
    loglist = face_record.r.lrange(name, start=0, end=-1)  # Extract all data
    return loglist


# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(['Registered Data', 'Logs', 'Attendance Data', 'Attendance Percentage'])

with tab2:
    if st.button('Refresh Logs'):
        st.write(load_logs())
with tab1:
    if st.button("Refresh Data"):
        with st.spinner('Retrieving Data from Redis DB'):
            redis_face_db = face_record.retrieve_data()
            st.dataframe(redis_face_db[['Name', 'Role', 'Course']])

with tab3:
    st.subheader('Attendance Report')
    # Load logs into attribute
    logs_list = load_logs()
    # Convert the logs from byte to str
    convert_byte_to_str = lambda x: x.decode('utf-8')
    logs_list_string = list(map(convert_byte_to_str, logs_list))

    # Step 2 split string
    split_string = lambda y: y.split('@')
    logs_nest = list(map(split_string, logs_list_string))

    # Convert to df
    logs_df = pd.DataFrame(logs_nest, columns=['Name', 'Role', 'Course', 'Timestamp'])

    # Step 3 time-based analysis
    logs_df['Timestamp'] = pd.to_datetime(logs_df['Timestamp'], format='ISO8601')
    logs_df['Date'] = logs_df['Timestamp'].dt.date

    # Calc In-time and Out-time
    # In time = time which person is first detected in that day
    # Out time = time which person is last detected that day (Max timestamp)
    report_df = logs_df.groupby(by=['Date', 'Name', 'Role', 'Course']).agg(
        In_time=pd.NamedAgg(column='Timestamp', aggfunc='min'),
        Out_time=pd.NamedAgg(column='Timestamp', aggfunc='max')
    ).reset_index()
    report_df['In_time'] = pd.to_datetime(report_df['In_time'])
    report_df['Out_time'] = pd.to_datetime(report_df['Out_time'])
    report_df['Duration'] = report_df['Out_time'] - report_df['In_time']

    # Marking those present and absent
    all_dates = report_df['Date'].unique()
    name_role_course = report_df[['Name', 'Role', 'Course']].drop_duplicates().values.tolist()
    date_name_role_course_zip = []
    for dt in all_dates:
        for name, role, course in name_role_course:
            date_name_role_course_zip.append([dt, name, role, course])
    date_name_role_course_zip_df = pd.DataFrame(date_name_role_course_zip, columns=['Date', 'Name', 'Role', 'Course'])
    # Left join with report_df
    date_name_role_course_zip_df = pd.merge(date_name_role_course_zip_df, report_df, how='left',
                                            on=['Date', 'Name', 'Role', 'Course'])
    # Duration
    date_name_role_course_zip_df['Duration_seconds'] = date_name_role_course_zip_df['Duration'].dt.seconds
    date_name_role_course_zip_df['Duration_hours'] = date_name_role_course_zip_df['Duration_seconds'] / (60 * 60)


    def status(x):
        if pd.Series(x).isnull().all():
            return 'Absent'
        elif x > 0 and x < 1:
            return 'Absent (less than 1 hr)'
        elif x == 0:
            return 'Absent (Pranking the system)'
        else:
            return 'Present'


    date_name_role_course_zip_df['Status'] = date_name_role_course_zip_df['Duration_hours'].apply(status)

    st.dataframe(date_name_role_course_zip_df)

with tab4:
    st.subheader('Attendance Percentage')

    # Ensure all_dates is defined in this scope
    if 'all_dates' not in locals() and 'report_df' in locals():
        all_dates = report_df['Date'].unique()

    if 'report_df' not in locals():
        st.error("No report data found.")
    else:
        # Calculate total classes held for each course by counting the teacher's attendance
        total_classes_held = report_df[report_df['Role'] == 'Teacher'].groupby('Course').size().reset_index(
            name='Total_Classes_Held')

        print("Total Classes Held:")
        print(total_classes_held)

        # Calculate student attendance for each course
        student_attendance = date_name_role_course_zip_df[date_name_role_course_zip_df['Role'] == 'Student'].groupby(['Name', 'Course'])['Status'].apply(lambda x: (x == 'Present').sum()).reset_index(name='Days_Present')

        print("Student Attendance:")
        print(student_attendance)

        # Merge to calculate attendance percentage
        attendance_summary = pd.merge(student_attendance, total_classes_held, on='Course', how='left')
        attendance_summary['Attendance_Percentage'] = (attendance_summary['Days_Present'] / attendance_summary['Total_Classes_Held']) * 100

        print("Attendance Summary:")
        print(attendance_summary)

        st.dataframe(attendance_summary)

