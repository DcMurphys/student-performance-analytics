# Import necessary libraries 
import streamlit as st
import pandas as pd
from controller import Controller
from view import helper 


# Execute the 'view.py' 
def __main__(): 
    # Page title 
    st.title('Student Status Prediction App')

    # Page description
    st.text('''Welcome to the Large Language Institute Student Status Prediction App! Get valuable insight into your academic journey and likely final status, whether you're on track to graduate. Simply input your past semester grades, completed units, previous qualification scores, and admission test results. In moments, receive a clear picture of your potential graduation status here at Large Language Institute. This powerful tool is built using advanced machine learning models, meticulously trained and rigorously tested on high-quality, validated student data. Leverage cutting-edge technology to understand your path forward and stay informed about your graduation prospects! Try it now and gain clarity on your future at Large Language Institute. 
    ''')

    st.write("---")

    if 'data' not in st.session_state:
        st.session_state.data = {}
        
    # Page form 
    with st.container():
        # Student profile section 
        st.header("Student profile")
        st.text("Please enter your student profile first before continue proceeding.")
        st.session_state.data['Age_at_enrollment'] = int(st.number_input(label='Age during enrollment', value=st.session_state.data.get('Age_at_enrollment', 19)))
        st.caption("The age must be between 15 - 100 years old")

        st.session_state.data['Gender'] = helper.create_radio(
            label="Gender",
            session_state_key='Gender',
            value_map=helper.Gender_map,
            default_value=1 
            )

        st.session_state.data['Marital_status'] = helper.create_selectbox(
            label="Marital status",
            session_state_key='Marital_status',
            value_map=helper.Marital_status_map,
            default_value=1
            )

        st.session_state.data['Displaced'] = helper.create_radio(
            label="Displaced?",
            session_state_key='Displaced',
            value_map=helper.yes_no_display_map,
            default_value=0 
            )

        st.session_state.data['Debtor'] = helper.create_radio(
            label="Debtor?",
            session_state_key='Debtor',
            value_map=helper.yes_no_display_map,
            default_value=0 
            )

        st.session_state.data['Scholarship_holder'] = helper.create_radio(
            label="Scholarship holder?",
            session_state_key='Scholarship_holder',
            value_map=helper.yes_no_display_map,
            default_value=0 
            )

        st.session_state.data['Tuition_fees_up_to_date'] = helper.create_radio(
            label="Tuition fees up to date?",
            session_state_key='Tuition_fees_up_to_date',
            value_map=helper.yes_no_display_map,
            default_value=1 
            )

        st.session_state.data['Application_order'] = st.segmented_control(
            label='Application order', 
            options=helper.Application_order_list, 
            format_func=helper.format_app_order, 
            selection_mode='single', 
            default=6
            )
        st.caption("The choice must be from 0 (first choice) to 9 (last choice)")

        st.session_state.data['Previous_qualification'] = helper.create_selectbox(
            label="Previous qualification",
            session_state_key='Previous_qualification',
            value_map=helper.Previous_qualification_map,
            default_value=1
            )
        
        left, right = st.columns(2, vertical_alignment="bottom")
        with left:
            st.session_state.data['Admission_grade'] = st.number_input(label='Admission grade (0 - 200)', value=st.session_state.data.get('Admission_grade', 122))
            st.caption("Admission grade out of 200 scale")

        with right:
            st.session_state.data['Previous_qualification_grade'] = st.number_input(
                label='Previous qualification grade (0 - 200)', 
                value=st.session_state.data.get('Previous_qualification_grade', 125)
                )
            st.caption("Previous qualification grade out of 200 scale") 

        st.write("---")

        # Course performance assessment section 
        st.header("Course performance assessment")
        st.text("Please input your grade on each semester, including your acquired curricular units.") 

        st.session_state.data['Course'] = helper.create_selectbox(
            label="Course",
            session_state_key='Course',
            value_map=helper.Course_map,
            default_value=9773
            )

        st.session_state.data['Daytime_evening_attendance'] = helper.create_radio(
            label="Daytime/evening attendance",
            session_state_key='Daytime_evening_attendance', 
            value_map=helper.Attendance_map,
            default_value=1
            )
        

        # First semester
        st.subheader("Semester 1")
        left, right = st.columns(2, vertical_alignment="bottom")
        with left: 
            st.session_state.data['Curricular_units_1st_sem_credited'] = st.number_input(
                label='Credited curricular units (1st)', 
                value=st.session_state.data.get('Curricular_units_1st_sem_credited', 0)
                )

        with right:
            st.session_state.data['Curricular_units_1st_sem_enrolled'] = st.number_input(
                label='Enrolled curricular units (1st)', 
                value=st.session_state.data.get('Curricular_units_1st_sem_enrolled', 6)
                )

        left, right = st.columns(2, vertical_alignment="bottom")
        with left:
            st.session_state.data['Curricular_units_1st_sem_evaluations'] = st.number_input(
                label='Evaluated curricular units (1st)', 
                value=st.session_state.data.get('Curricular_units_1st_sem_evaluations', 7)
                )

        with right: 
            st.session_state.data['Curricular_units_1st_sem_approved'] = st.number_input(
                label='Approved curricular units (1st)', 
                value=st.session_state.data.get('Curricular_units_1st_sem_approved', 5)
                )

        left, right = st.columns(2, vertical_alignment="bottom")
        with left:
            st.session_state.data['Curricular_units_1st_sem_grade'] = st.number_input(
                label='Grade of curricular units (1st)', 
                value=st.session_state.data.get('Curricular_units_1st_sem_grade', 14)
                )

        with right:
            st.session_state.data['Curricular_units_1st_sem_without_evaluations'] = st.number_input(
                label='Unevaluated curricular units (1st)', 
                value=st.session_state.data.get('Curricular_units_1st_sem_without_evaluations', 0)
                )


        # Second semester
        st.subheader("Semester 2")
        left, right = st.columns(2, vertical_alignment="bottom")
        with left:
            st.session_state.data['Curricular_units_2nd_sem_credited'] = st.number_input(
                label='Credited curricular units (2nd)', 
                value=st.session_state.data.get('Curricular_units_2nd_sem_credited', 0)
                )

        with right:
            st.session_state.data['Curricular_units_2nd_sem_enrolled'] = st.number_input(
                label='Enrolled curricular units (2nd)', 
                value=st.session_state.data.get('Curricular_units_2nd_sem_enrolled', 6)
                )

        left, right = st.columns(2, vertical_alignment="bottom")
        with left:
            st.session_state.data['Curricular_units_2nd_sem_evaluations'] = st.number_input(
                label='Evaluated curricular units (2nd)', 
                value=st.session_state.data.get('Curricular_units_2nd_sem_evaluations', 8)
                )

        with right:
            st.session_state.data['Curricular_units_2nd_sem_approved'] = st.number_input(
                label='Approved curricular units (2nd)', 
                value=st.session_state.data.get('Curricular_units_2nd_sem_approved', 5)
                )

        left, right = st.columns(2, vertical_alignment="bottom")
        with left:
            st.session_state.data['Curricular_units_2nd_sem_grade'] = st.number_input(
                label='Grade of curricular units (2nd)', 
                value=st.session_state.data.get('Curricular_units_2nd_sem_grade', 13)
                )

        with right:
            st.session_state.data['Curricular_units_2nd_sem_without_evaluations'] = st.number_input(
                label='Unevaluated curricular units (2nd)', 
                value=st.session_state.data.get('Curricular_units_2nd_sem_without_evaluations', 0)
                )


    # Display inputted data in a dataframe format with expander layout
    with st.expander("View your inputted data"):
        display_data = pd.DataFrame([st.session_state.data])
        st.dataframe(data=display_data, width=1920, height=96)


    # Interactive button to predict student's final status based on given data 
    if st.button("Predict your data"):
        student_data = pd.DataFrame([st.session_state.data])
        pred_status = Controller.predict_data(student_data)

        with st.expander("View the preprocessed data"):
            st.dataframe(data=Controller.preprocess_data(student_data), width=1920, height=96)
        
        # Display the interactive messaged according to the final status 
        helper.get_status_msg(pred_status)
