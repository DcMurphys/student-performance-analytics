# Import necessary libraries 
import streamlit as st
import pandas as pd
import numpy as np
from controller import Controller


# Define 'Application_order' in list form
Application_order_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Define 'Marital_status_map' in dict form 
Marital_status_map = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto Union",
    6: "Legally Separated"
}

# Define 'Gender_map' in dict form 
Gender_map = {0: "Female", 1: "Male"}
Gender_options_values = list(Gender_map.keys())

# Define 'Daytime_evening_attendance' in dict form
Attendance_map = {0: "Evening", 1: "Daytime"}
Attendance_options_values = list(Attendance_map.keys()) 

# Define 'Course' in dict form 
Course_map = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)"
}

# Define 'Previous_qualification' in dict form 
Previous_qualification_map = {
    1: "Secondary education",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree", 
    4: "Higher education - master's",
    5: "Higher education - doctorate", 
    6: "Frequency of higher education",
    9: "12th year of schooling - not completed",
    10: "11th year of schooling - not completed",
    12: "Other - 11th year of schooling",
    14: "10th year of schooling",
    15: "10th year of schooling - not completed",
    19: "Basic education 3rd cycle (9th/10th/11th year) or equiv.", 
    38: "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    42: "Professional higher technical course",
    43: "Higher education - master (2nd cycle)"
}

# Define 'Yes_no_options' in dict form 
yes_no_options_values = [1, 0]
yes_no_display_map = {1: "Yes", 0: "No"}

# Function to get interactive message of final status 
def get_status_msg(status):
    if status == 'Graduate':
        return st.success('Congratulations! You have been graduated.', icon="🎓") 
    elif status == 'Dropout':
        return st.error('It is unfortunate that you have been dropout...', icon="😓")
    elif status == 'Enrolled':
        return st.info('You are still enrolled in this institution.', icon="ℹ️")
    else:
        return st.exception(f'{Controller.predict_data(None)}')


# Function to create radio option 
def create_radio(label, session_state_key, value_map, default_value=None):
    if session_state_key not in st.session_state.data:
        st.session_state.data[session_state_key] = default_value
    current_value = st.session_state.data.get(session_state_key)
    options_values = list(value_map.keys())
    default_index = 0 
    try:
        if current_value in options_values:
             default_index = options_values.index(current_value)
    except ValueError:
        pass 

    def format_option(option_value):
        return value_map.get(option_value, f"Invalid: {option_value}")

    selected_value = st.radio(
        label=label,
        options=options_values,
        index=default_index,
        format_func=format_option,
        horizontal=True
    )

    st.session_state.data[session_state_key] = selected_value

    return selected_value


# Function to create selectbox option
def create_selectbox(label, session_state_key, value_map, default_value=None):
    if session_state_key not in st.session_state.data:
        st.session_state.data[session_state_key] = default_value
    current_value = st.session_state.data.get(session_state_key)
    options_values = list(value_map.keys())
    default_index = 0 
    try:
        if current_value in options_values:
             default_index = options_values.index(current_value)
    except ValueError:
        pass 

    def format_option(option_value):
        return value_map.get(option_value, f"Invalid value: {option_value}")

    selected_value = st.selectbox(
        label=label,
        options=options_values,
        index=default_index,
        format_func=format_option,
    )

    st.session_state.data[session_state_key] = selected_value

    return selected_value


# Function to format 'Application_order' option 
def format_app_order(option):
    return str(option)