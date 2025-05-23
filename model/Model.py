# Import all required libraries 
import joblib
import pandas as pd
import numpy as np

# Load all required models
# Encoder models
encoder_Application_order = joblib.load("model/encoder/encoder_Application_order.joblib")
encoder_Course = joblib.load("model/encoder/encoder_Course.joblib")
encoder_Daytime_evening_attendance = joblib.load("model/encoder/encoder_Daytime_evening_attendance.joblib")
encoder_Debtor = joblib.load("model/encoder/encoder_Debtor.joblib")
encoder_Displaced = joblib.load("model/encoder/encoder_Displaced.joblib")
encoder_Gender = joblib.load("model/encoder/encoder_Gender.joblib")
encoder_Marital_status = joblib.load("model/encoder/encoder_Marital_status.joblib")
encoder_Previous_qualification = joblib.load("model/encoder/encoder_Previous_qualification.joblib")
encoder_Scholarship_holder = joblib.load("model/encoder/encoder_Scholarship_holder.joblib")
encoder_target = joblib.load("model/encoder/encoder_target.joblib")
encoder_Tuition_fees_up_to_date = joblib.load("model/encoder/encoder_Tuition_fees_up_to_date.joblib")

# Scaler models
scaler_Admission_grade = joblib.load("model/scaler/scaler_Admission_grade.joblib")
scaler_Age_at_enrollment = joblib.load("model/scaler/scaler_Age_at_enrollment.joblib")
scaler_Change_units_without_evaluations = joblib.load("model/scaler/scaler_Change_units_without_evaluations.joblib")
scaler_Curricular_units_1st_sem_approved = joblib.load("model/scaler/scaler_Curricular_units_1st_sem_approved.joblib")
scaler_Curricular_units_1st_sem_credited = joblib.load("model/scaler/scaler_Curricular_units_1st_sem_credited.joblib")
scaler_Curricular_units_1st_sem_enrolled = joblib.load("model/scaler/scaler_Curricular_units_1st_sem_evaluations.joblib")
scaler_Curricular_units_1st_sem_evaluations = joblib.load("model/scaler/scaler_Curricular_units_1st_sem_evaluations.joblib")
scaler_Curricular_units_1st_sem_grade = joblib.load("model/scaler/scaler_Curricular_units_1st_sem_grade.joblib")
scaler_Curricular_units_1st_sem_without_evaluations = joblib.load("model/scaler/scaler_Curricular_units_1st_sem_without_evaluations.joblib")
scaler_Curricular_units_2nd_sem_approved = joblib.load("model/scaler/scaler_Curricular_units_2nd_sem_approved.joblib")
scaler_Curricular_units_2nd_sem_credited = joblib.load("model/scaler/scaler_Curricular_units_2nd_sem_credited.joblib")
scaler_Curricular_units_2nd_sem_enrolled = joblib.load("model/scaler/scaler_Curricular_units_2nd_sem_evaluations.joblib")
scaler_Curricular_units_2nd_sem_evaluations = joblib.load("model/scaler/scaler_Curricular_units_2nd_sem_evaluations.joblib")
scaler_Curricular_units_2nd_sem_grade = joblib.load("model/scaler/scaler_Curricular_units_2nd_sem_grade.joblib")
scaler_Curricular_units_2nd_sem_without_evaluations = joblib.load("model/scaler/scaler_Curricular_units_2nd_sem_without_evaluations.joblib")
scaler_Grade_vs_prev_qual = joblib.load("model/scaler/scaler_Grade_vs_prev_qual.joblib")
scaler_Overall_approval_rate = joblib.load("model/scaler/scaler_Overall_approval_rate.joblib")
scaler_Overall_approved_to_credited_ratio = joblib.load("model/scaler/scaler_Overall_approved_to_credited_ratio.joblib")
scaler_Overall_average_grade = joblib.load("model/scaler/scaler_Overall_average_grade.joblib")
scaler_Previous_qualification_grade = joblib.load("model/scaler/scaler_Previous_qualification_grade.joblib")
scaler_Sem1_approval_rate = joblib.load("model/scaler/scaler_Previous_qualification_grade.joblib")
scaler_Sem1_approved_to_credited_ratio = joblib.load("model/scaler/scaler_Sem1_approved_to_credited_ratio.joblib")
scaler_Sem1_without_evaluation = joblib.load("model/scaler/scaler_Sem1_without_evaluation.joblib")
scaler_Sem2_approval_rate = joblib.load("model/scaler/scaler_Sem2_approval_rate.joblib")
scaler_Sem2_approved_to_credited_ratio = joblib.load("model/scaler/scaler_Sem2_approved_to_credited_ratio.joblib")
scaler_Sem2_without_evaluation = joblib.load("model/scaler/scaler_Sem2_without_evaluation.joblib")

# PCA models
pca_1 = joblib.load("model/pca/pca_1.joblib")
pca_2 = joblib.load("model/pca/pca_2.joblib")

# Pretrained models 
rf_model = joblib.load("model/algorithm/random_forest.joblib")