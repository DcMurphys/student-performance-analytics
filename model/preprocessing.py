# Import necessary libraries 
import pandas as pd
import numpy as np
import joblib
from model import Model

# Load all required pretrained models
# Pretrained encoder models
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

# Pretrained scaler models
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


# Features classified as 'pca_num_cols_1' (principal component dimension 1) 
pca_num_cols_1 = [
    'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_1st_sem_grade',
    'Sem1_approval_rate',
    'Sem1_approved_to_credited_ratio',
    'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_without_evaluations',
    'Sem2_approval_rate',
    'Sem2_approved_to_credited_ratio', 
    'Sem2_without_evaluation',
] 

# Features classified as 'pca_num_cols_2' (principal component dimension 2)
pca_num_cols_2 = [
    'Previous_qualification_grade',
    'Change_units_without_evaluations',
    'Overall_approval_rate',
    'Overall_average_grade',
    'Overall_approved_to_credited_ratio',
    'Grade_vs_prev_qual'
]

# Reordered features (in list format) 
reordered_features = [
    'Marital_status',
    'Application_order',
    'Course',
    'Daytime_evening_attendance',
    'Previous_qualification',
    'Admission_grade',
    'Displaced',
    'Debtor',
    'Tuition_fees_up_to_date',
    'Gender',
    'Scholarship_holder',
    'Age_at_enrollment',
    'Curricular_units_2nd_sem_grade',
    'Sem1_without_evaluation',
    'Pc1_1',
    'Pc1_2',
    'Pc1_3',
    'Pc1_4',
    'Pc2_1',
    'Pc2_2'
]

# Input data preprocessing function
def preprocess_data(df):
    # Calculate first semester approval rate
    df['Sem1_approval_rate'] = df['Curricular_units_1st_sem_approved'] / df['Curricular_units_1st_sem_enrolled'].replace(0, np.nan)
    df['Sem1_approval_rate'] = df['Sem1_approval_rate'].fillna(0)

    # Calculate second semester approval rate
    df['Sem2_approval_rate'] = df['Curricular_units_2nd_sem_approved'] / df['Curricular_units_2nd_sem_enrolled'].replace(0, np.nan)
    df['Sem2_approval_rate'] = df['Sem2_approval_rate'].fillna(0)

    # Create a new feature named 'Overall_approval_rate'
    df['Overall_approval_rate'] = ((df['Sem1_approval_rate'] + df['Sem2_approval_rate']) / 2).replace(0, np.nan)
    df['Overall_approval_rate'] = df['Overall_approval_rate'].fillna(0)

    # Calculate first semester without evaluation
    df['Sem1_without_evaluation'] = df['Curricular_units_1st_sem_without_evaluations'] / df['Curricular_units_1st_sem_enrolled'].replace(0, np.nan)
    df['Sem1_without_evaluation'] = df['Sem1_without_evaluation'].fillna(0)

    # Calculate second semester without evaluation
    df['Sem2_without_evaluation'] = df['Curricular_units_2nd_sem_without_evaluations'] / df['Curricular_units_2nd_sem_enrolled'].replace(0, np.nan)
    df['Sem2_without_evaluation'] = df['Sem2_without_evaluation'].fillna(0)

    # Create a new feature named 'Change_units_without_evaluations'
    df['Change_units_without_evaluations'] = df['Sem2_without_evaluation'] - df['Sem1_without_evaluation']

    # Create a new feature named 'Overall_average_grade' 
    weighted_grade_sum = (df['Curricular_units_1st_sem_grade'] * df['Curricular_units_1st_sem_approved']) + \
                            (df['Curricular_units_2nd_sem_grade'] * df['Curricular_units_2nd_sem_approved'])

    total_approved = df['Curricular_units_1st_sem_approved'] + df['Curricular_units_2nd_sem_approved']

    df['Overall_average_grade'] = weighted_grade_sum / total_approved.replace(0, np.nan)
    df['Overall_average_grade'] = df['Overall_average_grade'].fillna(0)

    # Create a new feature named 'Grade_vs_prev_qual' 
    df['Grade_vs_prev_qual'] = df['Overall_average_grade'] - df['Previous_qualification_grade']

    # Calculate the ratio achieved on first semester
    df['Sem1_approved_to_credited_ratio'] = df['Curricular_units_1st_sem_approved'] / df['Curricular_units_1st_sem_credited'].replace(0, np.nan)
    df['Sem1_approved_to_credited_ratio'] = df['Sem1_approved_to_credited_ratio'].fillna(0)

    # Calculate the ratio achieved on second semester
    df['Sem2_approved_to_credited_ratio'] = df['Curricular_units_2nd_sem_approved'] / df['Curricular_units_2nd_sem_credited'].replace(0, np.nan)
    df['Sem2_approved_to_credited_ratio'] = df['Sem1_approved_to_credited_ratio'].fillna(0)

    # Calculate 'total_approved' and 'total_credited' 
    total_approved = df['Curricular_units_1st_sem_approved'] + df['Curricular_units_2nd_sem_approved']
    total_credited = df['Curricular_units_1st_sem_credited'] + df['Curricular_units_2nd_sem_credited']

    # Create a new feature named 'Overall_approved_to_credited_ratio' 
    df['Overall_approved_to_credited_ratio'] = total_approved / total_credited.replace(0, np.nan)
    df['Overall_approved_to_credited_ratio'] = df['Overall_approved_to_credited_ratio'].fillna(0)

    # Mengembalikan data hasil prapemrosesan 
    return df


# Function for transforming input data  
def transform_data(df, df_test=None):
    # Define numerical features will be scaled 
    Admission_grade = df[["Admission_grade"]]
    Age_at_enrollment = df[["Age_at_enrollment"]]
    Change_units_without_evaluations = df[["Change_units_without_evaluations"]]
    Curricular_units_1st_sem_approved = df[["Curricular_units_1st_sem_approved"]]
    Curricular_units_1st_sem_credited = df[["Curricular_units_1st_sem_credited"]]
    Curricular_units_1st_sem_enrolled = df[["Curricular_units_1st_sem_enrolled"]]
    Curricular_units_1st_sem_evaluations = df[["Curricular_units_1st_sem_evaluations"]]
    Curricular_units_1st_sem_grade = df[["Curricular_units_1st_sem_grade"]]
    Curricular_units_1st_sem_without_evaluations = df[["Curricular_units_1st_sem_without_evaluations"]]
    Curricular_units_2nd_sem_approved = df[["Curricular_units_2nd_sem_approved"]]
    Curricular_units_2nd_sem_credited = df[["Curricular_units_2nd_sem_credited"]]
    Curricular_units_2nd_sem_enrolled = df[["Curricular_units_2nd_sem_enrolled"]]
    Curricular_units_2nd_sem_evaluations = df[["Curricular_units_2nd_sem_evaluations"]]
    Curricular_units_2nd_sem_grade = df[["Curricular_units_2nd_sem_grade"]]
    Curricular_units_2nd_sem_without_evaluations = df[["Curricular_units_2nd_sem_without_evaluations"]]
    Grade_vs_prev_qual = df[["Grade_vs_prev_qual"]]
    Overall_approval_rate = df[["Overall_approval_rate"]]
    Overall_approved_to_credited_ratio = df[["Overall_approved_to_credited_ratio"]]
    Overall_average_grade = df[["Overall_average_grade"]]
    Previous_qualification_grade = df[["Previous_qualification_grade"]]
    Sem1_approval_rate = df[["Sem1_approval_rate"]]
    Sem1_approved_to_credited_ratio = df[["Sem1_approved_to_credited_ratio"]]
    Sem1_without_evaluation = df[["Sem1_without_evaluation"]]
    Sem2_approval_rate = df[["Sem2_approval_rate"]]
    Sem2_approved_to_credited_ratio = df[["Sem2_approved_to_credited_ratio"]]
    Sem2_without_evaluation = df[["Sem2_without_evaluation"]]

    # Scale numerical features with pretrained scaler 
    data = pd.DataFrame()

    data["Admission_grade"] = scaler_Admission_grade.transform(Admission_grade)[0]
    data["Age_at_enrollment"] = scaler_Age_at_enrollment.transform(Age_at_enrollment)[0]
    data["Change_units_without_evaluations"] = scaler_Change_units_without_evaluations.transform(Change_units_without_evaluations)[0]
    data["Curricular_units_1st_sem_approved"] = scaler_Curricular_units_1st_sem_approved.transform(Curricular_units_1st_sem_approved)[0]
    data["Curricular_units_1st_sem_credited"] = scaler_Curricular_units_1st_sem_credited.transform(Curricular_units_1st_sem_credited)[0]
    data["Curricular_units_1st_sem_enrolled"] = scaler_Curricular_units_1st_sem_enrolled.transform(Curricular_units_1st_sem_enrolled)[0]
    data["Curricular_units_1st_sem_evaluations"] = scaler_Curricular_units_1st_sem_evaluations.transform(Curricular_units_1st_sem_evaluations)[0]
    data["Curricular_units_1st_sem_grade"] = scaler_Curricular_units_1st_sem_grade.transform(Curricular_units_1st_sem_grade)[0]
    data["Curricular_units_1st_sem_without_evaluations"] = scaler_Curricular_units_1st_sem_without_evaluations.transform(Curricular_units_1st_sem_without_evaluations)[0]
    data["Curricular_units_2nd_sem_approved"] = scaler_Curricular_units_2nd_sem_approved.transform(Curricular_units_2nd_sem_approved)[0]
    data["Curricular_units_2nd_sem_credited"] = scaler_Curricular_units_2nd_sem_credited.transform(Curricular_units_2nd_sem_credited)[0]
    data["Curricular_units_2nd_sem_enrolled"] = scaler_Curricular_units_2nd_sem_enrolled.transform(Curricular_units_2nd_sem_enrolled)[0]
    data["Curricular_units_2nd_sem_evaluations"] = scaler_Curricular_units_2nd_sem_evaluations.transform(Curricular_units_2nd_sem_evaluations)[0]
    data["Curricular_units_2nd_sem_grade"] = scaler_Curricular_units_2nd_sem_grade.transform(Curricular_units_2nd_sem_grade)[0]
    data["Curricular_units_2nd_sem_without_evaluations"] = scaler_Curricular_units_2nd_sem_without_evaluations.transform(Curricular_units_2nd_sem_without_evaluations)[0]
    data["Grade_vs_prev_qual"] = scaler_Grade_vs_prev_qual.transform(Grade_vs_prev_qual)[0]
    data["Overall_approval_rate"] = scaler_Overall_approval_rate.transform(Overall_approval_rate)[0]
    data["Overall_approved_to_credited_ratio"] = scaler_Overall_approved_to_credited_ratio.transform(Overall_approved_to_credited_ratio)[0]
    data["Overall_average_grade"] = scaler_Overall_average_grade.transform(Overall_average_grade)[0]
    data["Previous_qualification_grade"] = scaler_Previous_qualification_grade.transform(Previous_qualification_grade)[0]
    data["Sem1_approval_rate"] = scaler_Sem1_approval_rate.transform(Sem1_approval_rate)[0]
    data["Sem1_approved_to_credited_ratio"] = scaler_Sem1_approved_to_credited_ratio.transform(Sem1_approved_to_credited_ratio)[0]
    data["Sem1_without_evaluation"] = scaler_Sem1_without_evaluation.transform(Sem1_without_evaluation)[0]
    data["Sem2_approval_rate"] = scaler_Sem2_approval_rate.transform(Sem2_approval_rate)[0]
    data["Sem2_approved_to_credited_ratio"] = scaler_Sem2_approved_to_credited_ratio.transform(Sem2_approved_to_credited_ratio)[0]
    data["Sem2_without_evaluation"] = scaler_Sem2_without_evaluation.transform(Sem2_without_evaluation)[0]

    # Encode the categorical features with pretrained encoder 
    data["Application_order"] = Model.encoder_Application_order.transform(df["Application_order"])
    data["Course"] = Model.encoder_Course.transform(df["Course"])
    data["Daytime_evening_attendance"] = Model.encoder_Daytime_evening_attendance.transform(df["Daytime_evening_attendance"])
    data["Debtor"] = Model.encoder_Debtor.transform(df["Debtor"])
    data["Displaced"] = Model.encoder_Displaced.transform(df["Displaced"])
    data["Gender"] = Model.encoder_Gender.transform(df["Gender"])
    data["Marital_status"] = Model.encoder_Marital_status.transform(df["Marital_status"])
    data["Previous_qualification"] = Model.encoder_Previous_qualification.transform(df["Previous_qualification"])
    data["Scholarship_holder"] = Model.encoder_Scholarship_holder.transform(df["Scholarship_holder"])
    data["Tuition_fees_up_to_date"] = Model.encoder_Tuition_fees_up_to_date.transform(df["Tuition_fees_up_to_date"])

    return data


# Function for PCA reduction on selected features 
def run_pca(df, pca_num_cols_1, pca_num_cols_2, df_test=None):
    df_pca = df.copy()

    # PCA reduction on features classified as 'pca_num_cols_1' 
    df_pca[["Pc1_1", "Pc1_2", "Pc1_3", "Pc1_4"]] = Model.pca_1.transform(df_pca[pca_num_cols_1])
    df_pca.drop(columns=pca_num_cols_1, axis=1, inplace=True)

    # PCA reduction on features classified as 'pca_num_cols_2' 
    df_pca[["Pc2_1", "Pc2_2"]] = Model.pca_2.transform(df_pca[pca_num_cols_2])
    df_pca.drop(columns=pca_num_cols_2, axis=1, inplace=True)
    
    # Reorder features to avoid issues while predicting model 
    df_pca_reordered = df_pca[reordered_features]

    return df_pca_reordered


# Execute the preprocessing.py file 
def __main__(df):
    # Preprocess input data
    df_preprocessed = preprocess_data(df)

    # Transform preprocessed input data with encoder & scaler 
    df_transformed = transform_data(df_preprocessed)

    # Transform preprocessed input data with PCA
    df_pca = run_pca(df_transformed, pca_num_cols_1, pca_num_cols_2)

    # Return transformed input data ready for prediction 
    print(df_pca)
    return df_pca

