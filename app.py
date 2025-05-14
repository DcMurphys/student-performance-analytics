# *****   BACK-END SIDE OF 'APPS.PY'   ***** 
# --- Import library yang diperlukan --- 
# Library bawaan 
import warnings
from IPython.display import display 
warnings.filterwarnings('ignore') # Digunakan untuk menghilangkan warning message saat menjalankan beberapa kode

# Library pihak ketiga untuk pengolahan data dan operasi numerik
import numpy as np
import pandas as pd

# Library pihak ketiga untuk menjalankan prototype sistem machine learning 
import streamlit as st
import joblib
from sklearn.decomposition import PCA 
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# --- Instansiasi variabel-variabel untuk menyimpan data input --- 
# Variabel list Application_order
Application_order_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Variabel dictionary Marital_status
Marital_status_map = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto Union",
    6: "Legally Separated"
}

# Variabel dictionary Gender
Gender_map = {0: "Female", 1: "Male"}
Gender_options_values = list(Gender_map.keys())

# Variabel dictionary Daytime_Evening_Attendance
Attendance_map = {0: "Evening", 1: "Daytime"}
Attendance_options_values = list(Attendance_map.keys()) 

# Variabel dictionary Course
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

# Variabel dictionary Previous_qualification 
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

# Variabel dictionary Yes or No
yes_no_options_values = [1, 0]
yes_no_display_map = {1: "Yes", 0: "No"}


# --- Fungsi untuk melakukan prapemrosesan data input  ---
def preprocess_data(data):
    # --- Membuat fitur untuk menghitung approval rate masing-masing semester ---
    # Menghitung approval rate semester 1
    data['Sem1_approval_rate'] = data['Curricular_units_1st_sem_approved'] / data['Curricular_units_1st_sem_enrolled'].replace(0, np.nan)
    data['Sem1_approval_rate'] = data['Sem1_approval_rate'].fillna(0)

    # Menghitung approval rate semester 2
    data['Sem2_approval_rate'] = data['Curricular_units_2nd_sem_approved'] / data['Curricular_units_2nd_sem_enrolled'].replace(0, np.nan)
    data['Sem2_approval_rate'] = data['Sem2_approval_rate'].fillna(0)

    # --- Membuat kolom bernama 'Overall_approval_rate' --- 
    data['Overall_approval_rate'] = ((data['Sem1_approval_rate'] + data['Sem2_approval_rate']) / 2).replace(0, np.nan)
    data['Overall_approval_rate'] = data['Overall_approval_rate'].fillna(0)

    # --- Membuat kolom bernama 'Change_units_without_evaluation' ---
    # Menghitung semester units without evaluation untuk semester 1
    data['Sem1_without_evaluation'] = data['Curricular_units_1st_sem_without_evaluations'] / data['Curricular_units_1st_sem_enrolled'].replace(0, np.nan)
    data['Sem1_without_evaluation'] = data['Sem1_without_evaluation'].fillna(0)

    # Menghitung semester units without evaluation untuk semester 1
    data['Sem2_without_evaluation'] = data['Curricular_units_2nd_sem_without_evaluations'] / data['Curricular_units_2nd_sem_enrolled'].replace(0, np.nan)
    data['Sem2_without_evaluation'] = data['Sem2_without_evaluation'].fillna(0)

    # Membuat kolom bernama 'Change_units_without_evaluations'
    data['Change_units_without_evaluations'] = data['Sem2_without_evaluation'] - data['Sem1_without_evaluation']

    # --- Membuat kolom bernama 'Overall_average_grade' --- 
    weighted_grade_sum = (data['Curricular_units_1st_sem_grade'] * data['Curricular_units_1st_sem_approved']) + \
                            (data['Curricular_units_2nd_sem_grade'] * data['Curricular_units_2nd_sem_approved'])

    total_approved = data['Curricular_units_1st_sem_approved'] + data['Curricular_units_2nd_sem_approved']

    data['Overall_average_grade'] = weighted_grade_sum / total_approved.replace(0, np.nan)
    data['Overall_average_grade'] = data['Overall_average_grade'].fillna(0)

    # --- Membuat kolom bernama 'Grade_vs_prev_qual' --- 
    data['Grade_vs_prev_qual'] = data['Overall_average_grade'] - data['Previous_qualification_grade']

    # --- Membuat fitur untuk menghitung Overall Approved to Credited Ratio  --- 
    # Menghitung rasio untuk semester 1
    data['Sem1_approved_to_credited_ratio'] = data['Curricular_units_1st_sem_approved'] / data['Curricular_units_1st_sem_credited'].replace(0, np.nan)
    data['Sem1_approved_to_credited_ratio'] = data['Sem1_approved_to_credited_ratio'].fillna(0)

    # Menghitung rasio untuk semester 2
    data['Sem2_approved_to_credited_ratio'] = data['Curricular_units_2nd_sem_approved'] / data['Curricular_units_2nd_sem_credited'].replace(0, np.nan)
    data['Sem2_approved_to_credited_ratio'] = data['Sem1_approved_to_credited_ratio'].fillna(0)

    # Menghitung 'total_approved' dan 'total_credited'
    total_approved = data['Curricular_units_1st_sem_approved'] + data['Curricular_units_2nd_sem_approved']
    total_credited = data['Curricular_units_1st_sem_credited'] + data['Curricular_units_2nd_sem_credited']

    # Menghitung 'Overall_approved_to_credited_ratio' 
    data['Overall_approved_to_credited_ratio'] = total_approved / total_credited.replace(0, np.nan)
    data['Overall_approved_to_credited_ratio'] = data['Overall_approved_to_credited_ratio'].fillna(0)

    # Mengembalikan data hasil prapemrosesan 
    return data


# --- Fungsi untuk mendapatkan data pelatihan yang sudah melalui prapemrosesan ---
def get_raw_data():
    URL = "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/students_performance/data.csv"
    try:
        raw_data = pd.read_csv(URL, sep=';')  
        print("File .csv berhasil diunduh dan terbaca.")
    except Exception as e:
        print(f"Ada error ketika mencoba mengunduh file .csv: {e}")

    # --- Mengubah tipe data sebagian besar kolom ke kategorikal  --- 
    # Mendefinisikan kolom-kolom yang perlu diubah tipe datanya 
    cat_cols = [
        'Marital_status',
        'Application_mode',
        'Application_order',
        'Course',
        'Daytime_evening_attendance',
        'Previous_qualification',
        'Nacionality',
        'Mothers_qualification', 
        'Fathers_qualification', 
        'Mothers_occupation',   
        'Fathers_occupation',   
        'Displaced',
        'Educational_special_needs', 
        'Debtor',
        'Tuition_fees_up_to_date',   
        'Gender',
        'Scholarship_holder',   
        'International'
    ]

    # Mengonversi tipe data kolom yang dimasukkan dalam 'cat_cols' ini ke kategorikal 
    cols_to_convert = [col for col in cat_cols if col in raw_data.columns]

    if cols_to_convert:
        raw_data[cols_to_convert] = raw_data[cols_to_convert].astype('object')
    else:
        print("Kolom yang akan diubah tidak ditemukan dalam dataframe. Tidak bisa menjalankan eksekusi...")

    # --- Mengganti tipe data 'Application_order' menjadi int64 --- 
    raw_data['Application_order'] = raw_data['Application_order'].astype('int64')

    # --- Memanggil fungsi 'preprocess_data' untuk melakukan prapemrosesan 'raw_data' --- 
    cleaned_data = preprocess_data(raw_data)

    # --- Handling Variable Skewing ---
    # Memeriksa variabel/fitur dengan skew melebihi 0.66
    MAX_SKEW_THRESHOLD = 0.66 
    skew_vals = cleaned_data.skew(numeric_only=True)
    skewed_cols_df = pd.DataFrame(skew_vals[(abs(skew_vals) > MAX_SKEW_THRESHOLD) & (skew_vals.index != 'Status')],
                    columns=['Skew']).sort_values(
                    by=['Skew'], ascending=False)

    # Melakukan log-transformation pada variabel yang memiliki skew melebihi 0.66
    display(skewed_cols_df.style.set_caption('Variabel atau fitur yang akan dilakukan log-transformation:'))

    for col in skewed_cols_df.index:
        cleaned_data[col] = np.log1p(cleaned_data[col])
        cleaned_data[col] = cleaned_data[col].replace([np.inf, -np.inf], np.nan)
        cleaned_data[col] = cleaned_data[col].fillna(0)

    # --- Undersampling data --- 
    # Membuat dataframe untuk masing-masing data berlabel (dibedakan berdasarkan mayoritas dan minoritas)
    major1_df = cleaned_data[(cleaned_data.Status == "Graduate")]
    major2_df = cleaned_data[(cleaned_data.Status == "Dropout")]
    minor_df = cleaned_data[(cleaned_data.Status == "Enrolled")] 

    # Melakukan undersampling terhadap data mayoritas menjadi 794 sampel
    undersampled_major1_df = resample(major1_df, n_samples=794, random_state=126)
    undersampled_major2_df = resample(major2_df, n_samples=794, random_state=126)

    # Menampilkan bentuk data setelah undersampling
    print("\nOverview sampel data setelah undersampling")
    print(f"Sampel data major1: {undersampled_major1_df.shape}")
    print(f"Sampel data major2: {undersampled_major2_df.shape}")

    # Menampilkan jumlah data setelah undersampling 
    print("\nJumlah sampel data setelah undersampling") 
    undersampled_data = pd.concat([undersampled_major1_df, undersampled_major2_df, minor_df], ignore_index=True)
    display(undersampled_data.Status)

    # Mengembalikan data hasil undersampling 
    return undersampled_data


# --- Fungsi untuk melakukan transformasi data ---
def transform_data(data):
    # --- Mendapatkan data yang belum dilakukan pembagian ---
    unsplitted_data = get_raw_data() 

    # --- Melakukan prapemrosesan terhadap data input dan data pelatihan --- 
    preprocessed_data = preprocess_data(data) 

    # --- Mengeliminasi kolom-kolom tidak relevan --- 
    # Menghilangkan kolom-kolom tidak relevan berdasarkan list dalam 'irrelevant_cols' 
    irrelevant_cols = [
        'Application_mode', 
        'Mothers_occupation',
        'Mothers_qualification',
        'Fathers_qualification',
        'Fathers_occupation',
        'GDP',
        'Unemployment_rate',
        'Inflation_rate',
        'Nacionality',
        'Educational_special_needs',
        'International'
    ]

    if irrelevant_cols: 
        unsplitted_data = unsplitted_data.drop(columns=irrelevant_cols, errors='ignore')
        preprocessed_data = preprocessed_data.drop(columns=irrelevant_cols, errors='ignore')  

    # --- Data splitting --- 
    # Membagi data ke dalam training set (train) dan testing set (test) (94% training, 6% testing)
    train_df, test_df = train_test_split(unsplitted_data, test_size=0.06, random_state=126, shuffle=True)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Menampilkan bentuk dari masing-masing variabel hasil data splitting
    print(train_df.shape) 
    print(test_df.shape) 

    # Mendefinisikan variabel X dan y untuk data pelatihan dan data uji 
    X_train = train_df.drop(columns="Status", axis=1)
    y_train = train_df["Status"]

    X_test = test_df.drop(columns="Status", axis=1)
    y_test = test_df["Status"]  

    # --- Melakukan encoding dan scaling terhadap fitur-fitur dalam data pelatihan dan data uji ---  
    # Fungsi helper untuk melakukan scaling fitur numerik dengan MinMaxScaler 
    def scale_data(cols, df, df_test=None):
        if df_test is not None:
            df = df.copy()
            df_test = df_test.copy()
            for col in cols:
                scaler = MinMaxScaler()
                X = np.asanyarray(df[col])
                X = X.reshape(-1, 1)
                scaler.fit(X)
                df["{}".format(col)] = scaler.transform(X)
                
                X_test = np.asanyarray(df_test[col])
                X_test = X_test.reshape(-1, 1)
                df_test["{}".format(col)] = scaler.transform(X_test)
                
            return df, df_test
            
        else:
            df = df.copy()
            for col in cols:
                scaler = MinMaxScaler()
                X = np.asanyarray(df[col])
                X = X.reshape(-1, 1)
                scaler.fit(X)
                df["{}".format(col)] = scaler.transform(X)
                
            return df


    # Fungsi helper untuk melakukan encoding fitur kategorikal dengan LabelEncoder 
    def encode_data(cols, df, df_test=None):
        if df_test is not None:
            df = df.copy()
            df_test = df_test.copy()
            for col in cols:
                encoder = LabelEncoder()
                encoder.fit(df[col])
                df["{}".format(col)] = encoder.transform(df[col])
                
                df_test["{}".format(col)] = encoder.transform(df_test[col])
                
            return df, df_test
            
        else:
            df = df.copy()
            for col in cols:
                encoder = LabelEncoder()
                encoder.fit(df[col])
                df["{}".format(col)] = encoder.transform(df[col])
                
            return df


    # Mendefinisikan fitur-fitur numerik dan kategorikal dalam data pelatihan dan data uji 
    cat_cols = X_train.select_dtypes(include="category").columns.tolist()
    num_cols = X_train.select_dtypes(include=["int", "float"]).columns.tolist()

    # Menjalankan proses encoding dan scaling 
    new_train_df, new_test_df = encode_data(cat_cols, X_train, preprocessed_data)
    new_train_df, new_test_df = scale_data(num_cols, new_train_df, new_test_df) 

    ordered_cols = new_train_df.columns.tolist()
    new_test_df = new_test_df[ordered_cols].copy()  

    # --- Transformasi mayoritas fitur ke dalam dimensi principal component --- 
    # Fitur-fitur yang dikelompokkan dalam 'pca_num_cols_1' (principal component dimension 1) 
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

    # Fitur-fitur yang dikelompokkan dalam 'pca_num_cols_2' (principal component dimension 2)
    pca_num_cols_2 = [
        'Previous_qualification_grade',
        'Change_units_without_evaluations',
        'Overall_approval_rate',
        'Overall_average_grade',
        'Overall_approved_to_credited_ratio',
        'Grade_vs_prev_qual'
    ]

    # Transformasi sebagian besar fitur pada data pelatihan ke dalam dua dimensi principal component
    # Transformasi data menjadi empat kolom dimensi principal component 1
    train_pca_df = new_train_df.copy().reset_index(drop=True)
    test_pca_df = new_test_df.copy().reset_index(drop=True)

    pca_1 = PCA(n_components=4, random_state=126)
    pca_1.fit(train_pca_df[pca_num_cols_1])
    princ_comp_1 = pca_1.transform(test_pca_df[pca_num_cols_1])
    test_pca_df[["Pc1_1", "Pc1_2", "Pc1_3", "Pc1_4"]] = pd.DataFrame(princ_comp_1, columns=["Pc1_1", "Pc1_2", "Pc1_3", "Pc1_4"])
    test_pca_df.drop(columns=pca_num_cols_1, axis=1, inplace=True)

    # Transformasi data menjadi dua kolom dimensi principal component 2
    pca_2 = PCA(n_components=2, random_state=126)
    pca_2.fit(train_pca_df[pca_num_cols_2])
    princ_comp_2 = pca_2.transform(test_pca_df[pca_num_cols_2])
    test_pca_df[["Pc2_1", "Pc2_2"]] = pd.DataFrame(princ_comp_2, columns=["Pc2_1", "Pc2_2"])
    test_pca_df.drop(columns=pca_num_cols_2, axis=1, inplace=True)

    # Mengembalikan data dengan lebih sedikit fitur 
    return test_pca_df


# --- Fungsi untuk memprediksi data input --- 
def predict_data(data=None):
    pred_data = data
    # Memuat file pretrained model 'model.joblib' yang sudah dilatih dengan algoritma RandomForest 
    model_filename = "model/random_forest.joblib" 
    label_filename = "model/encoder_target.joblib"

    final_pred_result = "Not available"
    loaded_model = None
    loaded_label = None

    try:
        loaded_model = joblib.load(model_filename)
        loaded_label = joblib.load(label_filename)
        print(f"File model {model_filename} dan {label_filename} berhasil dimuat.")
        # Menjalankan prediksi 'Status' akhir siswa berdasarkan data input yang diberikan 
        if loaded_model is not None and pred_data is not None:
            try:
                pred_result = loaded_model.predict(pred_data) 
                final_pred_result = loaded_label.inverse_transform(pred_result)[0]
                display(final_pred_result)
            except Exception as e:
                print(f"Terjadi kesalahan saat menjalankan prediksi: {e}")
                final_pred_result = "Prediction failed" # Apabila terjadi kesalahan saat menjalankan prediksi 

            # Mengembalikan nilai 'Status' ('final_pred_result') apa pun keadaannya 
            return final_pred_result

        else:
            # Perintah di bawah ini dijalankan apabila model atau data gagal diakses 
            print("\nGagal memuat model atau data masih tidak sesuai untuk digunakan. Prediksi tidak menghasilkan apa-apa...")

    except FileNotFoundError:
        print(f"File {model_filename} atau {label_filename} tidak ditemukan.")

    except Exception as e:
        print(f"Gagal memuat model: {e}")
    

# --- Fungsi untuk mendapatkan warna huruf berdasarkan kategori yang didapatkan dalam 'Status' --- 
def get_status_msg(status):
    if status == 'Graduate':
        return st.success('Congratulations! You have been graduated.', icon="🎓") 
    elif status == 'Dropout':
        return st.error('It is unfortunate that you have been dropout...', icon="😓")
    elif status == 'Enrolled':
        return st.info('You are still enrolled in this institution.', icon="ℹ️")
    else:
        return st.exception(f'{predict_data()}')


# --- Fungsi helper untuk membuat radio button --- 
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


# --- Fungsi helper untuk membuat selectbox option --- 
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


# --- Fungsi helper untuk formatting nilai Application_order ---
def format_app_order(option):
    return str(option)
    


# *****   FRONT-END VIEW OF 'APPS.PY'   *****
# --- Dashboard title --- 
st.title('Student Status Prediction App')


# --- Deskripsi dashboard --- 
st.text('''Welcome to the Jaya Jaya Institut Student Status Prediction App! Get valuable insight into your academic journey and likely final status, whether you're on track to graduate. Simply input your past semester grades, completed units, previous qualification scores, and admission test results. In moments, receive a clear picture of your potential graduation status here at Jaya Jaya Maju. This powerful tool is built using advanced machine learning models, meticulously trained and rigorously tested on high-quality, validated student data. Leverage cutting-edge technology to understand your path forward and stay informed about your graduation prospects! Try it now and gain clarity on your future at Jaya Jaya Maju. 
''')

st.write("---")

if 'data' not in st.session_state:
    st.session_state.data = {}
    
# --- Container formulir pengisian profil umum siswa dan performa penyelesaian kelas ---
with st.container():
    # Bagian profil siswa (student profile) 
    st.header("Student profile")
    st.text("Please enter your student profile first before continue proceeding.")
    st.session_state.data['Age_at_enrollment'] = int(st.number_input(label='Age during enrollment', value=st.session_state.data.get('Age_at_enrollment', 19)))
    st.caption("The age must be between 15 - 100 years old")

    st.session_state.data['Gender'] = create_radio(
        label="Gender",
        session_state_key='Gender',
        value_map=Gender_map,
        default_value=1 
        )

    st.session_state.data['Marital_status'] = create_selectbox(
        label="Marital status",
        session_state_key='Marital_status',
        value_map=Marital_status_map,
        default_value=1
        )

    st.session_state.data['Displaced'] = create_radio(
        label="Displaced?",
        session_state_key='Displaced',
        value_map=yes_no_display_map,
        default_value=0 
        )

    st.session_state.data['Debtor'] = create_radio(
        label="Debtor?",
        session_state_key='Debtor',
        value_map=yes_no_display_map,
        default_value=0 
        )

    st.session_state.data['Scholarship_holder'] = create_radio(
        label="Scholarship holder?",
        session_state_key='Scholarship_holder',
        value_map=yes_no_display_map,
        default_value=0 
        )

    st.session_state.data['Tuition_fees_up_to_date'] = create_radio(
        label="Tuition fees up to date?",
        session_state_key='Tuition_fees_up_to_date',
        value_map=yes_no_display_map,
        default_value=1 
        )

    st.session_state.data['Application_order'] = st.segmented_control(
        label='Application order', 
        options=Application_order_list, 
        format_func=format_app_order, 
        selection_mode='single', 
        default=6
        )
    st.caption("The choice must be from 0 (first choice) to 9 (last choice)")

    st.session_state.data['Admission_grade'] = st.number_input(label='Admission grade (0 - 200)', value=st.session_state.data.get('Admission_grade', 122))
    st.caption("Admission grade out of 200 scale")

    st.session_state.data['Previous_qualification'] = create_selectbox(
        label="Previous qualification",
        session_state_key='Previous_qualification',
        value_map=Previous_qualification_map,
        default_value=1
        )

    st.session_state.data['Previous_qualification_grade'] = st.number_input(
        label='Previous qualification grade (0 - 200)', 
        value=st.session_state.data.get('Previous_qualification_grade', 125)
        )
    st.caption("Previous qualification grade out of 200 scale") 

    st.write("---")

    # Bagian asesmen kinerja penyelesaian kelas (course performance assessment)
    st.header("Course performance assessment")
    st.text("Please input your grade on each semester, including your acquired curricular units.") 

    st.session_state.data['Course'] = create_selectbox(
        label="Course",
        session_state_key='Course',
        value_map=Course_map,
        default_value=9773
        )

    st.session_state.data['Daytime_evening_attendance'] = create_radio(
        label="Daytime/evening attendance",
        session_state_key='Attendance_options', 
        value_map=Attendance_map,
        default_value=1
        )
    

    # Kinerja penyelesaian kelas untuk semester 1
    st.subheader("Semester 1")
    st.session_state.data['Curricular_units_1st_sem_credited'] = st.number_input(
        label='Credited curricular units (1st)', 
        value=st.session_state.data.get('Curricular_units_1st_sem_credited', 0)
        )

    st.session_state.data['Curricular_units_1st_sem_enrolled'] = st.number_input(
        label='Enrolled curricular units (1st)', 
        value=st.session_state.data.get('Curricular_units_1st_sem_enrolled', 6)
        )

    st.session_state.data['Curricular_units_1st_sem_evaluations'] = st.number_input(
        label='Evaluated curricular units (1st)', 
        value=st.session_state.data.get('Curricular_units_1st_sem_evaluations', 7)
        )

    st.session_state.data['Curricular_units_1st_sem_approved'] = st.number_input(
        label='Approved curricular units (1st)', 
        value=st.session_state.data.get('Curricular_units_1st_sem_approved', 5)
        )

    st.session_state.data['Curricular_units_1st_sem_grade'] = st.number_input(
        label='Grade of curricular units (1st)', 
        value=st.session_state.data.get('Curricular_units_1st_sem_grade', 14)
        )

    st.session_state.data['Curricular_units_1st_sem_without_evaluations'] = st.number_input(
        label='Unevaluated curricular units (1st)', 
        value=st.session_state.data.get('Curricular_units_1st_sem_without_evaluations', 0)
        )



    # Kinerja penyelesaian kelas untuk semester 2
    st.subheader("Semester 2")
    st.session_state.data['Curricular_units_2nd_sem_credited'] = st.number_input(
        label='Credited curricular units (2nd)', 
        value=st.session_state.data.get('Curricular_units_2nd_sem_credited', 0)
        )

    st.session_state.data['Curricular_units_2nd_sem_enrolled'] = st.number_input(
        label='Enrolled curricular units (2nd)', 
        value=st.session_state.data.get('Curricular_units_2nd_sem_enrolled', 6)
        )

    st.session_state.data['Curricular_units_2nd_sem_evaluations'] = st.number_input(
        label='Evaluated curricular units (2nd)', 
        value=st.session_state.data.get('Curricular_units_2nd_sem_evaluations', 8)
        )

    st.session_state.data['Curricular_units_2nd_sem_approved'] = st.number_input(
        label='Approved curricular units (2nd)', 
        value=st.session_state.data.get('Curricular_units_2nd_sem_approved', 5)
        )

    st.session_state.data['Curricular_units_2nd_sem_grade'] = st.number_input(
        label='Grade of curricular units (2nd)', 
        value=st.session_state.data.get('Curricular_units_2nd_sem_grade', 13)
        )

    st.session_state.data['Curricular_units_2nd_sem_without_evaluations'] = st.number_input(
        label='Unevaluated curricular units (2nd)', 
        value=st.session_state.data.get('Curricular_units_2nd_sem_without_evaluations', 0)
        )


# Expander layout untuk menampilkan data yang sudah diinput dalam bentuk tabel dataframe
with st.expander("View your inputted data"):
    display_data = pd.DataFrame([st.session_state.data])
    st.dataframe(data=display_data, width=1920, height=96)


# Tombol interaktif untuk menjalankan prediksi status dropout siswa berdasarkan data yang sudah diinput 
if st.button("Predict your data"):
    student_data = pd.DataFrame([st.session_state.data])
    new_student_data = transform_data(student_data)
    pred_status = predict_data(data=new_student_data)

    with st.expander("View the preprocessed data"):
        st.dataframe(data=preprocess_data(student_data), width=1920, height=96)
    
    # Menampilkan pesan interaktif berdasarkan status akhir siswa 
    get_status_msg(pred_status)

    











    


    

    

