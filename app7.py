import base64
import hashlib
import pickle
import secrets
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model

# Streamlit App
st.set_page_config(page_title="GDM Prediction", layout="wide")
# st.title("Welcome to GDM Prediction Portal")

# Cache the loading of models and scalers for performance optimization
@st.cache_resource
def load_resources():
    ensemble_model = pickle.load(open('sources/ensemble_model.pkl', 'rb'))
    cnn_model = load_model('sources/cnn_model.h5')
    scaler = joblib.load('sources/scaler.pkl')
    return ensemble_model, cnn_model, scaler

ensemble_model, cnn_model, scaler = load_resources()

# Database setup with context manager for better resource handling
def init_db():
    with sqlite3.connect('users1.db') as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT,
                last_login TEXT,
                session_token TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS history (
                username TEXT, 
                input_data TEXT, 
                result TEXT, 
                timestamp TEXT
            )
        ''')
        conn.commit()

init_db()

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Utility Functions
def hash_password(password):
    salt = secrets.token_hex(8)  # Add a unique salt for each password
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${hashed}"  # Store salt alongside the hash

def verify_password(stored_password, provided_password):
    salt, hashed = stored_password.split('$')
    return hashlib.sha256((salt + provided_password).encode()).hexdigest() == hashed

def authenticate(username, password):
    """Authenticate user credentials."""
    with sqlite3.connect('users1.db') as conn:
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        return result and verify_password(result[0], password)

def signup(username, password):
    """Sign up a new user."""
    hashed_password = hash_password(password)
    with sqlite3.connect('users1.db') as conn:
        c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, password, last_login, session_token) VALUES (?, ?, ?, ?)",(username, hashed_password, datetime.now().isoformat(), None)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def generate_session_token():
    """Generate a unique session token."""
    return secrets.token_hex(16)

def update_login_time(username, session_token):
    """Update login time and store session token."""
    with sqlite3.connect('users1.db') as conn:
        c = conn.cursor()
    c.execute("UPDATE users SET last_login = ?, session_token = ? WHERE username = ?", (datetime.now().isoformat(), session_token, username))
    conn.commit()

def get_user_by_session_token(session_token):
    """Retrieve user by session token."""
    with sqlite3.connect('users1.db') as conn:
        c = conn.cursor()
    c.execute("SELECT username FROM users WHERE session_token = ?", (session_token,))
    user = c.fetchone()
    return user[0] if user else None

def clear_session_token(username):
    """Clear the session token for logout."""
    with sqlite3.connect('users1.db') as conn:
        c = conn.cursor()
    c.execute("UPDATE users SET session_token = NULL WHERE username = ?", (username,))
    conn.commit()

def predict_combined(input_data, ensemble_weight=0.3, cnn_weight=0.7):
    """Predict GDM using the ensemble and CNN models."""
    input_data_scaled = scaler.transform(input_data)
    ensemble_pred_proba = ensemble_model.predict_proba(input_data_scaled)[0][1]
    cnn_input = input_data_scaled[..., np.newaxis]
    cnn_pred_proba = cnn_model.predict(cnn_input, verbose=0)[0][0]
    combined_proba = (ensemble_weight * ensemble_pred_proba) + (cnn_weight * cnn_pred_proba)
    final_prediction = "GDM" if combined_proba > 0.5 else "Non-GDM"
    return {
        "ensemble_proba": ensemble_pred_proba,
        "cnn_proba": cnn_pred_proba,
        "combined_proba": combined_proba,
        "final_prediction": final_prediction
    }

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-position: center;
    background-size: cover;
    margin-top:10px;
    margin-bottom:10px;
    padding: 10px;
    width: auto;
    border-radius: 10px;
    font-size: 20px;
    
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('image.png')

def boxed_text(text):
    st.markdown(
        f"""
        <div style="
            border: 2px solid #d3d3d3; 
            border-radius: 10px; 
            background-color: #f0f0f0; 
            padding: 15px; 
            font-size: 20px;
            font-family: 'Times New Roman', serif;
            color: black;">
            {text}
        </div>
        """, 
        unsafe_allow_html=True
    )
def home():
    """Home page."""
    st.markdown("<font size='32' color='Black' font-weight='bold'  face='Times New Roman'>Gestational Diabetes Detection</font>", unsafe_allow_html=True)

    st.subheader("Introduction to Gestational Diabetes")
    boxed_text("Gestational Diabetes Mellitus (GDM) poses severe health risks to both mother and child, making early detection crucial to preventing complications. Traditional diagnostic methods, such as the Oral Glucose Tolerance Test (OGTT), are time-consuming and resource-intensive, highlighting the need for a more efficient, AI-driven predictive system. This project leverages deep learning and ensemble machine learning models to enhance early-stage GDM prediction, facilitating timely intervention and improved maternal healthcare outcomes.")

    st.subheader("Why Early Diagnosis Matters")
    boxed_text("Early detection of Gestational Diabetes Mellitus (GDM) is crucial in preventing severe health risks for both the mother and baby. If left undiagnosed or unmanaged, GDM can lead to complications such as preeclampsia, high blood pressure, and an increased likelihood of developing Type 2 diabetes in the mother. It can also cause excessive fetal growth (macrosomia), which raises the risk of birth injuries and the need for a cesarean section. For the baby, uncontrolled GDM can result in neonatal hypoglycemia, respiratory distress, and a higher chance of obesity and diabetes later in life. Early diagnosis allows for timely medical intervention, including lifestyle modifications, dietary adjustments, and, if necessary, insulin therapy to maintain stable blood sugar levels. Additionally, it reduces the need for intensive medical care, lowering the risk of neonatal intensive care unit (NICU) admissions and overall healthcare costs. By implementing early screening and AI-driven predictive models, healthcare providers can enhance maternal and fetal outcomes, ensuring a healthier pregnancy and long-term well-being for both mother and child.")

    st.subheader("Our Mission")
    boxed_text("The goal of this project is to create a machine learning-based GDM detection system that can forecast GDM risk and offer tailored suggestions. Our solution uses historical data and machine learning algorithms to increase early detection rates, improve patient engagement, and lower problems associated with GDM. A novel ensemble-based deep learning framework for early GDM prediction is proposed in this paper. Traditional classifiers like Support Vector Machines (SVM), Random Forests (RF), and Gradient Boosting Machines (GBM) are combined with Convolutional Neural Networks (CNN) in this model. The study uses the SMOTE-ENN technique to address class imbalance and ensure that GDM situations are fairly represented. This project intends to create a reliable and scalable prediction model for early GDM identification with sophisticated machine learning techniques. This strategy can enhance maternal and newborn health outcomes, decrease problems, and ensure prompt interventions.")

def aboutus():
    st.subheader("About Us")
    boxed_text(" P. Kavya Sri (21501A05E5), mail: 21501a05e5@pvpsit.ac.in")
    st.write("")
    boxed_text("K. Venkata Nitchaya Reddy (21501A05J3), mail: 21501a05j3@pvpsit.ac.in")
    st.write("")
    boxed_text("R. Chaitanya Reddy (21501A05F0), mail: 21501a05f0@pvpsit.ac.in")
    st.write("")
    boxed_text("P. Naveen Kumar (21501A05D3), mail: 21501a05d3@pvpsit.ac.in")
    st.write("")
    boxed_text("R. Pavan (21501A05E6), mail: 21501a05e6@pvpsit.ac.in")
    st.write("")
    boxed_text("Department of Computer Science and Engineering, Prasad V. Potluri Siddhartha Institute of Technology, (Permanently affiliated to JNTU-Kakinada, Approved by AICTE) (An NBA &amp; NAAC accredited and ISO 9001:2005 certified institute) Kanuru, Vijayawada-520 007") 
def login():
    """Login function."""
    st.subheader("Login")
    st.markdown('''

        <style>
        .stApp {
                color: black;
                text-align: center;
            }
        }''',
                unsafe_allow_html=True)
    with st.form(key='myform', clear_on_submit=True):
        css="""
        <style>
            [data-testid="stForm"] {
                background-color: lightblue;
                color:black;
                padding: 20px;
                border-radius: 10px;
                font-size: 16px;
                margin:10px;
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
                text-align: center;
                transition: background-color 0.5s ease;
                cursor: pointer;
                width: 50%;
                margin: auto;
                
            }
        </style>
        """
        st.write(css, unsafe_allow_html=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button(label='Login')
        if submit_button:
            user = authenticate(username, password)
            if user:
                session_token = generate_session_token()
                update_login_time(username, session_token)
                st.session_state.username = username
                st.session_state.session_token = session_token
                st.query_params['session_token'] = session_token
                #st.query_params[st.session_state.session_token] = session_token
                st.success(f"Welcome, {username}! Redirecting to Dashboard...")
                st.session_state.navigation = "Dashboard"
            else:
                st.error("Invalid credentials.")


def signuppage():
    """Signup function."""
    st.subheader("Create an Account")
    st.markdown('''

        <style>
        .stApp {
                color: black;
                text-align: center;
            }
        }'''
                ,
                unsafe_allow_html=True)
    with st.form(key='myform', clear_on_submit=True):
        css="""
        <style>
            [data-testid="stForm"] {
                background-color: lightblue;
                color:black;
                padding: 20px;
                border-radius: 10px;
                font-size: 16px;
                margin:10px;
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
                text-align: center;
                transition: background-color 0.5s ease;
                cursor: pointer;
                width: 50%;
                margin: auto;
                
            }
        </style>
        """
        st.write(css, unsafe_allow_html=True)
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit_button = st.form_submit_button(label='Create Account')
        if submit_button:
            if new_username and new_password:
                success = signup(new_username, new_password)
                if success:
                    session_token = generate_session_token()
                    update_login_time(new_username, session_token)
                    st.session_state.username = new_username
                    st.session_state.session_token = session_token
                    st.query_params[session_token] = session_token
                    st.success(f"Account created successfully! Redirecting to Dashboard...")
                    st.session_state.navigation = "Dashboard"
                else:
                    st.error("Username already exists. Please choose a different username.")
            else:
                st.error("Please fill in all fields.")
    st.markdown("""
        <style>
            button.step-up {display: none;}
            button.step-down {display: none;}
            div[data-baseweb] {border-radius: 4px;}
        </style>""",
        unsafe_allow_html=True)

def gdmdetector():
    """GDM detection function."""
    with sqlite3.connect('users1.db') as conn:
        c = conn.cursor()
    st.header("GDM Detection")
    st.markdown('''

        <style>
        .stApp {
                color: black;
                text-align: center;
            }
        }'''
                ,
                unsafe_allow_html=True)
    # Collect input data
    with st.form(key='myform', clear_on_submit=True):
        css="""
        <style>
            [data-testid="stForm"] {
                background-color: lightblue;
                color:black;
                padding: 20px;
                border-radius: 10px;
                font-size: 16px;
                margin:10px;
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
                text-align: center;
                transition: background-color 0.5s ease;
                cursor: pointer;
                width: 50%;
                margin: auto;
                
            }
        </style>
        """
        st.write(css, unsafe_allow_html=True)
        
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gestation = st.number_input("Gestation in Weeks", min_value=0, max_value=50, value=0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
        hdl = st.number_input("HDL Cholesterol", min_value=0.0, max_value=100.0, value=50.0)
        family_history = st.selectbox("Family History of Diabetes (Yes=1, No=0)", [0, 1])
        pcos = st.selectbox("PCOS (Yes=1, No=0)", [0, 1])
        dia_bp = st.number_input("Diastolic Blood Pressure", min_value=0.0, max_value=200.0, value=80.0)
        ogtt = st.number_input("OGTT (mg/dL)", min_value=0.0, max_value=500.0, value=140.0)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, value=12.0)
        prediabetes = st.selectbox("History of Prediabetes (Yes=1, No=0)", [0, 1])
        submit_button = st.form_submit_button(label='Predict GDM')
        if submit_button:
            input_data = np.array([[age, gestation, bmi, hdl, family_history, pcos, dia_bp, ogtt, hemoglobin, prediabetes]])
            prediction_result = predict_combined(input_data)
            st.subheader("Prediction Results")
            st.write(f"Ensemble Model Probability: {prediction_result['ensemble_proba']:.2f}")
            st.write(f"CNN Model Probability: {prediction_result['cnn_proba']:.2f}")
            st.write(f"Combined Probability: {prediction_result['combined_proba']:.2f}")
            st.write(f"Final Prediction: {prediction_result['final_prediction']}")
            st.success(f"Prediction: {prediction_result['final_prediction']}")
            c.execute("INSERT INTO history (username, input_data, result, timestamp) VALUES (?, ?, ?, ?)",(st.session_state.username, str(input_data.tolist()), str(prediction_result['final_prediction']), datetime.now().isoformat()))
            conn.commit()

def dashboard():
    """Dashboard function."""
    with sqlite3.connect('users1.db') as conn:
        c  = conn.cursor()
    st.header("User Dashboard")
    # Apply custom CSS for styling
    st.markdown(
        """
        <style>
        table {
            width: 75%;
            border-collapse: collapse;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
            text-align: center;
            background-color: lightblue;
            color:black;
        }
        th {
            background-color: #4CAF50;
            color: white;
            text-align: center;
            padding: 12px;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
            text-align: center
        }
        tr:hover {
            background-color: #f5f5f5;

        }
        </style>
        """,
        unsafe_allow_html=True
    )

    c.execute("SELECT input_data, result, timestamp FROM history WHERE username = ?", (st.session_state.username,))
    history = c.fetchall()
    df = pd.DataFrame(history, columns=["Input Data", "Result", "Timestamp"])

    # Display the styled table
    st.dataframe(df, use_container_width=True)
    
    if history==[]:
        st.write("No history found.")
    if st.button("Clear All Data"):
        c.execute("DELETE FROM history WHERE username = ?", (st.session_state.username,))
        conn.commit()
        st.success("All history cleared.")
    if st.button("Debug Users Table"):
        with sqlite3.connect('users1.db') as conn:
            c = conn.cursor()
            c.execute("SELECT username, password FROM users")
            users = c.fetchall()
            st.write("Users in the database:")
            st.table(users)

def logout():
    clear_session_token(st.session_state.username)
    st.session_state.username = None
    st.session_state.session_token = None
    st.query_params["session_token"] = "/"  # Clear session token from query params
    st.success("Logged out successfully.")
# Streamlit App
#st.set_page_config(page_title="GDM Prediction", layout="wide")
#st.title("Welcome to GDM Prediction Portal")

# Session state initialization
if 'session_token' not in st.session_state:
    st.session_state.session_token = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'navigation' not in st.session_state:
    st.session_state.navigation = None

#params = st.query_params()
# Restore session from query params
if st.session_state.session_token is None:
    #params = st.query_params()
    session_token = st.query_params.get("session_token", [None])[0]
    if session_token:
        username = get_user_by_session_token(session_token)
        if username:
            st.session_state.username = username
            st.session_state.session_token = session_token
            st.session_state.navigation = "Dashboard"  # Automatically navigate to Dashboard
            st.query_params['session_token'] = session_token
            st.success(f"Welcome back, {username}!")
            #st.experimental_set_query_params(session_token=session_token)
            


# Navigation logic
if st.session_state.session_token is None:
    # Login/Sign-Up Interface
    option = option_menu(
            menu_title=None,  # required
            options=["Home", "Login", "Sign Up", "About US"],  # required
            icons=["house", "login","person_add","envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
    st.markdown('''<style>
        .option_menu {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                padding: 20px;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
                margin-top: 15cm;
                margin: 0 auto;
        }
        </style>''', unsafe_allow_html=True)
    if option == "Home":
        home()
    elif option == "About US":
        aboutus()
    elif option == "Sign Up":
        signuppage()
    elif option == "Login":
        login()
else:
    # Logged-in view
    option = option_menu(
            menu_title=None,  # required
            options=["Home", "Dashboard", "GDM Detector", "Logout","About US"],  # required
            icons=["house", "medical_information","health_and_safety","logout", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
    if option == "Home":
        home()
    elif option == "About US":
        aboutus()
    elif option == "GDM Detector":
        gdmdetector()
    elif option == "Dashboard":
        dashboard()
    elif option == "Logout":
        logout()
